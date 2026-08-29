"""Politici smart-search web application.

Combined multi-politician archive: one process serves every politician in
config/*.json, the visitor picks one from a dropdown. Each request carries a
`person` slug; only that politician's index is searched.

- GET  /api/persons   politicians available in this instance (drives the dropdown)
- POST /api/search     semantic search over one politician's transcripts
- POST /api/ask        RAG: retrieve relevant fragments + LLM answer with citations
- GET  /api/stats      per-politician index stats (?person=<slug>)
- GET  /api/statistics per-politician usage stats (?person=<slug>)
- GET  /media/{f}      serve local audio/video with seekable playback
- /                    single-page frontend (static/index.html)

Same architecture as abo-ali-search; sources are official Tweede Kamer
verslagen (transcript_source=official), 1995-2013 Handelingen, and YouTube
ASR transcripts. The parliamentary transcript pool + debate video under
SHARED_DIR are shared across every politician.
"""
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from embedder import Embedder
from pipeline_config import load_all_configs

BASE = Path(__file__).parent
CONFIGS = load_all_configs()  # {slug: cfg} for every config/<slug>.json

# PERSON env var pins this process to one politician: only their index is
# loaded and /api/persons returns just them, so a legacy single-person
# systemd unit (wilders-search / yesilgoz-search) started from this same
# checkout behaves and uses memory exactly as before -- no dropdown, no
# sibling index loaded. Unset (the combined politicus-search unit) -> serve
# every config/*.json.
_PINNED = os.environ.get("PERSON") or None
SERVED_SLUGS = [_PINNED] if _PINNED in CONFIGS else list(CONFIGS)
# A request without an explicit person falls back to this: PERSON pin, else
# the DEFAULT_PERSON env (the combined app's landing politician), else the
# first slug alphabetically.
_DEFAULT_ENV = os.environ.get("DEFAULT_PERSON") or None
DEFAULT_SLUG = (_PINNED if _PINNED in CONFIGS else
               _DEFAULT_ENV if _DEFAULT_ENV in CONFIGS else
               (SERVED_SLUGS[0] if SERVED_SLUGS else None))
# Debat Direct video is shared across everyone -- resolve it from any config.
DG_DIR = next(iter(CONFIGS.values()))["_paths"]["debatgemist"] if CONFIGS else None
STATS_DB = BASE / "stats.sqlite"  # outside index/ so re-indexing keeps history

app = FastAPI(title="Politici Archief")

# ---------------------------------------------------------------- index state
# _state["persons"][slug] -> {db, matrix, dates, person_mask, videos, chunks,
#                             person_chunks}
# _state shared: embedder, device, dg_windows, llm_base_url (cache)
_state: dict = {"persons": {}}


def _load_person(slug: str, cfg: dict, device: str) -> dict | None:
    index_dir = cfg["_paths"]["index"]
    db_path = index_dir / "index.sqlite"
    emb_path = index_dir / "embeddings.npy"
    if not db_path.exists() or not emb_path.exists():
        print(f"  [skip] {slug}: no index at {index_dir} (config kept, not served)")
        return None
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    matrix = torch.from_numpy(np.load(emb_path)).to(device)  # (n, 1024) fp16
    # "only statements by X" mask -- same both-names convention as
    # tk_parse.person_speaks(), so "Bosma" doesn't also match "Bosman" etc.
    m = cfg["tk"]["match"]
    achter, voor = (m.get("achternaam") or "").lower(), (m.get("voornaam") or "").lower()
    rows = db.execute(
        "SELECT c.id, c.speaker, v.upload_date FROM chunks c JOIN videos v ON v.video = c.video ORDER BY c.id"
    ).fetchall()
    dates = np.array([int(r["upload_date"] or 0) for r in rows], dtype=np.int64)

    def _is_person(speaker: str) -> bool:
        n = (speaker or "").lower()
        return bool(achter) and achter in n and (not voor or voor in n)

    person = np.array([_is_person(r["speaker"]) for r in rows])
    entry = dict(
        db=db,
        matrix=matrix,
        dates=torch.from_numpy(dates).to(device),
        person_mask=torch.from_numpy(person).to(device),
        videos=db.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
        chunks=int(matrix.shape[0]),
        person_chunks=int(person.sum()),
    )
    print(f"  [ok]   {slug} ({cfg['person']}): {entry['chunks']} chunks, "
          f"{entry['person_chunks']} by {cfg['tk']['match']['achternaam']}")
    return entry


@app.on_event("startup")
def load_index():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    what = f"pinned to '{_PINNED}'" if _PINNED else f"{len(SERVED_SLUGS)} politician(s)"
    print(f"loading index(es) on {device} ({what}):")
    for slug in SERVED_SLUGS:
        entry = _load_person(slug, CONFIGS[slug], device)
        if entry is not None:
            _state["persons"][slug] = entry

    # Debat Direct video mapping: wallclock -> (file, video_start). Shared
    # pool, loaded once (keyed by date+slug, reused by anyone who spoke).
    dg_windows = []
    dg_state_path = DG_DIR / "state.json" if DG_DIR else None
    if dg_state_path and dg_state_path.exists():
        for fname, info in json.loads(dg_state_path.read_text()).items():
            try:
                t0 = datetime.strptime(info["video_start"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
                t1 = datetime.strptime(info["video_end"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            except (KeyError, ValueError):
                continue
            if (DG_DIR / fname).exists():
                dg_windows.append((t0, t1, fname))
    dg_windows.sort()
    _state.update(
        embedder=Embedder(device=device, max_length=512),
        device=device,
        dg_windows=dg_windows,
    )
    print(f"index ready: {len(_state['persons'])}/{len(SERVED_SLUGS)} politicians served, "
          f"{len(dg_windows)} debate videos, default person '{DEFAULT_SLUG}'")


def resolve_slug(person: str | None) -> str:
    """Request person -> a slug that is actually loaded, or HTTP 400."""
    slug = person or DEFAULT_SLUG
    if slug not in _state["persons"]:
        raise HTTPException(400, f"unknown or unavailable person: {person!r}")
    return slug


# ------------------------------------------------------------- usage tracking
# STATS_DB is one shared file on disk (also shared with any legacy
# single-person unit still running from this checkout). Every row is tagged
# with `person` (the config slug) so stats are always per-politician.
def _stats_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(STATS_DB, timeout=10)
    conn.execute("""CREATE TABLE IF NOT EXISTS queries (
        ts TEXT DEFAULT (datetime('now')), mode TEXT, ip TEXT, query TEXT, person TEXT)""")
    try:  # upgrade path for pre-existing stats.sqlite files without this column
        conn.execute("ALTER TABLE queries ADD COLUMN person TEXT")
    except sqlite3.OperationalError:
        pass
    return conn


def client_ip(request: Request) -> str:
    ip = request.headers.get("x-real-ip") or ""
    if not ip:
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip()
    return ip or (request.client.host if request.client else "unknown")


def log_query(request: Request, slug: str, mode: str, query: str):
    try:
        with _stats_conn() as conn:
            conn.execute(
                "INSERT INTO queries (mode, ip, query, person) VALUES (?,?,?,?)",
                (mode, client_ip(request), query[:500], slug),
            )
    except Exception as e:  # stats must never break search
        print(f"stats logging failed: {e}")


@app.get("/api/statistics")
def api_statistics(person: str | None = None):
    slug = resolve_slug(person)
    with _stats_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM queries WHERE person=?", (slug,)).fetchone()[0]
        by_mode = dict(conn.execute(
            "SELECT mode, COUNT(*) FROM queries WHERE person=? GROUP BY mode", (slug,)).fetchall())
        unique_ips = conn.execute(
            "SELECT COUNT(DISTINCT ip) FROM queries WHERE person=?", (slug,)).fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ip) FROM queries WHERE person=? AND date(ts) = date('now')",
            (slug,)).fetchone()
        per_day = conn.execute("""
            SELECT date(ts) d, COUNT(*), COUNT(DISTINCT ip)
            FROM queries WHERE person=? AND ts >= datetime('now','-30 days')
            GROUP BY d ORDER BY d""", (slug,)).fetchall()
        first = conn.execute(
            "SELECT MIN(date(ts)) FROM queries WHERE person=?", (slug,)).fetchone()[0]
    return {
        "person": CONFIGS[slug]["person"],
        "slug": slug,
        "total_queries": total,
        "search_queries": by_mode.get("search", 0),
        "ask_queries": by_mode.get("ask", 0),
        "unique_ips": unique_ips,
        "today_queries": today[0],
        "today_unique_ips": today[1],
        "since": first,
        "per_day": [{"date": d, "queries": q, "unique_ips": u} for d, q, u in per_day],
    }


# ------------------------------------------------------------------ retrieval
def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def yt_link(url: str, start: float) -> str:
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={max(int(start) - 2, 0)}s"


def dg_media(wallclock: str) -> tuple[str, str]:
    """Resolve a TK chunk wallclock to (media_file, media_url) of a
    downloaded Debat Direct video, or ('', '')."""
    if not wallclock:
        return "", ""
    try:
        wc = datetime.fromisoformat(wallclock)
    except ValueError:
        return "", ""
    for t0, t1, fname in _state.get("dg_windows", ()):
        if t0 <= wc <= t1:
            off = max(int((wc - t0).total_seconds()) - 2, 0)
            return fname, f"/media/{fname}#t={off}"
    return "", ""


def retrieve(slug: str, query: str, top_k: int, date_from: str | None, date_to: str | None,
             only_person: bool = False):
    st = {**_state, **_state["persons"][slug]}  # shared keys + this person's index
    q = st["embedder"].encode([query]).to(device=st["device"], dtype=st["matrix"].dtype)  # (1, 1024)
    scores = (st["matrix"] @ q.T).squeeze(1).float()  # (n,)
    neg = torch.tensor(-1.0, device=scores.device)
    if date_from:
        scores = torch.where(st["dates"] >= int(date_from), scores, neg)
    if date_to:
        scores = torch.where(st["dates"] <= int(date_to), scores, neg)
    if only_person:
        scores = torch.where(st["person_mask"], scores, neg)
    k = min(top_k, scores.shape[0])
    vals, idx = torch.topk(scores, k)
    results = []
    for score, cid in zip(vals.tolist(), idx.tolist()):
        if score < 0:
            continue
        row = st["db"].execute(
            """SELECT c.*, v.title, v.url, v.upload_date, v.media_file, v.source, v.transcript_source
               FROM chunks c JOIN videos v ON v.video = c.video WHERE c.id = ?""",
            (cid,),
        ).fetchone()
        d = row["upload_date"] or ""
        is_yt = (row["source"] or "").startswith("youtube")
        media_file = row["media_file"]
        media_url = f"/media/{media_file}#t={max(int(row['start']) - 2, 0)}" if media_file else ""
        if not media_file and row["source"] == "tk_verslag" and "wallclock" in row.keys():
            media_file, media_url = dg_media(row["wallclock"])
        results.append({
            "score": round(score, 4),
            "video": row["video"],
            "title": row["title"],
            "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d,
            "speaker": row["speaker"],
            "source": row["source"],
            "transcript_source": row["transcript_source"],
            "start": row["start"],
            "end": row["end"],
            # ob_handelingen has no timestamps; start/end are ordinal only
            "start_fmt": "" if row["source"] == "ob_handelingen" else fmt_time(row["start"]),
            "end_fmt": "" if row["source"] == "ob_handelingen" else fmt_time(row["end"]),
            "text": row["text"],
            "youtube_url": yt_link(row["url"], row["start"]) if is_yt else "",
            "source_url": "" if is_yt else (row["url"] or ""),
            "media_url": media_url,
            "media_file": media_file,
        })
    return results


class SearchReq(BaseModel):
    query: str
    person: str | None = None
    top_k: int = 20
    date_from: str | None = None  # YYYYMMDD
    date_to: str | None = None
    only_person: bool = False


@app.post("/api/search")
def api_search(req: SearchReq, request: Request):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    slug = resolve_slug(req.person)
    log_query(request, slug, "search", req.query)
    return {"results": retrieve(slug, req.query, min(req.top_k, 100),
                                req.date_from, req.date_to, req.only_person)}


# ------------------------------------------------------------------------ RAG
# Answer generation uses any OpenAI-compatible endpoint (llama.cpp, LM Studio,
# vLLM, Ollama...). Configure with env vars:
#   LLM_BASE_URL  e.g. http://localhost:1234/v1   (default: auto-discover the
#                 scrib-r llama.cpp container and use it)
#   LLM_MODEL_ID  default: qwen3-8b
#   LLM_API_KEY   default: none (local servers ignore it)
import subprocess

LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3-8b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "none")
# GPT-OSS/harmony reasoning effort for the RAG answer step (low/medium/high).
# NOT the Qwen3 "/no_think" convention this used to rely on — that has no
# effect on harmony-format models. low = fast, minimal thinking.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")


def llm_base_url() -> str | None:
    url = os.environ.get("LLM_BASE_URL")
    if url:
        return url.rstrip("/")
    cached = _state.get("llm_base_url")
    if cached:
        return cached
    try:  # auto-discover the scrib-r llama.cpp container
        ip = subprocess.run(
            ["docker", "inspect", "-f",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
             "scrib-r-backend-llama-1"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if ip:
            _state["llm_base_url"] = f"http://{ip}:8080/v1"
            return _state["llm_base_url"]
    except Exception:
        pass
    return None


def answer_system(person: str) -> str:
    return f"""Je bent een onderzoeksassistent voor een doorzoekbaar archief van alles wat
{person} in het openbaar heeft gezegd: officiële Tweede Kamer-verslagen en
transcripties van openbare video's. Je krijgt genummerde fragmenten uit dit
archief (ASR-fragmenten kunnen transcriptiefouten bevatten — ga daar slim mee om).
Beantwoord de vraag van de gebruiker uitsluitend op basis van de fragmenten:
- Zeg wat er gezegd is en wanneer (datum en tijdstip binnen het debat/de video).
- Zet na elke bewering het bronnummer tussen blokhaken, bijv. [3].
- Fragmenten van andere sprekers zijn context; schrijf niets aan {person} toe
  dat een andere spreker zei.
- Staat het antwoord niet in de fragmenten, zeg dat dan expliciet.
- Antwoord in helder, beknopt Nederlands. Blijf feitelijk en neutraal."""


class AskReq(BaseModel):
    question: str
    person: str | None = None
    top_k: int = 16
    date_from: str | None = None
    date_to: str | None = None
    only_person: bool = False


@app.post("/api/ask")
def api_ask(req: AskReq, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    slug = resolve_slug(req.person)
    log_query(request, slug, "ask", req.question)
    sources = retrieve(slug, req.question, min(req.top_k, 60), req.date_from, req.date_to, req.only_person)
    answer, error = None, None
    base_url = llm_base_url()
    if not base_url:
        return {"answer": None, "error": "no_llm", "sources": sources}
    try:
        import httpx
        excerpts = "\n\n".join(
            f"[{i+1}] Bron: {s['title']} | Datum: {s['date']} | Tijd: {s['start_fmt']}–{s['end_fmt']} | Spreker: {s['speaker'] or 'onbekend'}\n{s['text']}"
            for i, s in enumerate(sources)
        )
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL_ID,
                "max_tokens": 2048,
                "temperature": 0.3,
                "reasoning_effort": LLM_REASONING_EFFORT,
                "messages": [
                    {"role": "system", "content": answer_system(CONFIGS[slug]["person"])},
                    {"role": "user",
                     "content": f"Fragmenten:\n\n{excerpts}\n\nVraag: {req.question}"},
                ],
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        answer = message.get("content") or ""
        # strip <think>...</think> reasoning blocks some local models emit inline
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        if not answer and message.get("reasoning_content"):
            # reasoning model spent its whole token budget thinking instead of
            # answering — surface as an error instead of silently returning "".
            error = "empty_answer_reasoning_only"
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        if cited:
            for i, s in enumerate(sources):
                s["cited"] = (i + 1) in cited
    except Exception as e:
        error = str(e)
    return {"answer": answer, "error": error, "sources": sources}


def _person_summary(slug: str) -> dict:
    p = _state["persons"][slug]
    cfg = CONFIGS[slug]
    return {
        "slug": slug,
        "person": cfg["person"],
        "hero_image": cfg.get("hero_image"),
        "videos": p["videos"],
        "chunks": p["chunks"],
        "person_chunks": p["person_chunks"],
    }


@app.get("/api/persons")
def api_persons():
    """Every politician this instance can serve, for the frontend dropdown."""
    return {
        "default": DEFAULT_SLUG if DEFAULT_SLUG in _state["persons"] else next(iter(_state["persons"]), None),
        "persons": sorted((_person_summary(s) for s in _state["persons"]),
                          key=lambda d: d["person"]),
    }


@app.get("/api/stats")
def api_stats(person: str | None = None):
    return _person_summary(resolve_slug(person))


# ---------------------------------------------------------------------- media
RANGE_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _ranged_response(path: Path, media_type: str, request: Request) -> StreamingResponse:
    """Serve `path` honoring an incoming Range header. Debate recordings run
    for hours; without real 206/Content-Range support the <video> element
    can't seek at all -- Chrome silently clamps any currentTime jump back
    to 0 and the whole file has to be fetched just to reach frame 1."""
    file_size = path.stat().st_size
    start, end = 0, file_size - 1
    status_code = 200

    range_header = request.headers.get("range")
    if range_header:
        try:
            _, rng = range_header.split("=", 1)
            start_s, end_s = rng.split("-", 1)
            if start_s:
                start = int(start_s)
            if end_s:
                end = int(end_s)
        except ValueError:
            raise HTTPException(416, "invalid range header")
        if start > end or start >= file_size:
            raise HTTPException(416, "invalid range",
                                 headers={"Content-Range": f"bytes */{file_size}"})
        end = min(end, file_size - 1)
        status_code = 206

    length = end - start + 1

    def iterfile():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(RANGE_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(length)}
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    return StreamingResponse(iterfile(), status_code=status_code,
                              media_type=media_type, headers=headers)


# every politician's own YouTube audio dir + the one shared debate-video dir.
# Filenames are globally unique (YouTube IDs; "<date>-<slug>.mp4" for debates),
# so a plain filename lookup across all of them is unambiguous.
MEDIA_DIRS = [CONFIGS[s]["_paths"]["youtube"] for s in SERVED_SLUGS]
if DG_DIR:
    MEDIA_DIRS.append(DG_DIR)


@app.get("/media/{filename}")
def media(filename: str, request: Request):
    # prevent path traversal
    safe = os.path.basename(filename)
    if safe != filename:
        raise HTTPException(404, "not found")
    for directory in MEDIA_DIRS:
        path = directory / safe
        if path.is_file():
            if safe.endswith(".opus"):
                mt = "audio/ogg"
            elif safe.endswith(".mp4"):
                mt = "video/mp4"
            else:
                mt = "audio/mp4"
            return _ranged_response(path, mt, request)
    raise HTTPException(404, "not found")


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
