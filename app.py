"""Wilders-archief smart-search web application.

- POST /api/search  semantic search over all transcripts (date/speaker-filterable)
- POST /api/ask     RAG: retrieve relevant fragments + LLM answer with citations
- GET  /media/{f}   serve local audio (opus) with seekable playback
- /                 single-page frontend (static/index.html)

Same architecture as abo-ali-search; sources here are official Tweede Kamer
verslagen (transcript_source=official) and YouTube ASR transcripts.
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from embedder import Embedder
from pipeline_config import load_config

BASE = Path(__file__).parent
CFG = load_config()
INDEX_DIR = CFG["_paths"]["index"]
MEDIA_DIR = CFG["_paths"]["youtube"]
DG_DIR = CFG["_paths"]["debatgemist"]
PERSON = CFG["person"]
PERSON_SLUG = CFG["slug"]
PERSON_MATCH = CFG["tk"]["match"]["achternaam"]
STATS_DB = BASE / "stats.sqlite"  # outside index/ so re-indexing keeps history

app = FastAPI(title=f"{PERSON} Archief")

# ---------------------------------------------------------------- index state
_state: dict = {}


@app.on_event("startup")
def load_index():
    db = sqlite3.connect(INDEX_DIR / "index.sqlite", check_same_thread=False)
    db.row_factory = sqlite3.Row
    emb = np.load(INDEX_DIR / "embeddings.npy")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    matrix = torch.from_numpy(emb).to(device)  # (n, 1024) fp16
    # per-chunk upload_date + person mask for fast filtering on GPU
    rows = db.execute(
        "SELECT c.id, c.speaker, v.upload_date FROM chunks c JOIN videos v ON v.video = c.video ORDER BY c.id"
    ).fetchall()
    dates = np.array([int(r["upload_date"] or 0) for r in rows], dtype=np.int64)
    person = np.array([PERSON_MATCH.lower() in (r["speaker"] or "").lower() for r in rows])
    # Debat Direct video mapping: wallclock -> (file, video_start)
    dg_windows = []
    dg_state_path = DG_DIR / "state.json"
    if dg_state_path.exists():
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
        db=db,
        matrix=matrix,
        dates=torch.from_numpy(dates).to(device),
        person_mask=torch.from_numpy(person).to(device),
        embedder=Embedder(device=device, max_length=512),
        device=device,
        dg_windows=dg_windows,
    )
    print(f"index loaded: {matrix.shape[0]} chunks on {device} ({int(person.sum())} by {PERSON}), "
          f"{len(dg_windows)} debate videos")


# ------------------------------------------------------------- usage tracking
# NOTE: wilders-search and yesilgoz-search (and any future person) run from
# this SAME checkout with different PERSON env vars, so STATS_DB is one
# shared file on disk. Every row is tagged with `person` (the config slug)
# so each instance's /api/statistics only ever aggregates its own traffic.
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


def log_query(request: Request, mode: str, query: str):
    try:
        with _stats_conn() as conn:
            conn.execute(
                "INSERT INTO queries (mode, ip, query, person) VALUES (?,?,?,?)",
                (mode, client_ip(request), query[:500], PERSON_SLUG),
            )
    except Exception as e:  # stats must never break search
        print(f"stats logging failed: {e}")


@app.get("/api/statistics")
def api_statistics():
    with _stats_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM queries WHERE person=?", (PERSON_SLUG,)).fetchone()[0]
        by_mode = dict(conn.execute(
            "SELECT mode, COUNT(*) FROM queries WHERE person=? GROUP BY mode", (PERSON_SLUG,)).fetchall())
        unique_ips = conn.execute(
            "SELECT COUNT(DISTINCT ip) FROM queries WHERE person=?", (PERSON_SLUG,)).fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ip) FROM queries WHERE person=? AND date(ts) = date('now')",
            (PERSON_SLUG,)).fetchone()
        per_day = conn.execute("""
            SELECT date(ts) d, COUNT(*), COUNT(DISTINCT ip)
            FROM queries WHERE person=? AND ts >= datetime('now','-30 days')
            GROUP BY d ORDER BY d""", (PERSON_SLUG,)).fetchall()
        first = conn.execute(
            "SELECT MIN(date(ts)) FROM queries WHERE person=?", (PERSON_SLUG,)).fetchone()[0]
    return {
        "person": PERSON,
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


def retrieve(query: str, top_k: int, date_from: str | None, date_to: str | None,
             only_person: bool = False):
    st = _state
    q = st["embedder"].encode([query]).to(st["device"])  # (1, 1024)
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
    top_k: int = 20
    date_from: str | None = None  # YYYYMMDD
    date_to: str | None = None
    only_person: bool = False


@app.post("/api/search")
def api_search(req: SearchReq, request: Request):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    log_query(request, "search", req.query)
    return {"results": retrieve(req.query, min(req.top_k, 100), req.date_from, req.date_to, req.only_person)}


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


ANSWER_SYSTEM = f"""Je bent een onderzoeksassistent voor een doorzoekbaar archief van alles wat
{PERSON} in het openbaar heeft gezegd: officiële Tweede Kamer-verslagen en
transcripties van openbare video's. Je krijgt genummerde fragmenten uit dit
archief (ASR-fragmenten kunnen transcriptiefouten bevatten — ga daar slim mee om).
Beantwoord de vraag van de gebruiker uitsluitend op basis van de fragmenten:
- Zeg wat er gezegd is en wanneer (datum en tijdstip binnen het debat/de video).
- Zet na elke bewering het bronnummer tussen blokhaken, bijv. [3].
- Fragmenten van andere sprekers zijn context; schrijf niets aan {PERSON} toe
  dat een andere spreker zei.
- Staat het antwoord niet in de fragmenten, zeg dat dan expliciet.
- Antwoord in helder, beknopt Nederlands. Blijf feitelijk en neutraal."""


class AskReq(BaseModel):
    question: str
    top_k: int = 16
    date_from: str | None = None
    date_to: str | None = None
    only_person: bool = False


@app.post("/api/ask")
def api_ask(req: AskReq, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    log_query(request, "ask", req.question)
    sources = retrieve(req.question, min(req.top_k, 60), req.date_from, req.date_to, req.only_person)
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
                "messages": [
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user",
                     "content": f"Fragmenten:\n\n{excerpts}\n\nVraag: {req.question} /no_think"},
                ],
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        # strip <think>...</think> reasoning blocks some local models emit
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        if cited:
            for i, s in enumerate(sources):
                s["cited"] = (i + 1) in cited
    except Exception as e:
        error = str(e)
    return {"answer": answer, "error": error, "sources": sources}


@app.get("/api/stats")
def api_stats():
    db = _state["db"]
    return {
        "videos": db.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
        "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "person": PERSON,
        "person_chunks": int(_state["person_mask"].sum().item()),
        "hero_image": CFG.get("hero_image"),
    }


# ---------------------------------------------------------------------- media
@app.get("/media/{filename}")
def media(filename: str):
    # prevent path traversal
    safe = os.path.basename(filename)
    if safe != filename:
        raise HTTPException(404, "not found")
    for directory in (MEDIA_DIR, DG_DIR):
        path = directory / safe
        if path.is_file():
            if safe.endswith(".opus"):
                mt = "audio/ogg"
            elif safe.endswith(".mp4"):
                mt = "video/mp4"
            else:
                mt = "audio/mp4"
            return FileResponse(path, media_type=mt)
    raise HTTPException(404, "not found")


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
