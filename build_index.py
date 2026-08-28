"""Build the search index from the shared transcript pool + this person's
own transcripts.

Same design as abo-ali-search: reads every <base>.json + <base>.metadata.json,
merges consecutive same-speaker segments into ~700-char retrieval chunks,
embeds them with BGE-M3 on GPU, and writes:
  <data>/index/index.sqlite   - videos + chunks tables
  <data>/index/embeddings.npy - fp16 (n_chunks, 1024), row i == chunk id i

Extra columns vs abo-ali: videos.source ("tk_verslag" / "youtube:<channel>")
and videos.transcript_source ("official" / "asr").

Two sources, per pipeline_config.py:
  - <shared>/transcripts: tk_*/ob_* multi-speaker debate transcripts, shared
    across every tracked person -- included here only when this person's
    name appears in that transcript's `speakers` list (see tk_parse.py/
    ob_parse.py), so each person's index covers only debates they were
    actually in, even though the transcript pool itself is shared.
  - <data>/transcripts: this person's own YouTube ASR transcripts (yt_*),
    never shared with anyone else's index.
"""
import hashlib
import itertools
import json
import shutil
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from pipeline_config import load_config, ensure_dirs, SHARED_DIR

DIM = 1024  # BGE-M3 dense dim (see embedder.py); kept here so a warm-cache
# build (every chunk already embedded) never has to import torch/transformers.

MERGE_TARGET_CHARS = 700
SHRINK_GUARD = 0.9  # refuse to leave a rebuilt index with <90% of the prior video count (unless --force)
# One content-addressed BGE-M3 vector cache shared by every politician's
# build: TK/Handelingen debate text is byte-identical across politicians
# (same shared transcripts -> same chunks -> same vectors), so after the
# first politician populates it, every later rebuild is ~all cache hits and
# pass 2 costs seconds instead of embedding hundreds of thousands of chunks
# on CPU again. YouTube-ASR chunks are person-specific and simply never hit.
EMB_CACHE_DB = SHARED_DIR / "emb_cache.sqlite"


def iter_videos(transcripts_dir: Path):
    for meta_path in sorted(transcripts_dir.glob("*.metadata.json")):
        base = meta_path.name[: -len(".metadata.json")]
        transcript_path = transcripts_dir / f"{base}.json"
        if not transcript_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"skip {base}: {e}", file=sys.stderr)
            continue
        yield base, meta, transcript


def person_matches(speakers: list[str], cfg: dict) -> bool:
    """Same substring-match convention as tk_parse.person_speaks() / the
    match_naam check in ob_parse.parse_document(), applied to a shared
    transcript's precomputed speaker list instead of re-parsing segments."""
    tk_match = cfg.get("tk", {}).get("match", {})
    achter = (tk_match.get("achternaam") or "").lower()
    voor = (tk_match.get("voornaam") or "").lower()
    ob_naam = (cfg.get("ob", {}).get("match_naam") or "").lower()
    for name in speakers:
        n = name.lower()
        if achter and achter in n and (not voor or voor in n):
            return True
        if ob_naam and ob_naam in n:
            return True
    return False


def iter_shared_for_person(shared_dir: Path, cfg: dict):
    for base, meta, transcript in iter_videos(shared_dir):
        if person_matches(meta.get("speakers") or [], cfg):
            yield base, meta, transcript


def merge_segments(segments):
    """Greedy merge of consecutive same-speaker segments into chunks."""
    chunks = []
    cur = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if (
            cur is not None
            and seg.get("speaker_id") == cur["speaker_id"]
            and len(cur["text"]) + len(text) <= MERGE_TARGET_CHARS
        ):
            cur["text"] += " " + text
            cur["end"] = seg["end"]
        else:
            if cur:
                chunks.append(cur)
            cur = {
                "speaker_id": seg.get("speaker_id") or "",
                "speaker": seg.get("speaker") or "",
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "wallclock": seg.get("wallclock") or "",
            }
    if cur:
        chunks.append(cur)
    return chunks


BACKUP_RETENTION_DAYS = 14


def _video_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(db_path)
        try:
            return con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return None


def cache_lookup(hashes: list[bytes]) -> dict[bytes, bytes]:
    """{sha1 -> fp16 vector bytes} for the hashes already in the shared cache."""
    con = sqlite3.connect(EMB_CACHE_DB, timeout=60)
    try:
        con.execute("PRAGMA busy_timeout=60000")
        con.execute("CREATE TABLE IF NOT EXISTS cache (h BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID")
        con.commit()
        have: dict[bytes, bytes] = {}
        uniq = list(set(hashes))
        for k in range(0, len(uniq), 900):
            part = uniq[k : k + 900]
            rows = con.execute(
                f"SELECT h, v FROM cache WHERE h IN ({','.join('?' * len(part))})", part
            )
            have.update(rows)
        return have
    finally:
        con.close()


def cache_store(pairs: dict[bytes, bytes]) -> None:
    if not pairs:
        return
    con = sqlite3.connect(EMB_CACHE_DB, timeout=60)
    try:
        con.execute("PRAGMA busy_timeout=60000")
        con.executemany("INSERT OR IGNORE INTO cache (h, v) VALUES (?, ?)", list(pairs.items()))
        con.commit()
    finally:
        con.close()


def backup_and_prune_index(index_dir: Path) -> None:
    """Copy the current index.sqlite + embeddings.npy to
    <index_dir>/backups/<YYYY-MM-DD>/ before this run overwrites them, and
    drop backups older than BACKUP_RETENTION_DAYS. One backup per calendar
    day is enough (skip if today's already exists, e.g. a second run same
    day) -- rebuilding is destructive (db_path.unlink() below), so an
    index that took real, hard-to-redo work to build (transcription,
    diarization) must never be only one bad run away from being gone with
    nothing to fall back on. Cheap: the index is small; it's the source
    transcripts that are expensive, and this doesn't replace backing those
    up too where they aren't already durably stored elsewhere.
    """
    backups_dir = index_dir / "backups"
    today_dir = backups_dir / date.today().isoformat()
    src_db = index_dir / "index.sqlite"
    src_emb = index_dir / "embeddings.npy"
    if not today_dir.exists() and (src_db.exists() or src_emb.exists()):
        today_dir.mkdir(parents=True, exist_ok=True)
        for src in (src_db, src_emb):
            if src.exists():
                shutil.copy2(src, today_dir / src.name)
        print(f"backup: {today_dir}", flush=True)

    cutoff = date.today() - timedelta(days=BACKUP_RETENTION_DAYS)
    if backups_dir.exists():
        for d in backups_dir.iterdir():
            try:
                old = date.fromisoformat(d.name) < cutoff
            except ValueError:
                continue
            if old:
                shutil.rmtree(d, ignore_errors=True)


def find_media_file(youtube_dir: Path, base: str) -> str:
    if base.startswith("yt_"):
        for ext in (".opus", ".m4a", ".webm"):
            if (youtube_dir / f"{base[3:]}{ext}").exists():
                return f"{base[3:]}{ext}"
    return ""


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    cfg = load_config(pos[0] if pos else None)
    ensure_dirs(cfg)
    paths = cfg["_paths"]

    backup_and_prune_index(paths["index"])
    # the copy backup_and_prune_index just took (if any) is the "before" the
    # shrink guard at the end compares against
    guard_ref = paths["index"] / "backups" / date.today().isoformat() / "index.sqlite"
    videos_before = _video_count(guard_ref)

    db_path = paths["index"] / "index.sqlite"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE videos (
            video TEXT PRIMARY KEY, yt_id TEXT, title TEXT, url TEXT,
            upload_date TEXT, duration REAL, media_file TEXT,
            source TEXT, transcript_source TEXT
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, video TEXT, speaker TEXT,
            start REAL, end REAL, text TEXT, wallclock TEXT
        );
    """)

    print("pass 1: parsing transcripts...", flush=True)
    all_texts = []
    chunk_id = 0
    n_videos = 0
    for base, meta, transcript in itertools.chain(
        iter_shared_for_person(paths["shared_transcripts"], cfg),
        iter_videos(paths["transcripts"]),
    ):
        chunks = merge_segments(transcript.get("segments") or [])
        if not chunks:
            continue
        db.execute(
            "INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?)",
            (
                base,
                meta.get("id") or "",
                meta.get("title") or transcript.get("title") or base,
                meta.get("url") or "",
                meta.get("upload_date") or "",
                meta.get("duration_seconds") or transcript.get("duration_seconds") or 0,
                find_media_file(paths["youtube"], base),
                meta.get("source") or "",
                meta.get("transcript_source") or "",
            ),
        )
        for c in chunks:
            db.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                (chunk_id, base, c["speaker"], c["start"], c["end"], c["text"], c["wallclock"]),
            )
            all_texts.append(c["text"])
            chunk_id += 1
        n_videos += 1
        if n_videos % 250 == 0:
            print(f"  {n_videos} videos, {chunk_id} chunks", flush=True)
    db.commit()
    print(f"parsed {n_videos} videos -> {chunk_id} chunks", flush=True)

    t0 = time.time()
    print("pass 2: embedding (shared cache lookup first)...", flush=True)
    hashes = [hashlib.sha1(t.encode("utf-8")).digest() for t in all_texts]
    emb = np.zeros((len(all_texts), DIM), dtype=np.float16)
    have = cache_lookup(hashes)
    miss = [i for i, h in enumerate(hashes) if h not in have]
    for i, h in enumerate(hashes):
        v = have.get(h)
        if v is not None:
            emb[i] = np.frombuffer(v, dtype=np.float16)
    print(f"  cache: {len(all_texts) - len(miss)} hit, {len(miss)} to embed", flush=True)

    if miss:
        import torch
        from embedder import Embedder
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  embedding {len(miss)} chunks on {device}...", flush=True)
        embedder = Embedder(device=device)
        tok = embedder.tokenizer

        order = [miss[k] for k in np.argsort([len(all_texts[i]) for i in miss])]
        TOKEN_BUDGET = 16384
        batch_idx: list[int] = []
        batch_max_tok = 0
        done = 0

        def flush():
            nonlocal batch_idx, batch_max_tok, done
            if not batch_idx:
                return
            texts = [all_texts[i] for i in batch_idx]
            vecs = embedder.encode(texts).numpy().astype(np.float16)
            emb[batch_idx] = vecs
            done += len(batch_idx)
            if done % 5000 < len(batch_idx):
                rate = done / (time.time() - t0)
                eta = (len(miss) - done) / max(rate, 1)
                print(f"  {done}/{len(miss)}  {rate:.0f} chunks/s  eta {eta/60:.1f} min", flush=True)
            batch_idx = []
            batch_max_tok = 0

        for i in order:
            ntok = min(len(tok.encode(all_texts[i], add_special_tokens=True)), embedder.max_length)
            new_max = max(batch_max_tok, ntok)
            if batch_idx and new_max * (len(batch_idx) + 1) > TOKEN_BUDGET:
                flush()
                new_max = ntok
            batch_idx.append(int(i))
            batch_max_tok = new_max
        flush()

        cache_store({hashes[i]: emb[i].tobytes() for i in miss})

    np.save(paths["index"] / "embeddings.npy", emb)
    db.close()

    # shrink guard: a rebuild is destructive (index.sqlite was unlinked above);
    # if the fresh index covers far fewer debates than the pre-build one, that
    # is almost always a parse/source regression, not reality -- restore the
    # backup and bail rather than let it reach the app (see docs/postmortem.md
    # 5.11). daily_sync.py has its own copy of this check for the app-restart
    # step; this one covers a manual `build_index.py <slug>` run.
    videos_after = _video_count(paths["index"] / "index.sqlite")
    if (videos_before and videos_after is not None
            and videos_after < videos_before * SHRINK_GUARD and not force):
        shutil.copy2(guard_ref, paths["index"] / "index.sqlite")
        emb_ref = guard_ref.parent / "embeddings.npy"
        if emb_ref.exists():
            shutil.copy2(emb_ref, paths["index"] / "embeddings.npy")
        sys.exit(
            f"ABORT: rebuilt index has {videos_after} videos vs {videos_before} before "
            f"(>{int((1 - SHRINK_GUARD) * 100)}% drop). Restored the pre-build index "
            f"from {guard_ref.parent}. Re-run with --force if the shrink is real."
        )
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {paths['index']}", flush=True)


if __name__ == "__main__":
    main()
