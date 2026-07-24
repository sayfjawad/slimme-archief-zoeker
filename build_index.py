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
import itertools
import json
import shutil
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from embedder import Embedder, DIM
from pipeline_config import load_config, ensure_dirs

MERGE_TARGET_CHARS = 700


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
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ensure_dirs(cfg)
    paths = cfg["_paths"]

    backup_and_prune_index(paths["index"])

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

    print("pass 2: embedding on GPU...", flush=True)
    embedder = Embedder(device="cuda:0")
    tok = embedder.tokenizer

    lengths = [len(t) for t in all_texts]
    order = np.argsort(lengths)
    emb = np.zeros((len(all_texts), DIM), dtype=np.float16)

    TOKEN_BUDGET = 16384
    batch_idx: list[int] = []
    batch_max_tok = 0
    done = 0
    t0 = time.time()

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
            eta = (len(all_texts) - done) / max(rate, 1)
            print(f"  {done}/{len(all_texts)}  {rate:.0f} chunks/s  eta {eta/60:.1f} min", flush=True)
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

    np.save(paths["index"] / "embeddings.npy", emb)
    db.close()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {paths['index']}", flush=True)


if __name__ == "__main__":
    main()
