"""Load a per-person pipeline config (config/<slug>.json).

Raw sources that are objectively shared across every tracked politician --
Kamerverslagen/Handelingen XML, the parsed multi-speaker transcripts derived
from them (tk_parse.py/ob_parse.py already keep every speaker, not just the
person whose config triggered the parse), and Debat Direct video -- live
under a single SHARED_DIR, downloaded/parsed once and reused by anyone whose
config points at the same pool. A debate belongs to everyone who spoke in it;
duplicating it per person would waste disk and download time for no reason.

Person-specific material -- their own YouTube channel(s) and its ASR
transcripts, and their own search index -- stays under that person's
data_dir, since it has no other tracked person's content mixed into it and
no reliable cross-person attribution (see multi_project_playbook memory).
"""
import json
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "config"
SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/data/SHARED"))


def load_config(slug: str | None = None) -> dict:
    slug = slug or os.environ.get("PERSON", "wilders")
    cfg = json.loads((CONFIG_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    # DATA_DIR overrides the config file's own (production) data_dir --
    # lets a fresh checkout point at a local, writable directory (see
    # quickstart.sh) without editing the tracked config file.
    data_dir = Path(os.environ.get("DATA_DIR", cfg["data_dir"]))
    cfg["_paths"] = {
        "data": data_dir,
        "shared": SHARED_DIR,
        # shared raw sources + their derived multi-speaker transcripts
        "tk_xml": SHARED_DIR / "tk",
        "tk_state": SHARED_DIR / "tk" / "state.json",
        "ob_xml": SHARED_DIR / "ob",
        "ob_state": SHARED_DIR / "ob" / "state.json",
        "debatgemist": SHARED_DIR / "debatgemist",
        "shared_transcripts": SHARED_DIR / "transcripts",
        # person-specific
        "youtube": data_dir / "youtube",
        "transcripts": data_dir / "transcripts",
        "index": data_dir / "index",
    }
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("tk_xml", "ob_xml", "debatgemist", "shared_transcripts",
                "youtube", "transcripts", "index"):
        cfg["_paths"][key].mkdir(parents=True, exist_ok=True)
