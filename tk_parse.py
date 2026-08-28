"""Parse TK vlos verslag XML into abo-ali-compatible transcripts.

Reads downloaded XMLs from <data>/tk/xml/ (see tk_sync.py), picks the best
verslag per vergadering (Gecorrigeerd > Casco > Ongecorrigeerd, then latest
GewijzigdOp), and for every vergadering where the configured person speaks
writes:
  <data>/transcripts/<base>.json           - {"title", "duration_seconds", "segments"}
  <data>/transcripts/<base>.metadata.json  - id/title/url/upload_date/source fields

Segments carry speaker_id/speaker/start/end/text like abo-ali, plus a
"wallclock" ISO timestamp (markeertijdbegin) to link into Debat Gemist.
"""
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from pipeline_config import load_config, ensure_dirs

NS = "{http://www.tweedekamer.nl/ggm/vergaderverslag/v1.0}"
STATUS_RANK = {"Gecorrigeerd": 3, "Casco": 2, "Ongecorrigeerd": 1}


def tag(el) -> str:
    return el.tag.split("}")[-1]


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def speaker_of(block) -> tuple[str, str]:
    """Return (speaker_id, display name) for a woordvoerder/interrumpant block."""
    spr = block.find(f"{NS}spreker")
    if spr is None:
        return "", ""
    voor = (spr.findtext(f"{NS}voornaam") or "").strip()
    achter = (spr.findtext(f"{NS}achternaam") or spr.findtext(f"{NS}verslagnaam") or "").strip()
    fractie = (spr.findtext(f"{NS}fractie") or "").strip()
    name = f"{voor} {achter}".strip()
    if fractie:
        name = f"{name} ({fractie})"
    return spr.get("objectid", ""), name


SPEECH_TAGS = ("woordvoerder", "interrumpant")


def block_alineas(block):
    """Yield the text of each alinea under this block's own <tekst>, skipping
    anything that belongs to a nested woordvoerder/interrumpant."""

    def walk(el):
        for child in el:
            t = tag(child)
            if t in SPEECH_TAGS:
                continue
            if t == "alinea":
                items = ["".join(it.itertext()).strip() for it in child.iter(f"{NS}alineaitem")]
                # the first alineaitem is often the vlos speaker label ("De heer X:")
                if items and items[0].endswith(":") and len(items[0]) < 80:
                    items = items[1:]
                text = " ".join(x for x in items if x)
                if text:
                    yield text
            else:
                yield from walk(child)

    for tekst in block:
        if tag(tekst) == "tekst":
            yield from walk(tekst)


def extract_segments(root):
    """Walk the document in order and emit raw segments with wallclock times."""
    segments = []

    def walk(el):
        for child in el:
            t = tag(child)
            if t in ("woordvoerder", "interrumpant"):
                spk_id, spk = speaker_of(child)
                begin = parse_time(child.findtext(f"{NS}markeertijdbegin"))
                end = parse_time(child.findtext(f"{NS}markeertijdeind"))
                for text in block_alineas(child):
                    segments.append(
                        {"speaker_id": spk_id, "speaker": spk, "begin": begin, "end": end, "text": text}
                    )
                walk(child)  # nested interrupties keep document order
            else:
                walk(child)

    walk(root)
    return segments


def person_speaks(segments, match: dict) -> bool:
    achter = match.get("achternaam", "").lower()
    voor = match.get("voornaam", "").lower()
    for s in segments:
        name = s["speaker"].lower()
        if achter and achter in name and (not voor or voor in name):
            return True
    return False


def vergadering_meta(root) -> dict:
    verg = root.find(f".//{NS}vergadering")
    if verg is None:
        return {}
    return {
        "titel": (verg.findtext(f"{NS}titel") or "").strip(),
        "datum": (verg.findtext(f"{NS}datum") or "")[:10],
        "vergadering_objectid": verg.get("objectid", ""),
        "soort": verg.get("soort", ""),
    }


def parse_verslag(xml_path: Path, verslag_id: str, match: dict | None) -> tuple[dict, dict] | None:
    """Return (transcript, metadata), or None when there are no segments or
    (when `match` is given) the person does not speak. match=None -> keep
    every vergadering regardless of who spoke (pool pre-population)."""
    root = ET.parse(xml_path).getroot()
    raw = extract_segments(root)
    if not raw or (match is not None and not person_speaks(raw, match)):
        return None

    times = [s["begin"] for s in raw if s["begin"]]
    t0 = min(times) if times else None
    duration = (max(s["end"] for s in raw if s["end"]) - t0).total_seconds() if t0 else 0

    segments = []
    for s in raw:
        start = (s["begin"] - t0).total_seconds() if (t0 and s["begin"]) else 0.0
        end = (s["end"] - t0).total_seconds() if (t0 and s["end"]) else start
        segments.append(
            {
                "speaker_id": s["speaker_id"],
                "speaker": s["speaker"],
                "start": round(start, 1),
                "end": round(end, 1),
                "text": s["text"],
                "wallclock": s["begin"].isoformat() if s["begin"] else "",
            }
        )

    meta = vergadering_meta(root)
    title = meta.get("titel") or f"Vergadering {meta.get('datum', '')}"
    transcript = {"title": title, "duration_seconds": duration, "segments": segments}
    metadata = {
        "id": verslag_id,
        "title": title,
        "url": "",
        "upload_date": meta.get("datum", "").replace("-", ""),
        "duration_seconds": duration,
        "source": "tk_verslag",
        "transcript_source": "official",
        "vergadering_id": meta.get("vergadering_objectid", ""),
        "verslag_id": verslag_id,
    }
    return transcript, metadata


def best_per_vergadering(state: dict) -> dict:
    """Pick the best verslag id per vergadering from tk_sync state."""
    best: dict[str, tuple] = {}
    for vid, info in state.get("verslagen", {}).items():
        verg = info.get("vergadering_id") or vid
        key = (STATUS_RANK.get(info.get("status", ""), 0), info.get("gewijzigd_op", ""))
        if verg not in best or key > best[verg][0]:
            best[verg] = (key, vid)
    return {verg: vid for verg, (_, vid) in best.items()}


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    parse_all = "--all" in sys.argv  # pool pre-population: keep every debate
    shard = (0, 1)  # --shard i/n : split the verslag list across machines
    if "--shard" in sys.argv:                      # (only with --all; each shard
        i, n = sys.argv[sys.argv.index("--shard") + 1].split("/")  # writes a
        shard = (int(i), int(n))                   # disjoint set of files, so
    cfg = load_config(pos[0] if pos else None)     # parallel runs are safe)
    ensure_dirs(cfg)
    paths = cfg["_paths"]
    match = None if parse_all else cfg["tk"]["match"]

    state = json.loads(paths["tk_state"].read_text()) if paths["tk_state"].exists() else {}
    chosen = best_per_vergadering(state)
    if not chosen:  # no sync state: parse every downloaded xml
        chosen = {p.stem: p.stem for p in paths["tk_xml"].glob("*.xml")}

    # Snapshot both directories once: tk/ and the shared transcript pool live
    # on NFS, and tens of thousands of tracked vergaderingen means a per-item
    # exists()/glob() would mean tens of thousands of round-trips every run.
    existing_xml = {p.stem for p in paths["tk_xml"].glob("*.xml")}
    already_parsed = {
        p.name[: -len(".metadata.json")].rsplit("_", 1)[-1]
        for p in paths["shared_transcripts"].glob("tk_*.metadata.json")
    }

    # Output goes to the SHARED transcript pool: the parser keeps every
    # speaker in the vergadering, not just this config's person, so a
    # vergadering already parsed (by this or any other tracked person's run)
    # is reused as-is rather than reparsed -- same debate, same content.
    written = skipped = reused = 0
    new_speakers: set[str] = set()
    for verg, verslag_id in sorted(chosen.items()):
        if verslag_id not in existing_xml:
            continue
        if verslag_id[:8] in already_parsed:
            reused += 1
            continue
        if shard[1] > 1 and int(verslag_id[:8], 16) % shard[1] != shard[0]:
            continue
        xml_path = paths["tk_xml"] / f"{verslag_id}.xml"
        result = parse_verslag(xml_path, verslag_id, match)
        if result is None:
            skipped += 1
            continue
        transcript, metadata = result
        base = f"tk_{metadata['upload_date'] or 'nodate'}_{verslag_id[:8]}"
        metadata["speakers"] = sorted({s["speaker"] for s in transcript["segments"] if s.get("speaker")})
        new_speakers.update(metadata["speakers"])
        (paths["shared_transcripts"] / f"{base}.json").write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
        )
        (paths["shared_transcripts"] / f"{base}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1
        if written % 200 == 0:
            print(f"  wrote {written}...", flush=True)
    who = "no segments" if parse_all else f"without {cfg['person']}"
    print(f"done: {written} new, {reused} already in shared pool, {skipped} {who}")

    if parse_all:
        # manifest of every speaker in the debates added this run -- daily_sync
        # rebuilds only the politicians who appear in it, not all ~100.
        name = "last_parse_new.json" if shard[1] == 1 else f"last_parse_new.shard{shard[0]}.json"
        (paths["shared_transcripts"].parent / name).write_text(
            json.dumps({"date": datetime.now().strftime("%Y-%m-%d"),
                        "written": written, "new_speakers": sorted(new_speakers)}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
