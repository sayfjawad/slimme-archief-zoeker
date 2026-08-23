"""Parse historical Handelingen XML (officielebekendmakingen.nl) into
abo-ali-compatible transcripts, for documents where the person speaks.

Handles both document formats:
- old (~1995-2011): root <handeling>, blocks <spreker><wie><naam>/<partij>
  followed by <al> paragraphs; <voorz> = chairman blocks
- new (~2011-2013): root <officiele-publicatie>, <spreekbeurt><spreker> with
  <achternaam>/<politiek>, text in <tekst><al>

These records have no timestamps; segment start/end are ordinal numbers that
only preserve the order of speech within the document. metadata.url points to
the official page on zoek.officielebekendmakingen.nl for verification.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline_config import load_config, ensure_dirs


def strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


def alineas(el, skip_tags=()):
    """All <al> texts under el, in document order, skipping nested skip_tags."""
    out = []

    def walk(e):
        for ch in e:
            t = strip_ns(ch.tag)
            if t in skip_tags:
                continue
            if t == "al":
                text = " ".join("".join(ch.itertext()).split())
                if text:
                    out.append(text)
            else:
                walk(ch)

    walk(el)
    return out


def parse_old(root):
    """<handeling>: <spreker>/<voorz> blocks with <wie> and <al> children."""
    segments = []

    def walk(el):
        for ch in el:
            t = strip_ns(ch.tag)
            if t == "spreker":
                naam = ch.findtext(".//naam") or ""
                partij = ch.findtext(".//partij") or ""
                speaker = f"{naam} ({partij})" if partij else naam
                for text in alineas(ch, skip_tags=("spreker", "voorz", "wie")):
                    segments.append({"speaker": speaker, "text": text})
                walk(ch)
            elif t == "voorz":
                for text in alineas(ch, skip_tags=("spreker", "voorz")):
                    segments.append({"speaker": "De voorzitter", "text": text})
                walk(ch)
            else:
                walk(ch)

    walk(root)
    return segments


def parse_new(root):
    """<officiele-publicatie>: <spreekbeurt><spreker> + <tekst><al>."""
    segments = []
    def first(el, tag):
        for ch in el.iter():
            if strip_ns(ch.tag) == tag:
                return " ".join("".join(ch.itertext()).split())
        return ""

    for sb in root.iter():
        if strip_ns(sb.tag) != "spreekbeurt":
            continue
        naam = f"{first(sb, 'voorvoegsels')} {first(sb, 'achternaam')}".strip()
        partij = first(sb, "politiek")
        speaker = f"{naam} ({partij})" if naam and partij else naam
        tekst = next((c for c in sb.iter() if strip_ns(c.tag) == "tekst"), None)
        if tekst is None:
            continue
        for text in alineas(tekst, skip_tags=("spreekbeurt", "noot")):
            segments.append({"speaker": speaker, "text": text})
    return segments


def parse_document(xml_path: Path, match_naam: str):
    root = ET.parse(xml_path).getroot()
    if strip_ns(root.tag) == "handeling":
        raw = parse_old(root)
    else:
        raw = parse_new(root)
    if not raw:
        return None
    lower = match_naam.lower()
    if not any(lower in s["speaker"].lower() for s in raw):
        return None
    segments = [
        {
            "speaker_id": s["speaker"],
            "speaker": s["speaker"],
            "start": float(i),
            "end": float(i + 1),
            "text": s["text"],
        }
        for i, s in enumerate(raw)
    ]
    return segments


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ensure_dirs(cfg)
    paths = cfg["_paths"]
    match_naam = cfg["ob"]["match_naam"]
    xml_dir = paths["ob_xml"]
    state_path = paths["ob_state"]
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    # Snapshot both directories once: ob/ and the shared transcript pool live
    # on NFS, and thousands of tracked documents means a per-item exists()
    # check would mean thousands of round-trips every run.
    existing_xml = {p.stem for p in xml_dir.glob("*.xml")}
    already_parsed = {p.name[: -len(".metadata.json")] for p in paths["shared_transcripts"].glob("ob_*.metadata.json")}

    # Output goes to the SHARED transcript pool: parse_document() keeps every
    # speaker in the document, not just this config's person, so a document
    # already parsed (by this or any other tracked person's run) is reused
    # as-is -- same Handeling, same content, no need to redo the work.
    written = skipped = reused = 0
    for ident, info in sorted(state.items()):
        if ident not in existing_xml:
            continue
        date = (info.get("date") or "").replace("-", "")
        base = f"ob_{date or 'nodate'}_{ident}"
        if base in already_parsed:
            reused += 1
            continue
        xml_path = xml_dir / f"{ident}.xml"
        try:
            segments = parse_document(xml_path, match_naam)
        except ET.ParseError as e:
            print(f"  parse error {ident}: {e}", file=sys.stderr)
            continue
        if not segments:
            skipped += 1
            continue
        title = info.get("title") or ident
        transcript = {"title": title, "duration_seconds": 0, "segments": segments}
        metadata = {
            "id": ident,
            "title": title,
            "url": info.get("url", ""),
            "upload_date": date,
            "duration_seconds": 0,
            "source": "ob_handelingen",
            "transcript_source": "official",
            "speakers": sorted({s["speaker"] for s in segments if s.get("speaker")}),
        }
        (paths["shared_transcripts"] / f"{base}.json").write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
        )
        (paths["shared_transcripts"] / f"{base}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written += 1
        if written % 200 == 0:
            print(f"  ... {written} transcripts", flush=True)
    print(f"done: {written} new, {reused} already in shared pool, {skipped} without {match_naam} speaking")


if __name__ == "__main__":
    main()
