"""Sync historical Tweede Kamer Handelingen (pre-OData era, ~1995 - 2013-06)
from officielebekendmakingen.nl via the KOOP SRU API.

Finds all TK Handelingen documents in the configured date window whose full
text mentions the person, and downloads the XML manifestation to
<data>/ob/xml/<identifier>.xml with per-document metadata in
<data>/ob/state.json. ob_parse.py keeps only documents where the person
actually speaks. The OData pipeline (tk_sync/tk_parse) covers 2013-06-25+.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline_config import load_config

PAGE = 100
DOWNLOAD_DELAY = 0.5  # repository.overheid.nl answers 429 on unthrottled bulk fetches
NS = {
    "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
    "gzd": "http://standaarden.overheid.nl/sru",
    "dcterms": "http://purl.org/dc/terms/",
}


def fetch(url: str, tries: int = 5) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def sru_page(base: str, query: str, start: int) -> ET.Element:
    q = urllib.parse.urlencode({
        "operation": "searchRetrieve", "version": "2.0",
        "maximumRecords": str(PAGE), "startRecord": str(start),
        "query": query,
    })
    return ET.fromstring(fetch(f"{base}?{q}"))


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ob = cfg.get("ob")
    if not ob:
        print(f"no ob (Handelingen 1995-2013) config for {cfg['slug']}, skipping")
        return
    # shared pool: an SRU search is per-person (filtered by name in the
    # query), but the downloaded XML itself is shared across everyone --
    # dest.exists() below means a document another person's run already
    # fetched is never re-downloaded, just re-listed.
    xml_dir = cfg["_paths"]["ob_xml"]
    xml_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg["_paths"]["ob_state"]
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    query = (
        f'w.publicatienaam==Handelingen AND dt.creator=="{ob["creator"]}" '
        f'AND cql.textAndIndexes="{ob["match_naam"]}" '
        f'AND dt.date>="{ob["since"]}" AND dt.date<="{ob["until"]}"'
    )
    start, total, downloaded = 1, None, 0
    while True:
        root = sru_page(ob["sru_base"], query, start)
        if total is None:
            total = int(root.findtext("sru:numberOfRecords", "0", NS))
            print(f"{total} documents match", flush=True)
        records = root.findall(".//sru:record", NS)
        if not records:
            break
        for rec in records:
            ident = rec.findtext(".//dcterms:identifier", "", NS)
            if not ident:
                continue
            xml_url = ""
            for item in rec.findall(".//gzd:itemUrl", NS):
                if item.get("manifestation") == "xml":
                    xml_url = item.text or ""
            state[ident] = {
                "date": rec.findtext(".//dcterms:date", "", NS),
                "title": rec.findtext(".//dcterms:title", "", NS),
                "url": rec.findtext(".//gzd:preferredUrl", "", NS),
                "xml_url": xml_url,
            }
            dest = xml_dir / f"{ident}.xml"
            if xml_url and not dest.exists():
                try:
                    dest.write_bytes(fetch(xml_url))
                    downloaded += 1
                    time.sleep(DOWNLOAD_DELAY)
                except Exception as e:
                    print(f"  skip {ident}: {e}", file=sys.stderr)
        start += len(records)
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"  ... {start-1}/{total} listed, {downloaded} downloaded", flush=True)
        if start > total:
            break
    print(f"done: {len(state)} documents in state, {downloaded} new XMLs -> {xml_dir}")


if __name__ == "__main__":
    main()
