"""Rank every Tweede Kamer speaker by how much they spoke, per calendar year.

Reads the already-downloaded vlos verslag XML (SHARED_DIR/tk/*.xml, the whole
2013+ corpus -- tk_sync.py syncs it unfiltered) and, reusing tk_parse.py's
parsing helpers, tallies spoken words + speech turns + distinct debate days
per (achternaam, voornaam) per year.

Output (data/):
  speaker_ranking.json  {"generated", "by_year": {year: [rows...]},
                         "overall": [rows...]}  -- rows sorted by words desc
  speaker_ranking.csv    year,achternaam,voornaam,fractie,functie,words,turns,days

This is the input to build_onboarding_queue.py. No downloads, no writes to the
shared pool -- purely a read-side analysis of what is already on disk.

Usage: python3 rank_speakers.py [--limit N-xml-for-a-quick-test]
"""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pipeline_config import SHARED_DIR
from tk_parse import NS, tag, SPEECH_TAGS, block_alineas, vergadering_meta, best_per_vergadering

OUT_DIR = Path(__file__).parent / "data"

# vlos chair/clerk labels -- not politicians to onboard
CHAIR = {"voorzitter", "de voorzitter", "griffier", "de griffier"}


def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def key_of(achter: str, voor: str) -> tuple[str, str]:
    return achter.casefold().strip(), voor.casefold().strip()


def speaker_fields(block) -> tuple[str, str, str, str]:
    """(voornaam, achternaam, fractie, functie) for a woordvoerder/interrumpant."""
    spr = block.find(f"{NS}spreker")
    if spr is None:
        return "", "", "", ""
    voor = norm(spr.findtext(f"{NS}voornaam"))
    achter = norm(spr.findtext(f"{NS}achternaam")) or norm(spr.findtext(f"{NS}verslagnaam"))
    fractie = norm(spr.findtext(f"{NS}fractie"))
    functie = norm(spr.findtext(f"{NS}functie"))
    return voor, achter, fractie, functie


def walk_speeches(root):
    """Yield (voornaam, achternaam, fractie, functie, n_words) per speech turn."""
    def walk(el):
        for child in el:
            t = tag(child)
            if t in SPEECH_TAGS:
                voor, achter, fractie, functie = speaker_fields(child)
                words = sum(len(x.split()) for x in block_alineas(child))
                if achter and words:
                    yield voor, achter, fractie, functie, words
                yield from walk(child)  # nested interrupties
            else:
                yield from walk(child)

    yield from walk(root)


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    tk_dir = SHARED_DIR / "tk"
    state_path = SHARED_DIR / "tk" / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    chosen = best_per_vergadering(state)  # {vergadering: best verslag id}
    xml_ids = sorted(set(chosen.values())) if chosen else sorted(p.stem for p in tk_dir.glob("*.xml"))
    if limit:
        xml_ids = xml_ids[:limit]
    print(f"scanning {len(xml_ids)} verslagen from {tk_dir}", flush=True)

    # (achterKey, voorKey) -> {"display": (voor, achter), "fracties": Counter,
    #   "functies": Counter, "years": {year: {"words", "turns", "days": set}}}
    agg: dict = defaultdict(lambda: {
        "display": ("", ""), "fracties": Counter(), "functies": Counter(),
        "years": defaultdict(lambda: {"words": 0, "turns": 0, "days": set()}),
    })

    done = 0
    for xid in xml_ids:
        path = tk_dir / f"{xid}.xml"
        if not path.exists():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        datum = (vergadering_meta(root).get("datum") or "")[:10]
        year = datum[:4]
        if not year.isdigit():
            continue
        for voor, achter, fractie, functie, words in walk_speeches(root):
            if achter.casefold() in CHAIR or f"{voor} {achter}".casefold().strip() in CHAIR:
                continue
            e = agg[key_of(achter, voor)]
            e["display"] = (voor, achter)
            if fractie:
                e["fracties"][fractie] += 1
            if functie:
                e["functies"][functie] += 1
            y = e["years"][year]
            y["words"] += words
            y["turns"] += 1
            y["days"].add(datum)
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(xml_ids)}", flush=True)

    # ---- flatten
    by_year: dict[str, list] = defaultdict(list)
    overall: list = []
    for (_, _), e in agg.items():
        voor, achter = e["display"]
        fractie = e["fracties"].most_common(1)[0][0] if e["fracties"] else ""
        functie = e["functies"].most_common(1)[0][0] if e["functies"] else ""
        tot_words = tot_turns = 0
        all_days: set = set()
        for year, y in e["years"].items():
            row = {
                "year": int(year), "achternaam": achter, "voornaam": voor,
                "fractie": fractie, "functie": functie,
                "words": y["words"], "turns": y["turns"], "days": len(y["days"]),
            }
            by_year[year].append(row)
            tot_words += y["words"]
            tot_turns += y["turns"]
            all_days |= y["days"]
        overall.append({
            "achternaam": achter, "voornaam": voor, "fractie": fractie, "functie": functie,
            "words": tot_words, "turns": tot_turns, "days": len(all_days),
            "years_active": sorted(int(y) for y in e["years"]),
        })

    for year in by_year:
        by_year[year].sort(key=lambda r: r["words"], reverse=True)
    overall.sort(key=lambda r: r["words"], reverse=True)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "speaker_ranking.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verslagen_scanned": done,
        "by_year": {y: by_year[y] for y in sorted(by_year, reverse=True)},
        "overall": overall,
    }, ensure_ascii=False, indent=1))

    with (OUT_DIR / "speaker_ranking.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["year", "achternaam", "voornaam", "fractie", "functie", "words", "turns", "days"])
        for year in sorted(by_year, reverse=True):
            for r in by_year[year]:
                w.writerow([year, r["achternaam"], r["voornaam"], r["fractie"],
                            r["functie"], r["words"], r["turns"], r["days"]])

    print(f"\n{len(overall)} distinct speakers -> {OUT_DIR}/speaker_ranking.{{json,csv}}")
    print("\ntop 15 overall (2013+ combined):")
    for r in overall[:15]:
        print(f"  {r['words']:>10,}w  {r['voornaam']} {r['achternaam']} "
              f"({r['fractie'] or r['functie'] or '?'})  {r['years_active'][0]}-{r['years_active'][-1]}")


if __name__ == "__main__":
    main()
