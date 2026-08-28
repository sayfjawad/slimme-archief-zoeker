"""Turn data/speaker_ranking.json into an ordered onboarding queue.

For each year 2024->2013 (descending) take the top-N speakers by spoken-word
volume, then dedupe into one list where the FIRST (most recent) appearance
wins -- so the queue is "prominent most recently" first. Drops chair/clerk
pseudo-speakers and anyone already in config/*.json.

Output: data/onboarding_queue.csv
  rank,slug,achternaam,voornaam,fractie,functie,first_year,total_words,total_days,years_active

Review this file and reorder/trim before running onboard_batch.sh.

Usage: python3 build_onboarding_queue.py [--per-year N]   (default N=20)
"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

DATA = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "config"

# characters unicodedata's NFKD doesn't decompose to ASCII
TR = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
                    "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "ð": "d"})


def slugify(achternaam: str) -> str:
    s = achternaam.translate(TR)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "onbekend"


def existing_matches() -> list[tuple[str, str]]:
    out = []
    for p in CONFIG_DIR.glob("*.json"):
        m = (json.loads(p.read_text()).get("tk") or {}).get("match") or {}
        if m.get("achternaam"):
            out.append((m["achternaam"].casefold(), (m.get("voornaam") or "").casefold()))
    return out


def main():
    per_year = 20
    if "--per-year" in sys.argv:
        per_year = int(sys.argv[sys.argv.index("--per-year") + 1])

    ranking = json.loads((DATA / "speaker_ranking.json").read_text())
    by_year = ranking["by_year"]
    overall = {(r["achternaam"].casefold(), r["voornaam"].casefold()): r for r in ranking["overall"]}
    already = existing_matches()

    def is_existing(achter: str, voor: str) -> bool:
        a, v = achter.casefold(), voor.casefold()
        return any(ea in a and (not ev or ev in v) for ea, ev in already)

    seen: set[tuple[str, str]] = set()
    queue: list[dict] = []
    for year in sorted(by_year, key=int, reverse=True):
        for r in by_year[year][:per_year]:
            k = (r["achternaam"].casefold(), r["voornaam"].casefold())
            if k in seen or is_existing(r["achternaam"], r["voornaam"]):
                continue
            seen.add(k)
            o = overall.get(k, {})
            ya = o.get("years_active") or [int(year)]
            queue.append({
                "slug": slugify(r["achternaam"]),
                "achternaam": r["achternaam"], "voornaam": r["voornaam"],
                "fractie": r["fractie"], "functie": r["functie"],
                "first_year": int(year),
                "total_words": o.get("words", r["words"]),
                "total_days": o.get("days", r["days"]),
                "years_active": f"{ya[0]}-{ya[-1]}" if len(ya) > 1 else str(ya[0]),
            })

    # de-collide slugs
    counts: dict[str, int] = {}
    for row in queue:
        base = row["slug"]
        counts[base] = counts.get(base, 0) + 1
        if counts[base] > 1:
            row["slug"] = f"{base}-{row['voornaam'].split()[0].lower()}" if row["voornaam"] else f"{base}{counts[base]}"

    out = DATA / "onboarding_queue.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "slug", "achternaam", "voornaam", "fractie", "functie",
                    "first_year", "total_words", "total_days", "years_active"])
        for i, row in enumerate(queue, 1):
            w.writerow([i, row["slug"], row["achternaam"], row["voornaam"], row["fractie"],
                        row["functie"], row["first_year"], row["total_words"],
                        row["total_days"], row["years_active"]])

    print(f"{len(queue)} politicians queued -> {out}")
    print("\nfirst 25:")
    for i, row in enumerate(queue[:25], 1):
        print(f"  {i:>3}. {row['slug']:<22} {row['voornaam']} {row['achternaam']} "
              f"({row['fractie'] or row['functie'] or '?'})  first {row['first_year']}  "
              f"{row['total_words']:,}w")


if __name__ == "__main__":
    main()
