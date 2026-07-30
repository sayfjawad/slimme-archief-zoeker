"""Download Debat Direct (voorheen Debat Gemist) video for every plenary
debate session in which the person speaks, in the lowest available quality.

For each TK transcript (tk_*.json, produced by tk_parse.py) the script:
1. collects the wallclock timestamps of the person's segments,
2. asks the Debat Direct agenda API which plenaire-zaal debates ran that day,
3. selects the debates whose [startedAt, endedAt] window contains at least
   one of those timestamps,
4. fetches the debate's vodUrl and appends ?start=&end= (the windowing the
   web player uses) to get an HLS master for exactly that session,
5. downloads lowest-bandwidth video + audio with ffmpeg (stream copy) to
   <shared>/debatgemist/<date>_<slug>.mp4 -- a debate involves whoever spoke
   in it, so the video pool (like the parsed transcripts) is shared across
   every tracked person's config, keyed by date+slug; a session another
   person's run already downloaded is simply reused.

Video position 0 corresponds to the debate's startedAt wallclock, so
playback offset for a transcript segment = wallclock - startedAt.
Mapping is stored in <shared>/debatgemist/state.json.

The archive reaches back to ~2010; days before that simply yield no video.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# pipeline_config is imported lazily in main(): remote shard workers get only
# this file + a dates JSON, without the rest of the repo.

API = "https://api.debatdirect.tweedekamer.nl/api"
PLENAIR_LOCATION = "plenaire-zaal"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def parse_dd_time(s: str) -> datetime | None:
    """'2026-04-22T10:01:13+0200' -> naive local datetime."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
    except ValueError:
        return None


def person_wallclocks(transcript: dict, achternaam: str) -> list[datetime]:
    out = []
    for seg in transcript.get("segments", []):
        if achternaam.lower() in (seg.get("speaker") or "").lower() and seg.get("wallclock"):
            try:
                out.append(datetime.fromisoformat(seg["wallclock"]))
            except ValueError:
                pass
    return out


def windowed_master(vod_url: str, starts_at: str, ends_at: str) -> str:
    sep = "&" if "?" in vod_url else "?"
    return f"{vod_url}{sep}start={urllib.parse.quote(starts_at, safe='')}&end={urllib.parse.quote(ends_at, safe='')}"


def pick_streams(master_url: str) -> tuple[str, str] | None:
    """Return (lowest-bandwidth video variant URL, audio rendition URL)."""
    with urllib.request.urlopen(master_url, timeout=60) as resp:
        lines = resp.read().decode().splitlines()
    audio = video = None
    best_bw = None
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-MEDIA") and "TYPE=AUDIO" in line and audio is None:
            m = re.search(r'URI="([^"]+)"', line)
            audio = m.group(1) if m else None
        elif line.startswith("#EXT-X-STREAM-INF"):
            m = re.search(r"BANDWIDTH=(\d+)", line)
            bw = int(m.group(1)) if m else 0
            if (best_bw is None or bw < best_bw) and i + 1 < len(lines):
                best_bw, video = bw, lines[i + 1]
    if not video:
        return None
    return (urllib.parse.urljoin(master_url, video),
            urllib.parse.urljoin(master_url, audio) if audio else "")


def variant_duration(url: str) -> float:
    with urllib.request.urlopen(url, timeout=60) as resp:
        pl = resp.read().decode()
    return sum(float(x) for x in re.findall(r"#EXTINF:([\d.]+)", pl))


def hls_input_opts() -> list[str]:
    """ffmpeg >= 7/8 refuses the Kamer-CDN's .m4v HLS segments unless the
    extension allowlist is relaxed; older versions lack (some of) the options,
    so detect what this ffmpeg supports."""
    try:
        h = subprocess.run(["ffmpeg", "-hide_banner", "-h", "demuxer=hls"],
                           capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    opts = []
    if "allowed_extensions" in h:
        opts += ["-allowed_extensions", "ALL"]
    if "extension_picky" in h:
        opts += ["-extension_picky", "0"]
    return opts


HLS_OPTS = hls_input_opts()


# the Kamer CDN occasionally half-closes a segment connection mid-stream
# without ffmpeg noticing (observed: two multi-hour "marathon" debates, APB
# and a corona debate, hung for 24+ h with sockets stuck in CLOSE-WAIT and
# blocked every later date behind them on that shard). -rw_timeout makes
# ffmpeg itself give up on a stalled read; the subprocess timeout is a
# backstop for ffmpeg builds that don't honor it.
STALL_TIMEOUT_US = 20_000_000   # 20s, per read
PROC_TIMEOUT_S = 4 * 3600       # generous cap for the longest marathon debates


def download_debate(master_url: str, dest: Path) -> bool:
    streams = pick_streams(master_url)
    if not streams:
        return False
    video_url, audio_url = streams
    # a stub playlist ("nomeeting") means no footage for this window
    if variant_duration(video_url) < 60:
        return False
    tmp = dest.with_suffix(".part.mp4")
    rw = ["-rw_timeout", str(STALL_TIMEOUT_US)]
    cmd = ["ffmpeg", "-v", "error", "-y", *HLS_OPTS, *rw, "-i", video_url]
    maps = ["-map", "0:v"]
    if audio_url:
        cmd += [*HLS_OPTS, *rw, "-i", audio_url]
        maps += ["-map", "1:a"]
    # +faststart: moov atom up front so playback/seeking starts without
    # fetching the whole (often 1-2GB) file first — see fix_faststart.py.
    cmd += maps + ["-c", "copy", "-movflags", "+faststart", str(tmp)]
    try:
        rc = subprocess.run(cmd, timeout=PROC_TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        rc = 1
    if rc != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    return True


def build_dates(paths, achternaam) -> dict[str, list[datetime]]:
    """Collect person wallclocks per date from the local TK transcripts."""
    dates: dict[str, list[datetime]] = {}
    for meta_path in sorted(paths["shared_transcripts"].glob("tk_*.metadata.json")):
        base = meta_path.name[: -len(".metadata.json")]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        d = meta.get("upload_date", "")
        if len(d) != 8:
            continue
        date_iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        transcript = json.loads((paths["shared_transcripts"] / f"{base}.json").read_text(encoding="utf-8"))
        wcs = person_wallclocks(transcript, achternaam)
        if wcs:
            dates.setdefault(date_iso, []).extend(wcs)
    return dates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug", nargs="?", default=None, help="config slug (default wilders)")
    ap.add_argument("--shard", metavar="I/N", default=None,
                    help="only process dates whose sorted index modulo N equals I")
    ap.add_argument("--dates-json", metavar="FILE", default=None,
                    help="read {date: [wallclock,..]} from FILE instead of local "
                         "transcripts+config (for remote shard workers)")
    ap.add_argument("--export-dates", metavar="FILE", default=None,
                    help="write the dates JSON for remote workers and exit")
    ap.add_argument("--dest", metavar="DIR", default=None,
                    help="download directory (default <data>/debatgemist)")
    ap.add_argument("--have", metavar="FILE", default=None,
                    help="file with mp4 basenames that exist elsewhere; skip those")
    args = ap.parse_args()

    if args.dates_json:
        raw = json.loads(Path(args.dates_json).read_text(encoding="utf-8"))
        dates = {d: [datetime.fromisoformat(x) for x in wcs] for d, wcs in raw.items()}
        dg_dir = Path(args.dest) if args.dest else Path("out")
    else:
        from pipeline_config import load_config
        cfg = load_config(args.slug)
        paths = cfg["_paths"]
        dates = build_dates(paths, cfg["tk"]["match"]["achternaam"])
        dg_dir = Path(args.dest) if args.dest else paths["debatgemist"]
    dg_dir.mkdir(parents=True, exist_ok=True)

    if args.export_dates:
        Path(args.export_dates).write_text(json.dumps(
            {d: [wc.isoformat() for wc in wcs] for d, wcs in sorted(dates.items())},
            ensure_ascii=False), encoding="utf-8")
        print(f"{len(dates)} dates exported to {args.export_dates}")
        return

    shard_i = shard_n = None
    if args.shard:
        shard_i, shard_n = map(int, args.shard.split("/"))
    # sharded runs keep their own state file so parallel workers never write
    # the same file; dg_pull.sh merges them into state.json for the app
    state_path = dg_dir / (f"state.{shard_i}of{shard_n}.json" if args.shard else "state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    have = set()
    if args.have:
        have = {ln.strip() for ln in Path(args.have).read_text().splitlines() if ln.strip()}

    print(f"{len(dates)} dates to check"
          + (f" (shard {args.shard})" if args.shard else ""), flush=True)
    n_new = n_skip = n_novideo = 0
    for idx, date_iso in enumerate(sorted(dates)):
        if shard_n is not None and idx % shard_n != shard_i:
            continue
        try:
            agenda = get_json(f"{API}/agenda/{date_iso}")
        except Exception as e:
            print(f"  agenda {date_iso}: {e}", file=sys.stderr)
            continue
        for deb in agenda.get("debates", []):
            if deb.get("locationId") != PLENAIR_LOCATION:
                continue
            starts, ends = deb.get("startedAt"), deb.get("endedAt")
            t0, t1 = parse_dd_time(starts), parse_dd_time(ends)
            if not t0 or not t1:
                continue
            if not any(t0 <= wc <= t1 for wc in dates[date_iso]):
                continue
            slug = deb.get("slug") or deb["id"][:8]
            dest = dg_dir / f"{date_iso}_{slug}.mp4"
            key = dest.name
            # `key in state` covers files a remote worker downloaded that the
            # puller already drained; `have` covers files downloaded before
            # this worker's shard started
            if dest.exists() or key in have or key in state:
                n_skip += 1
                continue
            try:
                detail = get_json(f"{API}/debates/{deb['id']}")
                vod = ((detail.get("video") or {}).get("vodUrl")) or ""
                if not vod:
                    n_novideo += 1
                    continue
                master = windowed_master(vod, starts, ends)
                ok = download_debate(master, dest)
            except Exception as e:
                print(f"  {key}: {e}", file=sys.stderr)
                continue
            if ok:
                state[key] = {
                    "date": date_iso,
                    "debate_id": deb["id"],
                    "name": deb.get("name", ""),
                    "video_start": starts,
                    "video_end": ends,
                }
                state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
                mb = dest.stat().st_size / 1e6
                n_new += 1
                print(f"  + {key} ({mb:.0f} MB) {deb.get('name','')[:50]}", flush=True)
            else:
                n_novideo += 1
                print(f"  no footage: {date_iso} {deb.get('name','')[:50]}", flush=True)
            time.sleep(1)
    print(f"done: {n_new} downloaded, {n_skip} already present, {n_novideo} without footage")


if __name__ == "__main__":
    main()
