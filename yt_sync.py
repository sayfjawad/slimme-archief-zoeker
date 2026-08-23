"""Download audio (opus) + info-json for every video on the configured
YouTube channels, using yt-dlp's --download-archive for incremental sync.

Files land in <data>/youtube/ as <upload_date>_<yt_id>.opus + .info.json.
Playback later links straight to youtube.com with &t=<seconds>, so no video
is stored. transcribe_batch.py picks up the audio from here.
"""
import subprocess
import sys

from pipeline_config import load_config, ensure_dirs


def sync_channel(channel: dict, out_dir, archive) -> int:
    url = channel["url"].rstrip("/") + "/videos"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-update",
        "--ignore-errors",
        "-t", "sleep",  # delay between requests; without it YouTube rate-limits the session
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "opus",
        "--write-info-json",
        "--download-archive", str(archive),
        "-o", str(out_dir / "%(upload_date)s_%(id)s.%(ext)s"),
        url,
    ]
    print(f"syncing {channel['name']}: {url}")
    return subprocess.run(cmd).returncode


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ensure_dirs(cfg)
    out_dir = cfg["_paths"]["youtube"]
    archive = out_dir / "archive.txt"
    for channel in cfg["youtube"]["channels"]:
        rc = sync_channel(channel, out_dir, archive)
        if rc != 0:
            print(f"warning: yt-dlp exited {rc} for {channel['name']}", file=sys.stderr)


if __name__ == "__main__":
    main()
