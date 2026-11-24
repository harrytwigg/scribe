"""YouTube videos as audio, through yt-dlp (docs/cli.md, "YouTube").

The `scribe` command uses this to turn a YouTube link into an `Episode` (what `scribe resolve` prints)
and to download the video's audio-only stream. Nothing here touches the scribe server: the audio
reaches it as an ordinary upload, like a podcast's.

yt-dlp runs as a subprocess of the same Python (`python -m yt_dlp`), so the version in the
virtualenv is the one used. YouTube changes often and old yt-dlp releases stop working: update it
with `uv lock --upgrade-package yt-dlp` and reinstall (docs/cli.md).
"""

import json
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .podcast import AUDIO_LIMIT, Episode, FetchError, NotFound, PodcastError, Unsupported

HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
# The audio-only stream, m4a when there is one (no ffmpeg needed to read it), else whatever is best.
AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio"
RESOLVE_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 3600
# yt-dlp's own words for "this failed for a network reason"; anything else it reports is the video.
TRANSIENT = re.compile(
    r"unable to download|timed out|temporary failure|connection (?:reset|refused|aborted)|"
    r"http error 5\d\d|name or service not known|network is unreachable|incompleteread|remote end closed", re.I,
)


def video_id(link: str) -> str | None:
    """The video id a YouTube link names, or None if it is not a link to a single YouTube video.
    Playlists, channels and the home page are not videos: `list=` next to a `v=` is ignored (one video)."""
    text = link.strip()
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    parts = urllib.parse.urlsplit(text)
    host = (parts.hostname or "").lower()
    if host not in HOSTS:
        return None
    path = [p for p in parts.path.split("/") if p]
    if host.endswith("youtu.be"):
        found = path[0] if path else None
    elif path[:1] == ["watch"]:
        found = (urllib.parse.parse_qs(parts.query).get("v") or [None])[0]
    elif len(path) >= 2 and path[0] in {"shorts", "live", "embed", "v"}:
        found = path[1]
    else:
        found = None
    return found if found and VIDEO_ID.match(found) else None


def is_youtube(link: str) -> bool:
    """Whether a link is on a YouTube host at all (it may still not be a video)."""
    text = link.strip()
    host = urllib.parse.urlsplit(text if "://" in text else "https://" + text).hostname or ""
    return host.lower() in HOSTS


def canonical(link: str) -> str:
    found = video_id(link)
    if not found:
        raise Unsupported(
            f"{link} is not a link to one YouTube video (a watch, youtu.be, shorts or live link); "
            "playlists, channels and search pages are not supported"
        )
    return f"https://www.youtube.com/watch?v={found}"


def run_ytdlp(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run yt-dlp with `args`. The one place a process is started, so tests replace it."""
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--no-progress", "--no-colors", *args]
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as error:
        raise FetchError(f"yt-dlp did not finish within {int(timeout)} seconds") from error
    except OSError as error:
        raise FetchError(f"cannot run yt-dlp: {error}") from error


def failure(result: subprocess.CompletedProcess) -> PodcastError:
    """The error for a yt-dlp run that failed: its own `ERROR:` line says why."""
    lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("ERROR:")]
    if any("No module named yt_dlp" in line for line in lines):
        return FetchError("yt-dlp is not installed in scribe's environment (reinstall scribe: deploy/install.sh)")
    message = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?", "", (errors or lines or ["yt-dlp failed"])[-1])
    if TRANSIENT.search(message):
        return FetchError(f"YouTube download failed: {message}")
    hint = ""
    if re.search(r"sign in to confirm|not a bot|cookies", message, re.I):
        hint = " (YouTube is asking this machine to sign in; scribe has no YouTube login, so this video cannot be fetched)"
    elif re.search(r"private|members|age|unavailable|removed|terminated|live event|premiere|copyright|country", message, re.I):
        hint = " (only public, available videos can be fetched)"
    return NotFound(f"YouTube: {message}{hint}")


def resolve(link: str) -> Episode:
    """The video a link names, with its title, channel, date and length. Downloads no media."""
    url = canonical(link)
    result = run_ytdlp(["--skip-download", "--dump-single-json", "--", url], RESOLVE_TIMEOUT)
    if result.returncode != 0:
        raise failure(result)
    try:
        info = json.loads(result.stdout)
    except ValueError as error:
        raise FetchError("yt-dlp did not return the video's details") from error
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        raise NotFound("YouTube: that is a live stream or a premiere that has not ended; transcribe it when it is over")
    published = None
    if re.fullmatch(r"\d{8}", str(info.get("upload_date") or "")):
        published = datetime.strptime(info["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc)
    channel = info.get("channel") or info.get("uploader")
    return Episode(
        title=info.get("title") or video_id(link) or "video",
        audio_url=url,
        show=channel,
        publisher=channel,
        published=published,
        duration_s=float(info["duration"]) if info.get("duration") else None,
        page_url=info.get("webpage_url") or url,
        source="youtube",
        source_url=link.strip(),
    )


def download(episode: Episode, directory: Path, name: str, limit: int = AUDIO_LIMIT) -> Path:
    """Download the video's audio-only stream into `directory` as `<name>.<ext>` and return the path.
    The video stream is never fetched."""
    directory.mkdir(parents=True, exist_ok=True)
    result = run_ytdlp(
        [
            "-f", AUDIO_FORMAT, "--max-filesize", str(limit), "-o", str(directory / f"{name}.%(ext)s"),
            "--print", "after_move:filepath", "--", canonical(episode.audio_url),
        ],
        DOWNLOAD_TIMEOUT,
    )
    if result.returncode != 0:
        raise failure(result)
    paths = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    path = next((p for p in reversed(paths) if p.is_file()), None)
    if path is None:
        # yt-dlp skips a video over --max-filesize and exits 0 with nothing written.
        raise FetchError(f"yt-dlp downloaded no audio (the file may be over the {limit // (1024 * 1024)} MB limit)")
    return path
