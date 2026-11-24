"""`scribe`: transcribe a podcast episode (a Spotify, Apple Podcasts, RSS or audio link), a YouTube
video or a local audio file with the scribe server, from a script or an agent (docs/cli.md).

Standard-library code, plus yt-dlp for YouTube: it fetches and downloads here and sends the audio to
the server, which runs the model. Progress goes to stderr and the transcript to stdout.
"""

import argparse
import json
import os
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import podcast, youtube

OK, USAGE, NOT_FOUND, NETWORK, SERVER, FAILED, INTERRUPTED = 0, 2, 3, 4, 5, 6, 130
DEFAULT_SERVER = "http://127.0.0.1:8787"
ENV_FILE = "~/.config/scribe/env"

DESCRIPTION = """\
Transcribe a podcast episode, a YouTube video or an audio file with the scribe server.

SOURCE is any of:
  https://open.spotify.com/episode/...   a Spotify episode (its public RSS copy is fetched: Spotify's own
                                         audio is DRM-protected; a Spotify exclusive has no public copy and fails)
  https://open.spotify.com/show/...      a Spotify show: its newest episode (--episode N for older ones)
  https://podcasts.apple.com/...         an Apple Podcasts show or episode (?i=...)
  https://host/feed.xml                  an RSS feed: its newest episode (--episode N)
  https://host/episode.mp3               a direct audio URL
  https://www.youtube.com/watch?v=...    a public YouTube video (also youtu.be, /shorts/, /live/): only its audio is
                                         downloaded, with yt-dlp
  ./talk.m4a                             a local audio file (wav, flac, ogg, mp3, m4a, ...)
"""

EPILOG = """\
Output: the transcript on stdout (or, with --out, written to that file, and its path printed);
progress and errors on stderr. --format json gives text, timed segments, the episode and the model.

Exit codes: 0 ok; 2 bad usage; 3 no audio for that link (not a public podcast, Spotify exclusive, no matching
episode; a private, removed, age-restricted or live YouTube video, or one YouTube makes this machine sign in for); 4 network or download failure; 5 scribe server unreachable or no model active;
6 transcription failed; 130 interrupted.

The server: --server, else $SCRIBE_URL, else SCRIBE_SERVER_URL in ~/.config/scribe/env, else
http://127.0.0.1:8787. There is no token to give.

Examples:
  scribe transcribe https://open.spotify.com/episode/4rOoJ6Egrf8K2IrywzwOMk > episode.txt
  scribe transcribe https://feeds.example.com/show.xml --episode 3 --format json --out ep3.json
  scribe transcribe https://www.youtube.com/watch?v=jNQXAC9IVRw --format json --out zoo.json
  scribe transcribe talk.m4a --speakers --format srt --out talk.srt
  scribe resolve https://open.spotify.com/episode/...      # what would be fetched, no download
"""


class CliError(Exception):
    def __init__(self, message: str, code: int, kind: str, http_status: int | None = None, retry_after: float = 0):
        super().__init__(message)
        self.code, self.kind, self.http_status, self.retry_after = code, kind, http_status, retry_after


def say(message: str, quiet: bool = False, tag: str = "") -> None:
    if not quiet:
        print(f"scribe: {tag}{message}", file=sys.stderr, flush=True)


# -- settings ----------------------------------------------------------------------------------


def read_env_file() -> dict[str, str]:
    path = Path(os.environ.get("SCRIBE_ENV_FILE", ENV_FILE)).expanduser()
    values = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("\"'")
    return values


def server_url(flag: str | None) -> str:
    url = flag or os.environ.get("SCRIBE_URL") or read_env_file().get("SCRIBE_SERVER_URL") or DEFAULT_SERVER
    return url.rstrip("/")


class Server:
    def __init__(self, url: str):
        self.url = url

    def call(self, method: str, path: str, *, body=None, headers: dict | None = None, timeout: float = 60, raw: bool = False):
        request = urllib.request.Request(self.url + path, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (ValueError, AttributeError):
                pass
            code = SERVER if error.code in (503, 409, 413, 429) else FAILED
            retry = error.headers.get("Retry-After", "")
            raise CliError(
                f"the scribe server said {error.code}: {detail}", code, "server", error.code,
                float(retry) if retry.isdigit() else 0,
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise CliError(
                f"cannot reach the scribe server at {self.url} ({getattr(error, 'reason', error)}); is it running "
                f"(systemctl --user status scribe-server) and is --server / $SCRIBE_URL right?", SERVER, "unreachable",
            ) from error
        return data if raw else json.loads(data)


# -- getting the audio -------------------------------------------------------------------------


def looks_like_link(source: str) -> bool:
    return source.startswith((
        "http://", "https://", "spotify:", "open.spotify.com/", "podcasts.apple.com/", "spotify.link/",
        "youtube.com/", "www.youtube.com/", "m.youtube.com/", "music.youtube.com/", "youtu.be/",
    ))


def describe(episode: podcast.Episode) -> str:
    text = f"{episode.show} - {episode.title}" if episode.show else episode.title
    return f"{text} ({int(episode.duration_s // 60)} min)" if episode.duration_s else text


def resolve_source(source: str, nth: int, quiet: bool, tag: str = "") -> podcast.Episode:
    try:
        episode = youtube.resolve(source) if youtube.is_youtube(source) else podcast.resolve(podcast.Fetcher(), source, nth)
    except podcast.FetchError as error:
        raise CliError(str(error), NETWORK, "network") from error
    except podcast.PodcastError as error:
        raise CliError(str(error), NOT_FOUND, "not_found") from error
    say(f"found {describe(episode)} [{episode.source}] {episode.audio_url}", quiet, tag)
    match = episode.match
    if match and (match["date"] is False or match["duration"] is False or (match["date"] is None and match["duration"] is None)):
        # Always shown, even with -q: it is the one case where the audio may not be the episode asked for.
        print(
            f"scribe: {tag}warning: weak match for {describe(episode)}: title {match['title']}, date "
            f"{match['date']}, duration {match['duration']}; check it is the episode you wanted "
            f"(`scribe resolve` shows it; pass the feed URL with --episode N to be sure)", file=sys.stderr,
        )
    return episode


def fetch_audio(episode: podcast.Episode, directory: Path, quiet: bool, tag: str = "") -> Path:
    last = [0.0]

    def progress(done: int, total: int | None) -> None:
        if time.monotonic() - last[0] >= 5:
            last[0] = time.monotonic()
            say(f"downloading {done / 1e6:.1f}" + (f" of {total / 1e6:.1f} MB ({done * 100 // total}%)" if total else " MB"), quiet, tag)

    name = podcast.safe_name(f"{episode.show or ''} {episode.title}")
    try:
        if episode.source == "youtube":
            say("downloading the audio with yt-dlp", quiet, tag)
            path = youtube.download(episode, directory, name)
        else:
            path = podcast.download(podcast.Fetcher(timeout=60), episode.audio_url, directory, name, progress)
    except podcast.PodcastError as error:
        raise CliError(str(error), NETWORK, "network") from error
    say(f"downloaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)", quiet, tag)
    return path


# -- talking to the server ---------------------------------------------------------------------


def check_server(server: Server) -> None:
    """Fail early and accurately: unreachable, no model. (A server that refuses a
    request closes the connection while a large upload is still being sent, which the client sees
    only as a reset, so the server is checked before any audio is sent.)"""
    status = server.call("GET", "/api/status")
    if not status.get("active_model"):
        raise CliError(f"the scribe server at {server.url} has no active model; pick one in its GUI", SERVER, "no_model")


def start_job(server: Server, path: Path, query: str, quiet: bool, tag: str = "") -> dict:
    """Upload the audio; the server queues it behind any running transcriptions."""
    waited, dropped = 0.0, 0
    while True:
        try:
            with path.open("rb") as audio:
                return server.call(
                    "POST", f"/api/transcriptions?{query}", body=audio, timeout=3600,
                    headers={"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)},
                )
        except CliError as error:
            retry = error.http_status == 429  # the server's queue is full
            if error.kind == "unreachable":
                # A server that refuses a large upload closes the connection while it is still
                # being sent, which arrives here as a reset. Say why if the server answers now
                # (no model); otherwise try again.
                check_server(server)
                dropped += 1
                retry = dropped <= 5
            if not retry or waited > 3600:
                raise
            say("the server refused the upload (its queue is full); waiting", quiet, tag)
            time.sleep(error.retry_after or 15)
            waited += error.retry_after or 15


@dataclass
class Item:
    """One source on its way through: found, downloaded, uploaded, queued, transcribed."""

    index: int
    source: str
    episode: podcast.Episode | None = None
    title: str = ""
    job: dict | None = None  # the server's latest view of the job
    error: CliError | None = None
    shown: str = ""  # what was last said about it
    tag: str = ""


def prepare(server: Server, item: Item, args, scratch: Path, quiet: bool) -> None:
    """Find, download and upload one source; leaves its job running (or queued) on the server."""
    local = Path(item.source).expanduser()
    if local.is_file():
        audio, item.title, delete = local, local.stem, False
    else:
        item.episode = resolve_source(item.source, args.episode, quiet, item.tag)
        keep = Path(args.keep_audio) if args.keep_audio else None
        audio = fetch_audio(item.episode, keep or scratch, quiet, item.tag)
        item.title = f"{item.episode.show}: {item.episode.title}" if item.episode.show else item.episode.title
        delete = keep is None
    fields = {"filename": audio.name, "title": item.title, "diarize": "true" if args.speakers else "false"}
    if args.num_speakers:
        fields["speakers"] = args.num_speakers
    query = urllib.parse.urlencode(fields)
    say(f"uploading {audio.name} to {server.url}", quiet, item.tag)
    item.job = start_job(server, audio, query, quiet, item.tag)
    if delete:
        audio.unlink(missing_ok=True)


def progress_line(job: dict) -> str:
    if job["status"] == "queued":
        return f"queued (position {job['queue_position']})"
    if job["status"] == "transcribing":
        percent = int(job["stage_progress"] * 100) // 10 * 10
        return job["step"] or ("transcribing" + (f" {percent}%" if percent else ""))
    return job["status"]


def await_jobs(server: Server, items: list[Item], quiet: bool) -> None:
    """Poll every started job until all have finished; fills in `item.job` or `item.error`."""
    try:
        while True:
            waiting = [i for i in items if i.job and not i.error and i.job["status"] in ("queued", "fetching", "decoding", "transcribing")]
            if not waiting:
                return
            for item in waiting:
                item.job = server.call("GET", f"/api/transcriptions/{item.job['id']}")
                line = progress_line(item.job)
                if item.job["status"] in ("queued", "fetching", "decoding", "transcribing") and line != item.shown:
                    item.shown = line
                    say(line, quiet, item.tag)
                elif item.job["status"] not in ("queued", "fetching", "decoding", "transcribing") and item.job["status"] != "done":
                    item.error = CliError(
                        f"transcription {item.job['status']}: {item.job.get('error')}", FAILED, "transcription_failed"
                    )
            if any(i.job and i.job["status"] in ("queued", "fetching", "decoding", "transcribing") for i in waiting):
                time.sleep(1.5)
    except KeyboardInterrupt:
        for item in items:
            if item.job and item.job["status"] in ("queued", "fetching", "decoding", "transcribing"):
                try:
                    server.call("DELETE", f"/api/transcriptions/{item.job['id']}")
                except CliError:
                    pass
        raise


def clock(seconds: float) -> str:
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def render_text(result: dict, timestamps: bool, speakers: bool) -> str:
    segments = result["segments"]
    if not timestamps and not (speakers and result["speakers"] > 1):
        return result["text"]
    lines = []
    for seg in segments:
        who = f"{seg['speaker']}: " if result["speakers"] > 1 else ""
        lines.append(f"{'[' + clock(seg['start']) + '] ' if timestamps else ''}{who}{seg['text']}")
    return "\n".join(lines)


# -- commands ----------------------------------------------------------------------------------

EXTENSIONS = {"text": "txt", "json": "json", "srt": "srt", "md": "md"}
PARTIAL = 7


def read_sources(args) -> list[str]:
    sources = list(args.sources)
    if args.sources_from:
        try:
            text = sys.stdin.read() if args.sources_from == "-" else Path(args.sources_from).expanduser().read_text()
        except OSError as error:
            raise CliError(f"cannot read {args.sources_from}: {error}", USAGE, "usage") from error
        sources += [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not sources:
        raise CliError("give at least one SOURCE (a link or an audio file), or --from FILE", USAGE, "usage")
    for source in sources:
        if not Path(source).expanduser().is_file() and not looks_like_link(source):
            raise CliError(f"{source!r} is neither a file that exists nor a link", USAGE, "usage")
    return sources


def render(server: Server, item: Item, args) -> str:
    """The finished item as the requested format."""
    result, session_id = item.job["result"], item.job["session_id"]
    if args.format in ("srt", "md"):
        return server.call("GET", f"/api/sessions/{session_id}/export?format={args.format}", raw=True).decode()
    if args.format == "json":
        return json.dumps(json_result(server, item, args), indent=2, ensure_ascii=False)
    return render_text(result, args.timestamps, args.speakers)


def json_result(server: Server, item: Item, args) -> dict:
    result = item.job["result"]
    return {
        "title": item.job["title"], "source": item.source, "episode": item.episode.to_json() if item.episode else None,
        "model": result["model"], "duration_s": result["duration"], "asr_ms": result["asr_ms"],
        "speakers": result["speakers"], "text": result["text"], "segments": result["segments"],
        "session_id": None if args.no_save else item.job["session_id"], "server": server.url,
    }


def error_json(error: CliError) -> dict:
    return {"kind": error.kind, "message": str(error), "exit_code": error.code}


def cmd_transcribe(args) -> int:
    quiet = args.quiet
    if args.num_speakers is not None:
        if not 1 <= args.num_speakers <= 20:
            raise CliError("--num-speakers is between 1 and 20", USAGE, "usage")
        args.speakers = True
    sources = read_sources(args)
    many = len(sources) > 1
    server = Server(server_url(args.server))
    check_server(server)
    items = [Item(i, source, tag=f"[{i}/{len(sources)}] " if many else "") for i, source in enumerate(sources, 1)]
    with tempfile.TemporaryDirectory(prefix="scribe-") as scratch:
        # Fetch and upload one after another: the server queues them and starts the next as soon as
        # a slot is free, so the transcribing of one overlaps the downloading of the next.
        for item in items:
            try:
                prepare(server, item, args, Path(scratch), quiet)
            except CliError as error:
                if not many:
                    raise
                item.error = error
                print(f"scribe: {item.tag}error: {error}", file=sys.stderr)
        known = {id(i) for i in items if i.error}
        await_jobs(server, items, quiet)
        for item in items:
            if item.error and id(item) not in known and many:  # a job that failed on the server
                print(f"scribe: {item.tag}error: {item.error}", file=sys.stderr)
    if not many and items[0].error:
        raise items[0].error
    emit(server, items, args, many)
    for item in items:
        if item.job and item.job["status"] == "done":
            if args.no_save:
                server.call("DELETE", f"/api/sessions/{item.job['session_id']}")
            say(
                f"done: {item.job['result']['duration'] / 60:.1f} min of audio in {item.job['result']['asr_ms'] / 1000:.1f} s "
                f"with {item.job['result']['model']}", quiet, item.tag,
            )
    failed = [i.error for i in items if i.error]
    if not failed:
        return OK
    return failed[0].code if len(failed) == len(items) else PARTIAL


def emit(server: Server, items: list[Item], args, many: bool) -> list[str]:
    """Print or write each finished transcript; returns what was printed."""
    done = [i for i in items if i.job and i.job["status"] == "done" and not i.error]
    printed: list[str] = []
    if args.format == "json" and many and not args.out:
        # One JSON array, an entry per source in order, failures included.
        entries = [
            {"ok": True, **json_result(server, i, args)} if i in done else {"ok": False, "source": i.source, "error": error_json(i.error)}
            for i in items
        ]
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return printed
    if args.out and many:
        directory = Path(args.out).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        for item in done:
            path = directory / f"{item.index:02d}-{podcast.safe_name(item.title, 'transcript')}.{EXTENSIONS[args.format]}"
            path.write_text(render(server, item, args).rstrip("\n") + "\n")
            print(path)
            printed.append(str(path))
        if args.format == "json":  # and the failures, so a script has the whole outcome in one place
            index = directory / "index.json"
            index.write_text(json.dumps([
                {"ok": True, "source": i.source, "file": printed[done.index(i)]} if i in done else
                {"ok": False, "source": i.source, "error": error_json(i.error)} for i in items], indent=2) + "\n")
        return printed
    for item in done:
        output = render(server, item, args)
        if args.out:  # one source
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(output.rstrip("\n") + "\n")
            print(out)
        else:
            if many:
                print(f"=== {item.title} ({item.source}) ===")
            print(output)
        printed.append(output)
    return printed


def cmd_resolve(args) -> int:
    if not looks_like_link(args.source):
        raise CliError(f"{args.source!r} is not a link", USAGE, "usage")
    fetcher = podcast.Fetcher()
    try:
        if youtube.is_youtube(args.source):
            episodes = [youtube.resolve(args.source)]  # one video: --list has nothing more to list
        else:
            episodes = podcast.list_episodes(fetcher, args.source, args.limit) if args.list else [podcast.resolve(fetcher, args.source, args.episode)]
    except podcast.FetchError as error:
        raise CliError(str(error), NETWORK, "network") from error
    except podcast.PodcastError as error:
        raise CliError(str(error), NOT_FOUND, "not_found") from error
    print(json.dumps([e.to_json() for e in episodes] if args.list else episodes[0].to_json(), indent=2, ensure_ascii=False))
    return OK


def cmd_download(args) -> int:
    episode = resolve_source(args.source, args.episode, args.quiet)
    print(fetch_audio(episode, Path(args.out).expanduser(), args.quiet))
    return OK


def cmd_status(args) -> int:
    server = Server(server_url(args.server))
    print(json.dumps(server.call("GET", "/api/status"), indent=2))
    return OK


def cmd_recording(args) -> int:
    """What is recorded of a browser call, and how many gaps its transcript has."""
    server = Server(server_url(args.server))
    print(json.dumps(server.call("GET", f"/api/sessions/{args.session}/recording"), indent=2))
    return OK


def cmd_reprocess(args) -> int:
    """Fill a call's gaps from its recording (or all of it again) and wait for the result."""
    server = Server(server_url(args.server))
    job = server.call(
        "POST", f"/api/sessions/{args.session}/reprocess", body=json.dumps({"mode": "all" if args.all else "gaps"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    if args.no_wait:
        print(json.dumps(job, indent=2))
        return OK
    say(f"reprocessing {args.session} ({'all of it' if args.all else 'the gaps'})", args.quiet)
    last = ""
    while job["status"] in ("queued", "fetching", "decoding", "transcribing"):
        time.sleep(1)
        job = server.call("GET", f"/api/transcriptions/{job['id']}")
        progress = f"{job['progress'] * 100:.0f}%"
        if progress != last:
            say(progress, args.quiet)
            last = progress
    if job["status"] != "done":
        raise CliError(f"reprocessing failed: {job.get('error') or job['status']}", FAILED, "failed")
    print(json.dumps(job["result"], indent=2))
    return OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scribe", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="{transcribe,resolve,download,status,recording,reprocess}")

    def common(sub, server=True):
        sub.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")
        if server:
            sub.add_argument("--server", help="scribe server URL (default $SCRIBE_URL or http://127.0.0.1:8787)")

    def episode_flag(sub):
        sub.add_argument("--episode", type=int, default=1, metavar="N", help="of a show or feed: the Nth newest episode (default 1, the newest)")

    t = commands.add_parser("transcribe", help="download (if a link) and transcribe", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    t.add_argument("sources", metavar="SOURCE", nargs="*", help="a podcast or YouTube link or an audio file; several are queued and run in turn")
    t.add_argument("--from", dest="sources_from", metavar="FILE", help="also read SOURCEs from FILE, one per line ('-' is stdin; blank lines and # comments are skipped)")
    t.add_argument("--format", choices=["text", "json", "srt", "md"], default="text", help="text (default), json, srt or md")
    t.add_argument("--out", metavar="FILE", help="write the transcript here and print the path, instead of stdout; with several SOURCEs a directory, one file per source")
    t.add_argument("--speakers", action="store_true", help="separate speakers (Speaker 1, Speaker 2, ...)")
    t.add_argument("--num-speakers", type=int, metavar="N", help="separate speakers, and there are exactly N of them (use it when you know: it is more accurate than letting scribe count)")
    t.add_argument("--timestamps", action="store_true", help="text format: start each line with its [hh:mm:ss] time")
    t.add_argument("--no-save", action="store_true", help="do not keep the transcript in the server's History")
    t.add_argument("--keep-audio", metavar="DIR", help="keep the downloaded audio in DIR (default: deleted)")
    episode_flag(t)
    common(t)
    t.set_defaults(run=cmd_transcribe)

    r = commands.add_parser("resolve", help="show the episode a link names and its audio URL, without downloading")
    r.add_argument("source", metavar="LINK")
    r.add_argument("--list", action="store_true", help="list the show's newest episodes instead (use with --limit)")
    r.add_argument("--limit", type=int, default=20)
    episode_flag(r)
    common(r, server=False)
    r.set_defaults(run=cmd_resolve)

    d = commands.add_parser("download", help="download a link's audio without transcribing it")
    d.add_argument("source", metavar="LINK")
    d.add_argument("--out", metavar="DIR", default=".", help="directory to save into (default .)")
    episode_flag(d)
    common(d, server=False)
    d.set_defaults(run=cmd_download)

    s = commands.add_parser("status", help="show the server's state: active model, queue")
    common(s)
    s.set_defaults(run=cmd_status)

    rec = commands.add_parser("recording", help="show what is recorded of a browser call (segments, size, gaps in its transcript)")
    rec.add_argument("session", metavar="SESSION", help="the call's id (the last part of its address in the GUI)")
    common(rec)
    rec.set_defaults(run=cmd_recording)

    rp = commands.add_parser(
        "reprocess", help="fill the gaps in a browser call's transcript from its recording",
        description="Transcribe the recorded audio behind each [unavailable] line of a call and put the text in its place, "
        "with the speakers the call had. --all transcribes the whole recording again instead and replaces the transcript.",
    )
    rp.add_argument("session", metavar="SESSION", help="the call's id")
    rp.add_argument("--all", action="store_true", help="transcribe the whole recording again, replacing the transcript")
    rp.add_argument("--no-wait", action="store_true", help="start it and print the job instead of waiting for it")
    common(rp)
    rp.set_defaults(run=cmd_reprocess)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        return args.run(args)
    except CliError as error:
        print(f"scribe: error: {error}", file=sys.stderr)
        if getattr(args, "format", None) == "json":
            print(json.dumps({"error": {"kind": error.kind, "message": str(error), "exit_code": error.code}}))
        return error.code
    except KeyboardInterrupt:
        print("scribe: interrupted", file=sys.stderr)
        return INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
