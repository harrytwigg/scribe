"""`scribe`: transcribe a podcast episode (a Spotify, Apple Podcasts, RSS or audio link) or a local
audio file with the scribe server, from a script or an agent (docs/cli.md).

Plain standard-library code: it fetches and downloads here and sends the audio to the server,
which runs the model. Progress goes to stderr and the transcript to stdout.
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
from pathlib import Path

from . import podcast

OK, USAGE, NOT_FOUND, NETWORK, SERVER, FAILED, INTERRUPTED = 0, 2, 3, 4, 5, 6, 130
DEFAULT_SERVER = "http://127.0.0.1:8787"
ENV_FILE = "~/.config/scribe/env"

DESCRIPTION = """\
Transcribe a podcast episode or an audio file with the scribe server.

SOURCE is any of:
  https://open.spotify.com/episode/...   a Spotify episode (its public RSS copy is fetched: Spotify's own
                                         audio is DRM-protected; a Spotify exclusive has no public copy and fails)
  https://open.spotify.com/show/...      a Spotify show: its newest episode (--episode N for older ones)
  https://podcasts.apple.com/...         an Apple Podcasts show or episode (?i=...)
  https://host/feed.xml                  an RSS feed: its newest episode (--episode N)
  https://host/episode.mp3               a direct audio URL
  ./talk.m4a                             a local audio file (wav, flac, ogg, mp3, m4a, ...)
"""

EPILOG = """\
Output: the transcript on stdout (or, with --out, written to that file, and its path printed);
progress and errors on stderr. --format json gives text, timed segments, the episode and the model.

Exit codes: 0 ok; 2 bad usage; 3 no audio for that link (not a public podcast, Spotify exclusive, no matching
episode); 4 network or download failure; 5 scribe server unreachable, token rejected or no model active;
6 transcription failed; 130 interrupted.

The server: --server, else $SCRIBE_URL, else SCRIBE_SERVER_URL in ~/.config/scribe/env, else
http://127.0.0.1:8787. The token: --token, else $SCRIBE_API_TOKEN, else the SCRIBE_SECRETS_FILE
that env file names (a JSON file holding SCRIBE_API_TOKEN).

Examples:
  scribe transcribe https://open.spotify.com/episode/4rOoJ6Egrf8K2IrywzwOMk > episode.txt
  scribe transcribe https://feeds.example.com/show.xml --episode 3 --format json --out ep3.json
  scribe transcribe talk.m4a --speakers --format srt --out talk.srt
  scribe resolve https://open.spotify.com/episode/...      # what would be fetched, no download
"""


class CliError(Exception):
    def __init__(self, message: str, code: int, kind: str, http_status: int | None = None, retry_after: float = 0):
        super().__init__(message)
        self.code, self.kind, self.http_status, self.retry_after = code, kind, http_status, retry_after


def say(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"scribe: {message}", file=sys.stderr, flush=True)


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


def api_token(flag: str | None) -> str | None:
    if token := flag or os.environ.get("SCRIBE_API_TOKEN"):
        return token
    env = read_env_file()
    if token := env.get("SCRIBE_API_TOKEN"):
        return token
    secrets = os.environ.get("SCRIBE_SECRETS_FILE") or env.get("SCRIBE_SECRETS_FILE")
    if secrets:
        try:
            return json.loads(Path(secrets).expanduser().read_text()).get("SCRIBE_API_TOKEN")
        except (OSError, ValueError):
            return None
    return None


class Server:
    def __init__(self, url: str, token: str | None):
        self.url, self.token = url, token

    def call(self, method: str, path: str, *, body=None, headers: dict | None = None, timeout: float = 60, raw: bool = False):
        request = urllib.request.Request(self.url + path, data=body, method=method, headers=headers or {})
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (ValueError, AttributeError):
                pass
            if error.code == 401:
                raise CliError(
                    f"the scribe server at {self.url} rejected the token: set SCRIBE_API_TOKEN, pass --token, or "
                    f"point SCRIBE_SECRETS_FILE at a JSON file holding it", SERVER, "auth",
                ) from error
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
    return source.startswith(("http://", "https://", "spotify:", "open.spotify.com/", "podcasts.apple.com/", "spotify.link/"))


def describe(episode: podcast.Episode) -> str:
    text = f"{episode.show} - {episode.title}" if episode.show else episode.title
    return f"{text} ({int(episode.duration_s // 60)} min)" if episode.duration_s else text


def resolve_source(source: str, nth: int, quiet: bool) -> podcast.Episode:
    try:
        episode = podcast.resolve(podcast.Fetcher(), source, nth)
    except podcast.FetchError as error:
        raise CliError(str(error), NETWORK, "network") from error
    except podcast.PodcastError as error:
        raise CliError(str(error), NOT_FOUND, "not_found") from error
    say(f"found {describe(episode)} [{episode.source}] {episode.audio_url}", quiet)
    match = episode.match
    if match and (match["date"] is False or match["duration"] is False or (match["date"] is None and match["duration"] is None)):
        # Always shown, even with -q: it is the one case where the audio may not be the episode asked for.
        print(
            f"scribe: warning: weak match for {describe(episode)}: title {match['title']}, date "
            f"{match['date']}, duration {match['duration']}; check it is the episode you wanted "
            f"(`scribe resolve` shows it; pass the feed URL with --episode N to be sure)", file=sys.stderr,
        )
    return episode


def fetch_audio(episode: podcast.Episode, directory: Path, quiet: bool) -> Path:
    last = [0.0]

    def progress(done: int, total: int | None) -> None:
        if time.monotonic() - last[0] >= 5:
            last[0] = time.monotonic()
            say(f"downloading {done / 1e6:.1f}" + (f" of {total / 1e6:.1f} MB" if total else " MB"), quiet)

    name = podcast.safe_name(f"{episode.show or ''} {episode.title}")
    try:
        path = podcast.download(podcast.Fetcher(timeout=60), episode.audio_url, directory, name, progress)
    except podcast.PodcastError as error:
        raise CliError(str(error), NETWORK, "network") from error
    say(f"downloaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)", quiet)
    return path


# -- talking to the server ---------------------------------------------------------------------


def check_server(server: Server) -> None:
    """Fail early and accurately: unreachable, wrong token, no model. (A server that refuses a
    request closes the connection while a large upload is still being sent, which the client sees
    only as a reset, so the token is checked before any audio is sent.)"""
    status = server.call("GET", "/api/status")
    if not status.get("active_model"):
        raise CliError(f"the scribe server at {server.url} has no active model; pick one in its GUI", SERVER, "no_model")


def start_job(server: Server, path: Path, query: str, quiet: bool) -> dict:
    waited, dropped = 0.0, 0
    while True:
        busy = server.call("GET", "/api/status").get("transcriptions", {})
        if busy.get("running", 0) >= busy.get("max", 1 << 30) and waited <= 3600:
            say(f"the server is running {busy['running']} transcriptions already; waiting", quiet)
            time.sleep(15)
            waited += 15
            continue
        try:
            with path.open("rb") as audio:
                return server.call(
                    "POST", f"/api/transcriptions?{query}", body=audio, timeout=3600,
                    headers={"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)},
                )
        except CliError as error:
            retry = error.http_status == 429
            if error.kind == "unreachable":
                # A server that refuses a large upload (busy: 429) closes the connection while it
                # is still being sent, which arrives here as a reset. Say why if the server
                # answers now (wrong token, no model); otherwise it was busy: try again.
                check_server(server)
                dropped += 1
                retry = dropped <= 5
            if not retry or waited > 3600:
                raise
            say("the server refused the upload (busy); waiting", quiet)
            time.sleep(error.retry_after or 15)
            waited += error.retry_after or 15


def transcribe_file(server: Server, path: Path, title: str, speakers: bool, quiet: bool) -> dict:
    query = urllib.parse.urlencode({"filename": path.name, "title": title, "diarize": "true" if speakers else "false"})
    say(f"uploading {path.name} to {server.url}", quiet)
    job = start_job(server, path, query, quiet)
    stage, shown = None, -1.0
    try:
        while True:
            job = server.call("GET", f"/api/transcriptions/{job['id']}")
            if job["status"] not in ("queued", "fetching", "decoding", "transcribing"):
                break
            percent = int(job["progress"] * 100) // 10 * 10
            if job["status"] != stage or (job["status"] == "transcribing" and percent != shown):
                stage, shown = job["status"], percent
                say(job["status"] + (f" {percent}%" if job["status"] == "transcribing" and percent else ""), quiet)
            time.sleep(1.5)
    except KeyboardInterrupt:
        try:
            server.call("DELETE", f"/api/transcriptions/{job['id']}")
        except CliError:
            pass
        raise
    if job["status"] != "done":
        raise CliError(f"transcription {job['status']}: {job.get('error')}", FAILED, "transcription_failed")
    return job


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


def cmd_transcribe(args) -> int:
    quiet = args.quiet
    local = Path(args.source).expanduser()
    if not local.is_file() and not looks_like_link(args.source):
        raise CliError(f"{args.source!r} is neither a file that exists nor a link", USAGE, "usage")
    server = Server(server_url(args.server), api_token(args.token))
    check_server(server)
    episode = None
    with tempfile.TemporaryDirectory(prefix="scribe-") as scratch:
        if local.is_file():
            audio, title = local, local.stem
        else:
            episode = resolve_source(args.source, args.episode, quiet)
            audio = fetch_audio(episode, Path(args.keep_audio) if args.keep_audio else Path(scratch), quiet)
            title = f"{episode.show}: {episode.title}" if episode.show else episode.title
        job = transcribe_file(server, audio, title, args.speakers, quiet)
    result, session_id = job["result"], job["session_id"]
    if args.format in ("srt", "md"):
        output = server.call("GET", f"/api/sessions/{session_id}/export?format={args.format}", raw=True).decode()
    elif args.format == "json":
        output = json.dumps(
            {
                "title": job["title"], "source": args.source, "episode": episode.to_json() if episode else None,
                "model": result["model"], "duration_s": result["duration"], "asr_ms": result["asr_ms"],
                "speakers": result["speakers"], "text": result["text"], "segments": result["segments"],
                "session_id": None if args.no_save else session_id, "server": server.url,
            },
            indent=2, ensure_ascii=False,
        )
    else:
        output = render_text(result, args.timestamps, args.speakers)
    if args.no_save:
        server.call("DELETE", f"/api/sessions/{session_id}")
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output.rstrip("\n") + "\n")
        print(out)
    else:
        print(output)
    say(f"done: {result['duration'] / 60:.1f} min of audio in {result['asr_ms'] / 1000:.1f} s with {result['model']}", quiet)
    return OK


def cmd_resolve(args) -> int:
    if not looks_like_link(args.source):
        raise CliError(f"{args.source!r} is not a link", USAGE, "usage")
    fetcher = podcast.Fetcher()
    try:
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
    server = Server(server_url(args.server), api_token(args.token))
    print(json.dumps(server.call("GET", "/api/status"), indent=2))
    return OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scribe", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="{transcribe,resolve,download,status}")

    def common(sub, server=True):
        sub.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")
        if server:
            sub.add_argument("--server", help="scribe server URL (default $SCRIBE_URL or http://127.0.0.1:8787)")
            sub.add_argument("--token", help="API token (default $SCRIBE_API_TOKEN or the secrets file)")

    def episode_flag(sub):
        sub.add_argument("--episode", type=int, default=1, metavar="N", help="of a show or feed: the Nth newest episode (default 1, the newest)")

    t = commands.add_parser("transcribe", help="download (if a link) and transcribe", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    t.add_argument("source", metavar="SOURCE", help="a podcast link or an audio file")
    t.add_argument("--format", choices=["text", "json", "srt", "md"], default="text", help="text (default), json, srt or md")
    t.add_argument("--out", metavar="FILE", help="write the transcript here and print the path, instead of stdout")
    t.add_argument("--speakers", action="store_true", help="separate speakers (Speaker 1, Speaker 2, ...)")
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
