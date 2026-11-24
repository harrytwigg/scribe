# The `scribe` command

Transcribe a podcast episode, a YouTube video or an audio file with the scribe server, from a script or an agent.
Non-interactive: arguments in, transcript on stdout, progress on stderr, a distinct exit code per
kind of failure.

```sh
scribe transcribe https://open.spotify.com/episode/3tdrNundncNjVFHyNy9zXM   # prints the transcript
scribe transcribe https://www.youtube.com/watch?v=jNQXAC9IVRw --format json # a YouTube video's audio
scribe transcribe talk.m4a --speakers --format srt --out talk.srt           # writes talk.srt, prints its path
scribe resolve https://open.spotify.com/episode/...                        # what would be fetched; no download
scribe transcribe LINK1 LINK2 talk.m4a --out transcripts/ --format md       # a queue: one file per source
scribe transcribe --from links.txt --format json > all.json                # sources from a file, one per line
```

On scribe-host it is installed by `deploy/install.sh` at `~/.local/bin/scribe`. From a checkout:
`cd server && uv run scribe ...`. It needs the scribe server running (`systemctl --user status
scribe-server`) with a model active (the default, Parakeet, is activated at start).

## Commands

| Command | Does |
| --- | --- |
| `scribe transcribe SOURCE...` | Fetch the audio (if a SOURCE is a link) and transcribe it. Several SOURCEs, or `--from FILE`, are queued and run in turn. |
| `scribe resolve LINK` | Print the episode a link names and its audio URL as JSON. `--list` lists the show's newest episodes. Downloads nothing. |
| `scribe download LINK --out DIR` | Download the audio only; prints the file's path. |
| `scribe status` | The server's state as JSON: active model, queue depth. |
| `scribe recording SESSION` | What is recorded of a browser call (docs/spec.md, 4.4): segments, size and time per channel, and how many `[unavailable]` gaps its transcript has. |
| `scribe reprocess SESSION [--all]` | Fill those gaps from the recording and wait for it, printing the result as JSON; `--all` transcribes the whole recording again and replaces the transcript; `--no-wait` just starts it. SESSION is the id in the call's address in the GUI (`#history/<id>`). |

**SOURCE** (or LINK) is any of:

| Source | Resolves to |
| --- | --- |
| `https://open.spotify.com/episode/<id>` (also `spotify:episode:<id>`, `spotify.link/...`, `/intl-xx/` forms) | That episode's public copy (see below) |
| `https://open.spotify.com/show/<id>` | The show's newest episode, or `--episode N` for the Nth newest |
| `https://podcasts.apple.com/<cc>/podcast/<slug>/id<id>` | The show's newest episode, or `--episode N`; with `?i=<id>`, that episode (among the ~300 newest Apple lists) |
| An RSS feed URL, or a web page that links one | The feed's newest episode, or `--episode N` |
| A direct audio URL (`.mp3`, `.m4a`, ...) | That file |
| `https://www.youtube.com/watch?v=<id>` (also `youtu.be/<id>`, `/shorts/<id>`, `/live/<id>`, `/embed/<id>`, `m.` and `music.` hosts) | That video's audio-only stream, fetched with yt-dlp (see "YouTube" below) |
| A local file | That file (anything libsndfile or FFmpeg reads: wav, flac, ogg, opus, mp3, m4a, aac, mp4) |

### How a Spotify link is turned into audio

Spotify's own audio is DRM-protected and not downloadable. scribe does what podcastmp3.com does
instead: it reads the episode's title, show, release date and length from Spotify's public pages,
finds the show's public RSS feed through Apple's podcast directory, matches the episode in the
feed (title first; date within a couple of days and length within a few percent confirm it), and
downloads the feed's `<enclosure>` audio. It refuses rather than guess: an episode whose title is the
same but whose date and length both disagree (another week's "Weekly Roundup"), or that differs only
in a number ("Part 1" / "Part 2", "Episode 12" / "13"), is not accepted unless the date or length agrees. This is the same recording most listeners hear, though
Spotify and the feed may insert different ads.

A **Spotify exclusive** (or any show not in Apple's directory with a public feed) has no public
copy, so it cannot be fetched from its link: `scribe` exits with code 3 and says so. If you have
the show's RSS URL, an mp3 URL or the audio file, pass that instead. Private or region-locked
Spotify pages likewise fail.

### YouTube

A link to one public YouTube video is downloaded as **audio only** with [yt-dlp](https://github.com/yt-dlp/yt-dlp)
(run as `python -m yt_dlp` from scribe's own virtualenv, on the machine running `scribe`, not on the
server) and then uploaded like any other audio. The title, channel and upload date are in the JSON's
`episode` (`source` is `youtube`; `show` and `publisher` are the channel); `scribe resolve LINK` shows
them without downloading, and `scribe download LINK --out DIR` keeps the audio. A 19-minute video
takes about 6 s to download and 16 s to transcribe.

- **One video only.** A `list=` next to a `v=` is ignored. A playlist, channel (`/@name`) or search link
  is refused with exit 3. `--episode N` is ignored, and `resolve --list` lists just the one video.
- **Only public videos.** Private, removed, members-only, age-restricted, region-locked and not-yet-ended
  live videos fail with exit 3 and YouTube's own reason. scribe has no YouTube login or cookies, so a video
  YouTube makes this machine "sign in to confirm you're not a bot" for fails the same way; a network failure
  (exit 4) may be transient.
- **YouTube's terms** restrict downloading content, and YouTube can stop serving a machine that does it a lot.
  Use it for what you would otherwise watch or quote: not bulk, and not other people's private or call audio.
  There is no consent step here either.
- **Keep yt-dlp current.** YouTube changes often and an old yt-dlp stops working (the error says so). Update
  it with `cd server && uv lock --upgrade-package yt-dlp`, then `deploy/install.sh`.
  yt-dlp warns that it wants a JavaScript runtime (deno) to solve YouTube's download-speed challenge; it
  works without one for ordinary videos, and scribe does not enable yt-dlp's downloading of solver
  scripts from GitHub.
- The transcript is the speech, transcribed by the model. YouTube's own captions are a different (faster, less
  accurate) source that scribe does not use.
- The server's GUI and `POST /api/transcriptions/link` do not take YouTube links: the server does not run yt-dlp.

## Options of `transcribe`

| Option | Meaning |
| --- | --- |
| `--format text\|json\|srt\|md` | Output format (default `text`). `json` is the one for programs. |
| `--out FILE` | Write the transcript to FILE (parents created) and print the path, instead of printing the transcript. With several sources, a directory: `01-<title>.<ext>`, `02-...`, one path printed per file (and an `index.json` with `--format json`). |
| `--from FILE` | Also read SOURCEs from FILE, one per line (`-` is stdin; blank lines and `#` comments are skipped). |
| `--speakers` | Separate speakers: `Speaker 1`, `Speaker 2`, ... (slower; CPU diarization on the server). scribe counts the speakers itself and folds scraps (a few seconds of someone's voice on a bad line) into the real ones. |
| `--num-speakers N` | Separate speakers, ending with at most `N` of them (1 to 20): the closest-sounding voices are merged until `N` remain, scraps are not folded below `N`, and if scribe found fewer it clusters again more finely first (that second pass roughly doubles the separation time). Fewer than `N` can still come out if the recording really has fewer distinct voices. Use it when you know how many people speak. Implies `--speakers`. |
| `--timestamps` | `text` format only: start each line with `[hh:mm:ss]`. |
| `--episode N` | Of a show or feed: the Nth newest episode (default 1). |
| `--no-save` | Do not keep the transcript in the server's History (default: kept, titled with the episode). |
| `--keep-audio DIR` | Keep the downloaded audio in DIR (default: deleted after the upload). |
| `-q` | No progress on stderr. |
| `--server URL` | The scribe server. Default below. |

### JSON output

```json
{
  "title": "Up First from NPR: Democratic Surrogates Campaign, ...",
  "source": "https://open.spotify.com/episode/3tdrNundncNjVFHyNy9zXM",
  "episode": {"title": "...", "show": "Up First from NPR", "publisher": "NPR", "published": "2026-10-22T10:08:23+00:00",
              "duration_s": 746.0, "audio_url": "https://...mp3", "feed_url": "https://feeds.npr.org/510318/podcast.xml",
              "source": "spotify", "source_url": "https://open.spotify.com/episode/...",
              "match": {"title": 1.0, "overall": 1.0, "date": true, "duration": true}},
  "model": "parakeet-tdt-0.6b-v2", "duration_s": 874.9, "asr_ms": 11367, "speakers": 1,
  "text": "full transcript ...",
  "segments": [{"speaker": "Speaker 1", "start": 0.15, "end": 22.54, "text": "..."}],
  "session_id": "b9ac0d3bdb97", "server": "http://127.0.0.1:8787"
}
```

`episode` is `null` for a local file. For a Spotify link, `episode.match` says how sure the match is:
`title` and `overall` are 0 to 1, `date` and `duration` say whether Spotify's and the feed's agreed
(`null`: the feed did not say). An identical title alone is accepted only when nothing contradicts it;
treat `date` and `duration` both `null` or a `title` below 1 as a weaker match worth a glance. `segments` are stretches of speech (up to 25 s each, or whole
turns with `--speakers`) with start and end in seconds. `session_id` is the History entry
(`null` with `--no-save`).

### Several sources

The sources are fetched and uploaded one after another and the server queues them (2 run at once;
the rest wait their turn, in order), so the transcribing of one overlaps the downloading of the
next. Without `--out`, text/srt/md transcripts are printed in order, each under a `=== title (source) ===`
line, and `--format json` prints one JSON array with an entry per source, in order:
`{"ok": true, ...the object above}` or `{"ok": false, "source", "error": {"kind", "message", "exit_code"}}`.
A source that fails does not stop the others; its error is on stderr too. A SOURCE that is neither a link
nor an existing file is a usage error (exit 2) before anything is sent.

On failure with `--format json` (one source), stdout is `{"error": {"kind", "message", "exit_code"}}` (and
the message is on stderr either way).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Transcript produced |
| 2 | Bad usage (SOURCE is neither an existing file nor a link) |
| 3 | No audio for that link: not a public podcast, Spotify exclusive, no matching episode, unsupported link kind; a private, removed, age-restricted or live YouTube video, a YouTube playlist or channel link. Retrying will not help. |
| 4 | Network or download failure (the link, the feed or the audio host). May be transient. |
| 5 | The scribe server is unreachable, has no active model, or refused the upload (too large) |
| 6 | The server failed to transcribe (undecodable audio, internal error) |
| 7 | Several sources: some succeeded and some failed (each failure is in the JSON array and on stderr). If every source failed, the first failure's code is used |
| 130 | Interrupted (the running transcription is cancelled on the server) |

## Configuration

- **Server:** `--server`, else `$SCRIBE_URL`, else `SCRIBE_SERVER_URL` in `~/.config/scribe/env`,
  else `http://127.0.0.1:8787`.
- There is no API token: on scribe-host `scribe` works with no setup, and elsewhere only the server URL is needed.
- `$SCRIBE_ENV_FILE` overrides the env file's path.

## Limits and behaviour worth knowing

- Audio is uploaded to the server (one request, up to 512 MB; streamed to a temporary file there and
  deleted once decoded) and transcribed as a background job; the CLI polls it. The server runs at most
  2 transcriptions at once and queues up to 50 more, first come first served; progress lines say
  `queued (position N)`, `decoding`, `transcribing 40%`. A 15-minute episode takes about 15 s with Parakeet on the GPU. A long episode
  yields to live calls: its speech is transcribed chunk by chunk through the same queue.
- `--speakers` on a long recording holds the GPU for that recording's whole transcription in one
  piece (a live call waits for it); prefer it for recordings of an hour or less.
- Transcripts are kept in History (the GUI's History tab, with Markdown/SRT/JSON export) unless
  `--no-save`. Audio is never kept on the server.
- Text is English-only with the default model (Parakeet v2); pick another model in the GUI for other
  languages (it applies to the CLI too, since the server has one active model).

## The same thing in the browser

The dashboard's **Transcribe** tab takes uploaded audio files (several at once) or pasted podcast links
(one per line; resolved by the server the same way, public addresses only). Each becomes a row in a
queue with its own progress bar covering the whole job (downloading a link, decoding, transcribing,
or its place in the queue) and a Cancel button; a finished row shows the transcript with exports and is
saved in History. The server API behind both: `POST /api/transcriptions` (body = the audio file;
`?filename=&title=&diarize=&speakers=`), `POST /api/transcriptions/link` (`{"url", "diarize", "speakers", "episode"}`; `speakers` is 1 to 20 and implies `diarize`),
`GET /api/transcriptions/{id}` (poll: `status` of queued, fetching, decoding, transcribing, done, error or
cancelled; `progress` 0 to 1 over the whole job, `stage_progress` of the transcribing, `download` {done, total}
bytes for a link, `queue_position`, `step`, `error`, `result`, `session_id`),
`DELETE /api/transcriptions/{id}` (cancel). No token, like the rest of the API.
