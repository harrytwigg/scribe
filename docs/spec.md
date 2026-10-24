# scribe: self-hosted call transcription

Status: phase 0 built and tested; phase 1 deployed on harry-box with the live receive test outstanding (section 7). 2026-10-12.

## 1. Goal

A transcription service Harry controls end to end. A Discord bot joins a voice channel and
transcribes each speaker; the speech recognition (ASR) runs on harry-box's GPU; a web GUI picks
the ASR model from a catalogue of open-source models, Handy-style, and shows live and past
transcripts. Off-the-shelf bots (Scripty, which Harry uses today) fix the model; this one does not.

Google Meet and other browser calls come later, through a Chrome extension.

## 2. Research summary

Full notes with sources: `docs/research/discord.md` and `docs/research/asr.md`.

### Discord voice receive is feasible, with one library choice that matters

- Discord made DAVE (end-to-end encrypted voice) mandatory on 2026-03-01. A bot whose voice
  library lacks DAVE cannot join a voice channel at all, and receiving needs DAVE decryption too.
- Per-user receive under DAVE is implemented in **discord.js** (`@discordjs/voice` 0.19.2 with
  `@snazzah/davey`), **songbird** 0.6.x (Rust) and **JDA** 6.7 + JDAVE (Java 25). Implemented is
  not proven: the discord.js issues found are closed with fixes, but one report (2026-06) got zero
  audio from this exact stack and only Craig worked for its author. **Reliability is unproven until
  our own live test** (phase 1, first item).
- It does not work in the Python libraries: `discord-ext-voice-recv` has only unmerged DAVE PRs,
  py-cord's receive decrypt is unfinished, nextcord has no receive.
- `GuildVoiceStates` is not a privileged intent. The bot needs Connect + View Channel and must join
  undeafened. Discord's developer policy has no voice-recording rule, but requires clear opt-in;
  consent is ours to handle (section 8).
- discord.js #11653 (open, 2026-10-08) is on the **main path**, not an edge: the bot always joins
  a channel where someone is already present (whoever ran `/scribe join`), so it receives an MLS
  welcome at transition 0, which `lastTransitionId` treats as falsy, and decrypt errors are never
  recovered. The reporter's suggested `!== undefined` change is unmerged and partly a guess, and
  the failure counter is per session, not per speaker. We carry the fix as a pnpm patch
  (`bot/patches/`), which pnpm applies only to that exact `@discordjs/voice` version, and it is
  only trusted once the live receive test passes.

### ASR: both model families run in one Python process on the Turing GPU

Measured on harry-box (Quadro RTX 5000, 16 GB, compute 7.5), 11 s clip, warm:

| Model | Runtime | Time | Result |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v2 | onnx-asr 0.12 on onnxruntime-gpu 1.24.4 (CUDA 12) | 0.056 s | exact |
| Whisper large-v3-turbo | faster-whisper 1.2.1 (CTranslate2 4.8, fp16) | 0.54 s | exact |

Two traps found and solved: onnxruntime-gpu 1.27+ on PyPI is CUDA 13 and rejects models in the
Hugging Face cache's symlinked layout (so: pin 1.24.4, and download models into a plain directory
of our own); and both runtimes need the pip CUDA 12 libraries preloaded (`onnxruntime.preload_dlls()`).

Open ASR leaderboard (mean WER / meeting WER on AMI / speed RTFx), for the catalogue:

| Model | WER | AMI | RTFx | Licence | Runs via |
| --- | --- | --- | --- | --- | --- |
| Parakeet TDT 0.6B v2 (English) | 6.05 | 11.16 | 3386 | CC-BY-4.0 | onnx-asr |
| Parakeet TDT 0.6B v3 (25 languages) | 6.34 | 11.31 | 3333 | CC-BY-4.0 | onnx-asr |
| Canary 1B v2 (multilingual) | n/a | n/a | n/a | CC-BY-4.0 | onnx-asr |
| distil-large-v3.5 | 7.21 | - | 202 | MIT | faster-whisper |
| Whisper large-v3 (figures unverified) | 7.44 | - | 146 | MIT | faster-whisper |
| Whisper large-v3-turbo (figures unverified) | 7.83 | - | 200 | MIT | faster-whisper |
| Whisper small / base (English) | higher | - | fast | MIT | faster-whisper |

Better-scoring models exist (Granite-speech 4.1 2B, Canary-Qwen 2.5B, Kyutai STT), but they ship
BF16 weights Turing cannot run natively, or need NeMo/PyTorch. They are out of the MVP catalogue
and can be added later as a third engine. A model enters the shipped catalogue only once the GPU
test has transcribed the reference clip with it.

### Build vs reuse

| Piece | Decision | Why |
| --- | --- | --- |
| Discord bot | **Build** (~300 lines TS on discord.js) | Hosted Scripty works under DAVE (it transcribed a call on 2026-09-21), but its model is fixed and its self-hostable source was archived on 2025-06-17. Craig (ISC, active) records under DAVE but has no ASR; it is the fallback if discord.js receive fails (section 8). No maintained whisper-based Discord bot exists. |
| Voice transport | **Reuse** discord.js + davey | The only mature DAVE receive in a mainstream library. |
| ASR runtimes | **Reuse** faster-whisper, onnx-asr | Proven above. |
| ASR server | **Build** a thin FastAPI server; keep its API OpenAI-compatible | speaches (MIT) would cover model loading, but the sessions, transcript store and GUI are needed regardless, the swap semantics (section 3.3) stay under our control, and the onnxruntime/model-directory traps are already solved on our exact stack (unverified whether speaches hits them). speaches has also stalled at a release candidate (2025-12). Ours is a few hundred lines. Keeping `POST /v1/audio/transcriptions` OpenAI-shaped means speaches, a cloud API or anything else can be swapped in behind the bot later. |
| GUI | **Build**, borrowing Handy's model-picker UX | Handy is a desktop dictation app (Tauri); its catalogue schema (size, speed, accuracy, download state) is the idea worth taking, not the code. |
| Meet capture | **Build** a Chrome extension (phase 3) | A bot joining Meet sits in the lobby, is auto-denied by Google's 2026 risk queue, or needs a bot Google account. Vexa (Apache-2.0) and Attendee (Elastic) are heavyweight multi-container stacks for unattended bots. For one person's calls, `chrome.tabCapture` + mic in an offscreen document, streamed to our server, is simpler and needs no account. |

## 3. Architecture

```
 Discord voice (DAVE, UDP)                       Harry's browser (tailnet)
          |                                               |
          v                                               v
 +------------------ harry-box ---------------------------------------+
 |  bot (Node 22, discord.js)  --HTTP-->  server (Python 3.12, FastAPI)|
 |   per-user Opus -> PCM                 /v1/audio/transcriptions     |
 |   utterance segmentation               /api/models (catalogue)      |
 |   16 kHz mono WAV per utterance        /api/sessions (+ SSE live)   |
 |                                        GUI at /                     |
 |                                        engines: faster-whisper,     |
 |                                                 onnx-asr (CUDA)     |
 |                                        SQLite + models/ on disk     |
 +---------------------------------------------------------------------+
          ^ (phase 2, optional)
          | tailnet
 tools-production-1 (Caddy edge, joins the tailnet) <-- https://scribe.<domain>  (Chrome extension off-tailnet)
```

### 3.1 Ingress: the Discord path needs none

A Discord bot makes only outbound connections (the gateway websocket and the voice UDP socket), so
**the MVP needs no public endpoint, tunnel or webhook**. The GUI is reached over the tailnet
(`http://harry-box:8787`, or HTTPS via `tailscale serve`), behind the server's own auth (3.7).

Public ingress matters only for clients that are not on the tailnet, mainly the Chrome extension
on a machine where Tailscale cannot be installed. For those, phase 2 uses the existing **tools**
app in `universal-react-monorepo` (`tools-production-1`, its Caddy edge and `host.addSite`), which
already has a public address, so no Cloudflare tunnel is needed:

- The tools host joins the tailnet with a **tagged, pre-authorised auth key** (`tag:tools-edge`),
  node-key expiry disabled for that tagged node so the edge does not silently drop off, and the
  auth key itself short-lived (single use).
- The tailnet policy lets `tag:tools-edge` reach only `harry-box:8787`. If the tailnet still runs
  the default allow-all policy, that restriction means writing explicit grants for every other
  device: a **tailnet-wide policy change**, not a one-line addition.
- A new tools workload adds the edge site `scribe.<domain>`, reverse-proxied to
  `harry-box.<tailnet>.ts.net:8787`. The server's own token auth (3.7) still applies behind it.

This is a public-exposure decision and needs Harry (H3).

### 3.2 Audio pipeline (Discord)

1. `/scribe join` (slash command): the bot joins the caller's voice channel undeafened and muted,
   sets its nickname to "🔴 transcribing", posts a notice in the channel's text chat, and opens a
   server session (`POST /api/sessions`).
2. On each `speaking.start` from a user who has not opted out, the bot subscribes to that user's
   Opus stream, ending after 800 ms of silence, and decodes to 48 kHz stereo PCM. Each packet is
   decoded on its own and one that will not decode is skipped, because a stream decoder that
   errors stops for good, which once lost a whole speaker for a call. Frames still DAVE-encrypted
   (passed through when the bot cannot decrypt them yet, just after joining or mid key change;
   they end in the 0xFAFA marker) are dropped before decoding: half of them decode as noise, and
   decoding many corrupts the shared Opus decoder's memory. `/scribe status` lists each
   person in the call and how many lines of theirs were transcribed, so a missing speaker shows
   during the call rather than after it.
3. An utterance is cut at the silence or at **25 s**, whichever comes first. Utterances shorter
   than **400 ms** are dropped (keyboard, breath). Each is downmixed to mono, wrapped as a 48 kHz
   WAV (the server resamples to 16 kHz), and posted to `POST /api/sessions/{id}/utterances` with the speaker's id,
   display name and start time.
4. The server transcribes it with the active model (3.3), stores the line, and pushes it to live
   listeners (SSE).
5. If the server answers 503 (queue full) or times out, the bot retries twice with backoff
   (1 s, 3 s), then records a `[unavailable: N s of audio from <speaker>]` marker line instead of
   blocking the call (`POST /api/sessions/{id}/markers`).
6. `/scribe stop`, usable by **anyone in the channel**, ends the session and the bot leaves. The
   session also ends when the bot is alone in the channel.
7. Consent: a user who joins after the start is re-told by a channel message on their
   `voiceStateUpdate` join. `/scribe optout` (and `/scribe optin`) is per user and persistent: an
   opted-out user is never subscribed. Echoing transcript lines into a Discord thread exists but is
   **off by default** (H2).

Receive and transport are separated by a `VoiceSource` interface. A replay source stands in for
Discord in tests: it Opus-encodes a WAV into 20 ms packets and emits them as a per-user stream.
That slots in **after `receiver.subscribe`**: it proves the pipeline from Opus packets onwards
(decode, segmentation, HTTP, ASR, storage, attribution) but **not Discord receive itself**
(transport decrypt, DAVE/MLS, SSRC-to-user mapping, speaking events, #11653). Only the live test
proves those.

Per-user streams give speaker attribution for free; no diarization is needed on the Discord path.

### 3.3 Model server

- **Engines**: `faster-whisper` (CTranslate2, fp16 on CUDA) and `onnx-asr` (onnxruntime CUDA
  provider). An engine is an adapter with `load(dir)`, `transcribe(audio16k) -> text, segments`,
  `unload()`.
- **Catalogue** (`catalogue.py`): id, display name, engine, source repo, size, languages, licence,
  WER and speed hints, recommended flag. Same shape Handy shows.
- **State**: per model, `not_downloaded | downloading(progress) | downloaded | error`, and the
  server holds one `active` model id with a `loading` flag. Models download into `models/<id>/`
  (a plain directory, not the HF cache) via a `.partial` directory renamed on success; a failed or
  interrupted download removes its partial directory, and leftovers are cleaned at startup.
- **GPU worker**: all inference runs on **one worker thread** fed by a bounded queue (default depth
  32), off the event loop. Utterances from several speakers queue rather than contend for the GPU.
  A full queue answers 503 (the bot's retry handles it).
- **Swap**: activating a model loads it on the side, then swaps the active pointer under a lock.
  Each job holds a reference to the model it started with, and the old model is unloaded only when
  its in-flight count drains to zero. Nothing is dropped mid-swap. The active choice persists
  across restarts.
- **Idempotency**: the bot sends a `client_key` per utterance (speaker id and start time); a key
  already stored returns the stored line, and two attempts racing on the GPU still store one row
  (a unique index), so retrying after a timeout cannot duplicate a line.
- **Rules**: activate requires the model to be downloaded (409 otherwise, including mid-download);
  DELETE of the active model is refused (409); DELETE mid-download cancels the download and removes
  the partial directory. `model=<id>` on `/v1/audio/transcriptions` must be the active model or be
  omitted; any other id is 409 with the active id in the message (one resident model; no temporary
  loads).
- **Noise and hallucination**: every input runs through VAD first (Silero, from onnx-asr). Audio
  with no speech returns empty text without calling the model, which also stops Whisper's
  "Thank you." on noise. Whisper additionally runs with `vad_filter=True`.
- **Length**: every input, not only uploads, is split at speech pauses into chunks of at most 25 s
  before ASR, because Parakeet in onnx-asr handles ~20-30 s per call.
- **API**:
  - `GET /api/models`, `POST /api/models/{id}/download`, `DELETE /api/models/{id}`,
    `POST /api/models/{id}/activate`, `GET /api/status`.
  - `POST /v1/audio/transcriptions` (OpenAI-compatible multipart; `model` optional).
  - `POST /api/sessions`, `POST /api/sessions/{id}/utterances`, `POST /api/sessions/{id}/end`,
    `GET /api/sessions[/{id}]`, `DELETE /api/sessions/{id}`, `GET /api/sessions/{id}/events`
    (SSE), `GET /api/sessions/{id}/export?format=md|srt|json`.
- **Audio decode**: WAV/FLAC/OGG via soundfile, resampled to 16 kHz. (PyAV in faster-whisper 1.2.1
  is broken against PyAV 16, so we never pass it file paths.)

### 3.4 GUI

One static page served by the server, no build step:

- **Models**: the catalogue as cards: name, size, languages, accuracy/speed hints, licence; buttons
  Download (with progress), Use (activate), Delete. The active model is marked.
- **Live**: open sessions, transcript lines streaming in by speaker.
- **History**: past sessions, export Markdown / SRT / JSON, delete.
- Live and History share one viewer that follows an open call without a refresh: the SSE stream
  delivers lines, a poll every 4 s catches anything it missed (lines are keyed by id), and new or
  ended calls appear in the lists by the same poll. The tab and the open call are in the URL
  (`#live/<id>`, `#history/<id>`), so a refresh keeps them, and a viewer never jumps to another
  call on its own. Clicking a call's title renames it (`PATCH /api/sessions/{id}`); clicking a
  speaker renames them (3.8).
- **Try it**: upload an audio file or record from the mic, see the transcript and timing, to
  compare models on the same clip.

### 3.5 Storage and output

SQLite at `data/scribe.db`: `sessions(id, source, guild, channel, started_at, ended_at, title)`,
`utterances(id, session_id, speaker_id, speaker_name, started_at, duration_ms, text, model_id,
asr_ms, client_key)`, `speakers(session_id, speaker_id, name, voice, weight)` (names given in
the GUI, and diarized speakers' average voices, 3.8), `optouts(source, user_id)`. Audio is **not** retained by default (`SCRIBE_KEEP_AUDIO`
stores WAVs under `data/audio/` for debugging). Transcripts are kept until deleted (H4).

### 3.6 Running

Two systemd `--user` units on harry-box, `scribe-server` and `scribe-bot`. The server binds
`127.0.0.1` and the tailnet address only. The bot's server URL is configurable, so the bot can move
host (H5) without code changes.

Secrets (bot token, server API token) are read at start from the instance's
`secrets/api-keys.json` on the mounted instance home, never copied into the repo or a unit file.

**harry-box powers itself off when idle** (the `harrybox-auto-shutdown` cron, which probes the box
over one ssh call in `bin/harrybox-idle-check.sh`). Making it scribe-aware, a `curl` of
`localhost:8787/api/status` inside that probe plus a test case in its harness, is a **company
infrastructure change**, reviewed on its own, not part of this repo. Whether the bot should be
reachable while the box is off is H5.

### 3.7 Auth (from phase 0)

- Every request must carry the server's API token, **including from localhost**: harry-box also
  runs agent sessions, so loopback is not a trust boundary. `Authorization: Bearer` for the bot and
  API clients. (`SCRIBE_TRUST_LOCALHOST=1` exempts requests from the machine itself, loopback or the
  server's own listen addresses, unless they came through a proxy: the dashboard then opens without
  signing in on that machine, and any process there can call the API without the token. Off by
  default; harry-box turns it on, by Harry's decision, 2026-10-23, so its own browser needs no token.
  Other tailnet machines and the Chrome extension still need it.)
- The GUI signs in with the same token once and gets a session cookie (`HttpOnly`,
  `SameSite=Strict`). State-changing requests from the GUI must also carry a matching `Origin`
  header. This blocks CSRF from any other page open in the browser (a multipart POST needs no
  preflight) and keeps other tailnet nodes, including the gateway that runs agent sessions, out of
  transcripts and model controls.
- The same scheme carries over unchanged behind the phase-2 edge.

### 3.8 Diarization: telling speakers apart in a mixed channel

Discord delivers each user separately. A browser call's tab does not: everyone on the far side
arrives as one stream. The server splits such a channel into speakers when the client marks the
utterance `diarize` (the extension does, for the tab channel only).

- **Models**, ONNX on the CPU through **sherpa-onnx** 1.13.8 (Apache-2.0), downloaded on first
  start into `models/_diarization/` (~45 MB): pyannote **segmentation 3.0** (MIT) finds speaker
  turns; NVIDIA **TitaNet small** (CC-BY-4.0) gives each turn a voice embedding. Live calls and
  uploads each have their own worker thread and models, so neither the GPU queue nor a call waits
  on a long upload. sherpa-onnx carries its own CPU onnxruntime (1.28.2, soname `libonnxruntime.so`,
  versioned symbols), which loads beside onnxruntime-gpu 1.24.4 (`libonnxruntime.so.1`) without
  clashing; all 8 models pass with both loaded. (The separate CPU `onnxruntime` wheel that
  faster-whisper pulls in shares onnxruntime-gpu's package directory, so it is dropped by a uv
  override, and the install script reinstalls onnxruntime-gpu.)
- **Live.** Each utterance is split into turns (clustering set to over-split). Each turn of 1 s
  or more is matched on its own voice against the session's **speaker book**, the running average
  voice of everyone heard so far; shorter turns are pooled by their clusterer label. A match is
  cosine similarity >= 0.35 (`SCRIBE_DIARIZE_MATCH`); only a turn of 2 s or more can found a new
  speaker, and a shorter unmatched one goes to the nearest. Matching per turn against the book
  matters: one utterance's clustering lumped a short turn in with the next speaker, which the
  book, built over the whole call, gets right. Where turns overlap (crosstalk), whoever started
  later has the overlap.
- **Words, not clips.** The utterance is transcribed once, exactly as without diarization, with
  word timings (Parakeet's token times, Whisper's word timestamps), and each word goes to the turn
  it falls in (placed near its start: onnx-asr word ends run on to the next word, into any pause).
  Transcribing each turn on its own lost words (5.1% WER against 2.0% undiarized: a
  clip cut at a turn edge loses context, and Parakeet returned nothing for one 3 s turn). Canary
  gives no timings, so it falls back to a clip per turn, widened to the middle of the gaps around
  it. Measured: diarized WER equals undiarized on Parakeet (2.0%), Whisper turbo (3.1%) and
  Canary (4.1%).
- **Lines.** Each turn is stored as its own line, speaker `tab#1`, `tab#2`, ... named "Speaker 1",
  ... The book is saved with the session (numbers are never reused), so labels survive a restart.
  An utterance's turns are kept until its lines are stored, so a retry after a busy queue or a
  client timeout reuses them, even one arriving while the first attempt is still diarizing (it
  waits for it): the book never learns the same audio twice and the retry's lines match. If
  diarization fails or finds no turns, the utterance (or upload) is transcribed as one speaker:
  speech is never lost to it.
- **Renaming.** Clicking a name in the GUI renames that speaker for the session, past lines and
  later ones (`POST /api/sessions/{id}/speakers/{speaker_id}`). This works for any speaker,
  Discord users included.
- **Whole recordings.** "Try it" (and `/v1/audio/transcriptions` with `diarize=true`) diarizes an
  upload in one pass and returns `segments` with speakers.
- **Scraps and a known count.** On a long recording the clusterer over-splits: one measured 44-minute
  two-person podcast came out as 11 speakers, 93% of the speech in two of them and the rest scraps of
  1 to 75 seconds (a voice over music, the same person on a different line). After clustering,
  `absorb_scraps` measures each speaker's voice (TitaNet, on up to 45 s of their longest turns) and
  folds a speaker with under 15 s of speech into the one they sound most like, and one with under 5%
  of the speech into them if they also sound like them (the `MATCH` similarity), so a real third voice
  that spoke for a minute stays; the same recording came out as 2. If the caller knows how many people
  there are, `speakers` (API, 1 to 20; GUI "How many?"; CLI `--num-speakers`) merges the closest
  voices (`merge_down`) until that many remain, after the scraps; if the first pass found fewer, it
  clusters again more finely (threshold 0.5) first. sherpa's own fixed-count clustering is not used: on
  a real podcast it gave one speaker of 680 s and one of seconds. Live calls are not affected: their
  book already
  merges speakers. Thresholds: `SCRIBE_DIARIZE_SCRAP_S`, `SCRIBE_DIARIZE_SCRAP_SHARE`.
- **Its own processes.** sherpa-onnx holds Python's global lock for the whole of a call: measured,
  4 minutes of audio froze every thread of the server for 19 s, and an hour-long recording froze it
  (the dashboard, History, the Discord bot's posts) for minutes. So the models run in two child
  processes (`diarize.py`, `Diarizer(isolated=True)`, the default): one for whole recordings, one for
  live calls, so a long upload still never holds up a call. They are started with `spawn`, loaded
  once at start (`/api/status` shows `diarization: loading` until both are up), run at a lower
  priority (nice 10), and take audio over shared memory when it is over a minute long. If one dies
  (killed for memory, say) that call falls back to one speaker like any diarization failure, and the
  next call starts a new process. The tests hold the lock on purpose with a stand-in model
  (`tests/test_diarize_process.py`): the in-process control stalls for the full call, the isolated
  one does not.
- **Measured** (tests/test_diarize_gpu.py and a scratch stress test): a 42 s three-speaker
  LibriSpeech conversation, live in two utterances, put every turn on the right, consistent
  speaker. So did a whole-file pass (5 s of CPU for 42 s of audio). Four male speakers taking 16
  turns, cut into utterances every 12 s or 25 s: 16/16 right. JFK cut into three utterances
  stayed one speaker. The thresholds came from TitaNet similarities on LibriSpeech test-clean:
  different speakers peaked at 0.46, and the same speaker's short turns dipped to 0.36.
- **Limits.** All of that is clean read speech. Meet audio is compressed, and real calls have
  crosstalk; overlapping speech goes to whoever dominates the turn. A newcomer who only says a word
  or two is labelled as the nearest known speaker until they speak for 2 s. Until the models load
  (or if they fail; `/api/status` shows `diarization`), a mixed channel is one speaker, as before.

## 4. Chrome extension: Meet and any browser call, with no bot (phase 3)

Nothing joins the call. The extension (`extension/`, MV3, no build step) captures the call tab
and the user's mic in their own browser and posts utterances to the server like the bot does.

1. A click on **Transcribe this tab** in the popup calls `chrome.tabCapture.getMediaStreamId`.
   Tab capture is only allowed from such a click.
2. The background service worker opens an offscreen document (`USER_MEDIA`). The document opens
   the tab stream and, if allowed, the mic (echo cancellation on), and plays the tab back, because
   capturing a tab mutes it. It holds the call because a service worker cannot keep a MediaStream
   and is stopped when idle.
3. Each stream runs into its own 16 kHz `AudioContext` and an AudioWorklet tap. A per-channel
   segmenter cuts utterances with the bot's rules: 800 ms silence, 25 s cap, 400 ms minimum. Speech
   is energy 12 dB above the channel's noise floor, the quietest 20 ms of the last 10 s. The server
   still runs Silero VAD on every utterance.
4. Utterances go to `POST /api/sessions/{id}/utterances` with the bot's retry, idempotency key and
   marker behaviour, in a `browser` session titled after the tab. The tab is speaker `tab`
   ("Others" by default) and the mic `mic` ("Me"), so "me" vs "them" needs no diarization. The tab
   channel is then diarized on the server (3.8) into "Speaker 1", "Speaker 2", ... (setting
   "Tell the other side's speakers apart", on by default).
5. A call ends on **Stop**, when the tab closes, when a new document loads in the captured tab
   (`webNavigation.onCommitted`; capture itself survives navigation), or after 10 minutes with no
   transcribed line on either channel (leaving a Meet call can keep the tab open). Same-document
   URL changes (`history.pushState`, `#fragment`) do not end it: Meet makes them mid-call. The
   idle stop also ends a call after any 10-minute lull (a break, a silent screen-share); the badge
   clears and the popup says why. Ending waits for the last utterances, then ends the session.
6. While a call runs, the extension sends a heartbeat every 30 s. The server ends any `browser`
   session that has had no line and no heartbeat for 5 minutes (`SCRIBE_BROWSER_IDLE_SECONDS`),
   because a browser that quits or crashes mid-call never ends its own. Otherwise an orphan would
   show as live forever and keep `open-sessions` above zero. A call whose session the server
   ended (after the laptop slept, say) stops itself at the next heartbeat.

**Why HTTP utterances and not the WebSocket stream planned earlier:** the bot's ingestion path,
idempotency and retries are reused unchanged, the server gains no streaming code, and it works
through the phase-2 Caddy edge as it is.

**Auth.** The token is kept in `chrome.storage.local` and sent as a bearer token. Browsers send
`Origin: chrome-extension://<id>` on the extension's POSTs, which the CSRF check (3.7) would refuse,
so the server also admits the extension's own origin (`SCRIBE_EXTENSION_ORIGINS`). That origin is
fixed by the public `key` in the manifest: the unpacked extension has ID
`mcnpmckibkddnnenfondlblnddldonfm` on every machine. A web page cannot send that origin, and
requests from it still need the token or a signed-in cookie, like any other. Login (getting the
cookie) stays same-origin only.

**Reach.** The host permissions cover `harry-box`, `*.ts.net`, `localhost` and `127.0.0.1`. Any
other server URL (the phase-2 public domain) is requested as an optional permission when it is
saved in the settings.

**What Google and the call can see.** Nothing announces it. Tab capture happens inside Chrome,
and no web API tells a page that its tab is being captured or that another page has the mic open.
The extension injects nothing into meet.google.com: no content script, no web-accessible
resources. So Meet has no known way to detect it, other participants get no "recording" notice
(that comes only from Meet's own recording or transcription), and no audio goes to Google: it goes
to harry-box over the tailnet. Three things can show:
- Chrome's own indicators on Harry's screen: the tab's capture icon, and the mic in use.
- Echo heard by the far side if Harry is on speakers (above).
- On a Google Workspace managed browser or profile, the admin can list installed extensions and can
  block developer-mode extensions or tab capture by policy.

Not being detectable is not consent. Telling people remains Harry's call, and the popup says so.

**Limits.** Meet mixes everyone on the far side into the tab's audio, so telling them apart is
diarization (3.8), which numbers them; their names are given by renaming in the GUI. Reading
names from Meet's own speaker UI would need a content script on meet.google.com (not built).
**Use headphones.** Capturing the tab moves the call's playback out of Meet and into the
extension, so Meet's own echo cancellation may no longer recognise it. On speakers, the others
could hear themselves echo, and the mic could repeat the call as "Me" lines. Both are unverified;
the first live call on speakers should ask the far side. The mic can be turned off in settings.

## 5. Model options and defaults

Default active model: **Parakeet TDT 0.6B v2** (best English accuracy and by far the fastest; word
timestamps). Alternates: Parakeet v3 for non-English, Whisper large-v3-turbo as the robust
all-rounder, distil-large-v3.5 for speed with Whisper behaviour, Whisper small as the light option.

## 6. Testing

- **Server** (pytest): catalogue, download/delete/activate rules, swap with in-flight requests
  (old model drains before unload), queue-full 503, auth and Origin checks, API contract, exports.
  A `gpu` marker runs real models end to end: download, activate, transcribe the reference clip,
  swap, transcribe again, with a WER check; and silence returns empty text for every model.
- **Bot** (vitest): segmentation (silence cut, 25 s cap, 400 ms floor), downmix/resample, WAV
  encoding, retry/marker behaviour.
- **Replay end to end**: two replay `VoiceSource`s speaking at once through the real bot pipeline
  into a running server, with a model swap mid-stream; checks attribution and transcripts. This
  proves everything downstream of `receiver.subscribe` (3.2), not Discord receive.
- **GUI**: driven in a headless browser: sign in, switch model, upload a clip, see the transcript.
- **Extension** (`node --test`): segmentation, WAV encoding, client retries and markers. With the
  `extension` marker, pytest loads the unpacked extension in Chromium. It saves settings through
  the options page, checks that a bad tab stream fails cleanly through the background and offscreen
  document, clears a stale status, and runs the capture code on the reference clip, as the tab (an audio element's stream)
  and as the mic (Chromium's fake device), into a real server on the GPU. Both channels must come
  back word-perfect and attributed, in a session that Stop then ends; a start while it drains is
  refused. Server tests cover the reaper and heartbeat. `chrome.tabCapture` itself
  and the mic permission prompt need a person.
- **Live Discord** (phase 1, first item): DAVE smoke test in a real voice channel: join with two or
  more people already talking, someone joins mid-call, bot restart mid-session.

## 7. Phased plan

| Phase | Scope | Needs Harry |
| --- | --- | --- |
| 0 (MVP) | Server, catalogue, model swap, auth, GUI; bot with replay-tested pipeline; all local on harry-box. Proves audio in, transcript out, model choice through the GUI. Does not prove Discord receive. | Done |
| 1 | **Live DAVE test first** (6): the join/handshake half passed (DAVE v1 negotiated, connection Ready); the receive half needs a person speaking. Systemd units deployed on harry-box. Then a soak test; a dedicated Discord application (H1); and the auto-shutdown gate as a separate company-infra change | Receive test: someone talking in a call; the application (H1) |
| 2 | Public ingress via the tools host + tailnet policy | H3 |
| 3 | Chrome extension for Meet / any browser call (4): built and tested in Chromium; the first real Meet call is the live test | Load it unpacked and try a call; H3 if off the tailnet |
| 4 | Diarization for single-stream sources (built: 3.8; a live Meet call is its real test), LLM summaries, always-on bot host, extra engines (NeMo/BF16 models) | Per item |
| 5 | Podcast links and recordings: the `scribe` CLI, background transcription jobs (`/api/transcriptions`) saved to History, and the Transcribe tab (`docs/cli.md`) | Done |

## 8. Risks

- **Discord receive is unproven on our stack.** discord.js under DAVE is implemented but has a
  contrary report and the open main-path #11653. Mitigation: live test first in phase 1, patch
  pinned. Fallbacks, in order: songbird 0.6 (Rust) as the receive layer posting to the same server
  API; or self-hosted Craig (ISC) recording, with its per-user tracks fed to the server after the
  call (loses live transcripts).
- **DAVE regressions.** Discord changes the voice protocol; pin versions and re-run the smoke test
  on every bump.
- **Consent and policy.** The bot announces itself (nickname, channel message, re-notice for late
  joiners), anyone can stop it, users can opt out, transcripts stay off Discord by default, audio is
  not kept. Not legal advice; Harry decides where it is used (H2).
- **GPU stack fragility.** CTranslate2 and onnxruntime both need CUDA 12 / cuDNN 9; pin
  onnxruntime-gpu 1.24.4 and the pip CUDA libraries together. A future ORT move to CUDA 13 must
  move both.
- **harry-box availability.** It is off most of the time; see H5.
- **Extension echo and attribution.** On speakers, the far side may hear an echo and the mic may
  repeat the call as "Me" lines, so use headphones until a live test says otherwise. All remote
  speakers share one label (4).
- **Public exposure (phase 2).** The edge is the only public surface; tailnet policy and the
  server's own auth limit it.
- **Licences.** Parakeet and Canary are CC-BY-4.0 (attribution), Whisper MIT; fine for personal
  use.

## 9. Decisions

Harry asked for the build to go ahead without further questions (2026-10-12). The recommended
option was taken for H2 to H5. **H1 deviates from its recommendation**, because the recommended
option needs Harry's own hands. Any can be revisited.

- **H1 Bot identity: the existing bot application, for now** ("Longbow Dev", user "Harry test"),
  whose token is already in the instance's secrets file (`DISCORD_BOT_TOKEN`). A dedicated
  application (the recommendation) has to be created in Harry's Discord Developer Portal, which an
  agent cannot do. The existing application also holds the Message Content Intent, which scribe
  does not need. The bot reads its token under `SCRIBE_DISCORD_TOKEN_KEY`, so moving to a dedicated
  application is a config change (phase 1).
- **H2 Where: the Longbow server**, slash commands registered there only (`SCRIBE_GUILD_ID`). The
  notice, nickname and `/scribe optout` are the consent mechanism; transcripts are not echoed into
  Discord (`SCRIBE_ECHO` off).
- **H3 Ingress: tailnet-only.** Public exposure is not needed for the Discord path (3.1). The tools
  host route stays the plan for phase 3, when the Chrome extension needs off-tailnet access.
- **H4 Retention: no audio kept; transcripts kept until deleted** in the GUI.
- **H5 Availability: on demand.** harry-box must be on for a call; both services start at boot. An
  always-on bot host is phase 4.
