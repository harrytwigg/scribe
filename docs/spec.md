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
   Opus stream, ending after 800 ms of silence, and decodes to 48 kHz stereo PCM.
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
- **Try it**: upload an audio file or record from the mic, see the transcript and timing, to
  compare models on the same clip.

### 3.5 Storage and output

SQLite at `data/scribe.db`: `sessions(id, source, guild, channel, started_at, ended_at, title)`,
`utterances(id, session_id, speaker_id, speaker_name, started_at, duration_ms, text, model_id,
asr_ms)`, `optouts(source, user_id)`. Audio is **not** retained by default (`SCRIBE_KEEP_AUDIO`
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
  API clients. (`SCRIBE_TRUST_LOCALHOST=1` re-enables a loopback exemption for a single-user
  machine; off by default.)
- The GUI signs in with the same token once and gets a session cookie (`HttpOnly`,
  `SameSite=Strict`). State-changing requests from the GUI must also carry a matching `Origin`
  header. This blocks CSRF from any other page open in the browser (a multipart POST needs no
  preflight) and keeps other tailnet nodes, including the gateway that runs agent sessions, out of
  transcripts and model controls.
- The same scheme carries over unchanged behind the phase-2 edge.

## 4. Chrome extension (phase 3)

MV3 extension: the popup's click triggers `chrome.tabCapture.getMediaStreamId`; an offscreen
document (`USER_MEDIA`) opens the tab stream and the mic, plays the tab audio back to the user, and
streams 16 kHz PCM over a WebSocket (`/api/stream`, token-authenticated) with tab and mic on
separate channels. That labels "me" vs "them" without diarization. The server cuts utterances with
VAD per channel and feeds them through the same session pipeline. Multiple remote speakers on the
tab channel would need diarization (pyannote or Sortformer), which is a later addition.

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
- **Live Discord** (phase 1, first item): DAVE smoke test in a real voice channel: join with two or
  more people already talking, someone joins mid-call, bot restart mid-session.

## 7. Phased plan

| Phase | Scope | Needs Harry |
| --- | --- | --- |
| 0 (MVP) | Server, catalogue, model swap, auth, GUI; bot with replay-tested pipeline; all local on harry-box. Proves audio in, transcript out, model choice through the GUI. Does not prove Discord receive. | Done |
| 1 | **Live DAVE test first** (6): the join/handshake half passed (DAVE v1 negotiated, connection Ready); the receive half needs a person speaking. Systemd units deployed on harry-box. Then a soak test, and the auto-shutdown gate as a separate company-infra change | Receive test: someone talking in a call |
| 2 | Public ingress via the tools host + tailnet policy | H3 |
| 3 | Chrome extension for Meet / any browser call | No on the tailnet; H3 if off it |
| 4 | Diarization for single-stream sources, LLM summaries, always-on bot host, extra engines (NeMo/BF16 models) | Per item |

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
- **Public exposure (phase 2).** The edge is the only public surface; tailnet policy and the
  server's own auth limit it.
- **Licences.** Parakeet and Canary are CC-BY-4.0 (attribution), Whisper MIT; fine for personal
  use.

## 9. Decisions

Harry asked for the build to go ahead without further questions (2026-10-12), so the recommended
option was taken for each. Any can be revisited.

- **H1 Bot identity: reuse the existing bot application** ("Longbow Dev", user "Harry test"), whose
  token is already in the instance's secrets file (`DISCORD_BOT_TOKEN`). The bot reads it under
  `SCRIBE_DISCORD_TOKEN_KEY`, so moving to a dedicated application later is a config change.
- **H2 Where: the Longbow server**, slash commands registered there only (`SCRIBE_GUILD_ID`). The
  notice, nickname and `/scribe optout` are the consent mechanism; transcripts are not echoed into
  Discord (`SCRIBE_ECHO` off).
- **H3 Ingress: tailnet-only.** Public exposure is not needed for the Discord path (3.1). The tools
  host route stays the plan for phase 3, when the Chrome extension needs off-tailnet access.
- **H4 Retention: no audio kept; transcripts kept until deleted** in the GUI.
- **H5 Availability: on demand.** harry-box must be on for a call; both services start at boot. An
  always-on bot host is phase 4.
