# scribe: self-hosted call transcription

Status: draft for review, 2026-10-12.

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
- Per-user receive under DAVE works today in **discord.js** (`@discordjs/voice` 0.19.2 with
  `@snazzah/davey`), **songbird** 0.6.x (Rust) and **JDA** 6.7 + JDAVE (Java 25).
- It does not work in the Python libraries: `discord-ext-voice-recv` has only unmerged DAVE PRs,
  py-cord's receive decrypt is unfinished, nextcord has no receive.
- `GuildVoiceStates` is not a privileged intent. The bot needs Connect + View Channel and must join
  undeafened. Discord's developer policy has no voice-recording rule, but requires clear opt-in;
  consent is ours to handle (section 8).
- Known bug: discord.js #11653 (open, 2026-10-08), decrypt errors when the bot joins after users
  are already talking. One-line fix; we carry it as a patch until it is released.

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
| Whisper large-v3 | 7.44 | - | 146 | MIT | faster-whisper |
| Whisper large-v3-turbo | 7.83 | - | 200 | MIT | faster-whisper |
| Whisper small / base (English) | higher | - | fast | MIT | faster-whisper |

Better-scoring models exist (Granite-speech 4.1 2B, Canary-Qwen 2.5B, Kyutai STT), but they ship
BF16 weights Turing cannot run natively, or need NeMo/PyTorch. They are out of the MVP catalogue
and can be added later as a third engine.

### Build vs reuse

| Piece | Decision | Why |
| --- | --- | --- |
| Discord bot | **Build** (~300 lines TS on discord.js) | Scripty is archived (2025-06) and pre-DAVE, so it cannot join voice. Craig (ISC, active) records but has no ASR; adapting it costs more than a thin bot. No maintained whisper-based Discord bot exists. |
| Voice transport | **Reuse** discord.js + davey | The only mature DAVE receive in a mainstream library. |
| ASR runtimes | **Reuse** faster-whisper, onnx-asr | Proven above. |
| ASR server | **Build** a thin FastAPI server; keep its API OpenAI-compatible | speaches (MIT) does most of this, but has no notion of "the active model" the GUI controls, pins Python 3.12 exactly, has stalled at a release candidate (2025-12), and would still need our sessions, transcript store and GUI. Ours is ~400 lines. Keeping `POST /v1/audio/transcriptions` OpenAI-shaped means speaches, a cloud API or anything else can be swapped in behind the bot later. |
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
(`http://harry-box:8787`, or HTTPS via `tailscale serve`).

Public ingress matters only for clients that are not on the tailnet, mainly the Chrome extension
on a machine where Tailscale cannot be installed. For those, phase 2 uses the existing **tools**
app in `universal-react-monorepo` (`tools-production-1`, its Caddy edge and `host.addSite`), which
already has a public address, so no Cloudflare tunnel is needed:

- The tools host joins the tailnet with a **tagged, pre-authorised auth key** (`tag:tools-edge`).
- A tailnet ACL lets `tag:tools-edge` reach only `harry-box:8787`, nothing else.
- A new tools workload adds the edge site `scribe.<domain>`, reverse-proxied to
  `harry-box.<tailnet>.ts.net:8787`, with authentication at the edge (a bearer token for the
  extension's API calls; a login for the GUI).
- The server additionally requires its own API token for any request that does not come from
  localhost, so a misconfigured edge does not expose it.

This is a public-exposure decision and needs Harry (section 9).

### 3.2 Audio pipeline (Discord)

1. `/scribe join` (slash command) in a guild: the bot joins the caller's voice channel undeafened
   and muted, posts a notice in the channel that transcription has started, and opens a server
   session (`POST /api/sessions`).
2. On each `speaking.start`, the bot subscribes to that user's Opus stream, ending after 800 ms of
   silence, and decodes to 48 kHz stereo PCM.
3. An utterance is cut at the silence or at 30 s, whichever comes first (long monologues are not
   held back). Each is downmixed and resampled to 16 kHz mono, wrapped as WAV, and posted to
   `POST /api/sessions/{id}/utterances` with the speaker's id, display name and start time.
4. The server transcribes it with the active model, stores the line, and pushes it to live
   listeners (SSE). Optionally the bot echoes lines into a text-channel thread.
5. `/scribe leave`, or the channel emptying, ends the session.

Receive and transport are separated (`VoiceSource` interface) so a replay source can stand in for
Discord in tests: it Opus-encodes a WAV into 20 ms packets, exactly what Discord delivers.

Per-user streams give speaker attribution for free; no diarization is needed on the Discord path.

### 3.3 Model server

- **Engines**: `faster-whisper` (CTranslate2, fp16 on CUDA) and `onnx-asr` (onnxruntime CUDA
  provider). An engine is an adapter with `load(dir)`, `transcribe(audio16k) -> text, segments`,
  `unload()`.
- **Catalogue** (`catalogue.py`): id, display name, engine, source repo, size, languages, licence,
  WER and speed hints, recommended flag. Same shape Handy shows.
- **State**: per model, `not_downloaded | downloading(progress) | downloaded | loading | active |
  error`. Models download into `models/<id>/` (a plain directory, not the HF cache).
- **One active model** is resident on the GPU. Switching loads the new model, then swaps it in
  under a lock, then frees the old one, so in-flight requests finish on the old model and nothing
  is dropped. The active choice persists across restarts. VRAM on 16 GB is ample for any single
  catalogue model.
- **API**:
  - `GET /api/models`, `POST /api/models/{id}/download`, `DELETE /api/models/{id}`,
    `POST /api/models/{id}/activate`, `GET /api/status`.
  - `POST /v1/audio/transcriptions` (OpenAI-compatible multipart; `model` optional, defaults to
    the active one).
  - `POST /api/sessions`, `POST /api/sessions/{id}/utterances`, `POST /api/sessions/{id}/end`,
    `GET /api/sessions[/{id}]`, `GET /api/sessions/{id}/events` (SSE),
    `GET /api/sessions/{id}/export?format=md|srt|json`.
- **Audio decode**: WAV/FLAC/OGG via soundfile, resampled to 16 kHz. (PyAV in faster-whisper 1.2.1
  is broken against PyAV 16, so we never pass it file paths.)
- **Long files** (GUI uploads): split on silence into ≤25 s chunks before ASR, because onnx-asr
  Parakeet takes ~20-30 s per call.

### 3.4 GUI

One static page served by the server, no build step:

- **Models**: the catalogue as cards: name, size, languages, accuracy/speed bars, licence; buttons
  Download (with progress), Use (activate), Delete. The active model is marked.
- **Live**: open sessions, transcript lines streaming in by speaker.
- **History**: past sessions, export Markdown / SRT / JSON.
- **Try it**: upload an audio file or record from the mic, see the transcript and timing, to
  compare models on the same clip.

### 3.5 Storage and output

SQLite at `data/scribe.db`: `sessions(id, source, guild, channel, started_at, ended_at, title)`,
`utterances(id, session_id, speaker_id, speaker_name, started_at, duration_ms, text, model_id,
asr_ms)`. Audio is **not** retained by default (a `SCRIBE_KEEP_AUDIO` switch stores WAVs under
`data/audio/` for debugging). Transcripts export as Markdown, SRT or JSON.

### 3.6 Running

Two systemd `--user` units on harry-box, `scribe-server` and `scribe-bot`. The server binds
`127.0.0.1` and the tailnet address only.

**harry-box powers itself off when idle** (the `harrybox-auto-shutdown` cron). Phase 1 makes that
cron skip shutdown while `GET /api/status` reports an open session. The bot is offline while the
box is off, which is acceptable for an on-demand tool. Moving the bot to an always-on host and
waking harry-box on `/scribe join` is a phase 4 option.

## 4. Chrome extension (phase 3)

MV3 extension: the popup's click triggers `chrome.tabCapture.getMediaStreamId`; an offscreen
document (`USER_MEDIA`) opens the tab stream and the mic, plays the tab audio back to the user, and
streams 16 kHz PCM over a WebSocket (`/api/stream`) with tab and mic on separate channels. That
labels "me" vs "them" without diarization. The server cuts utterances with VAD (Silero via
onnx-asr) per channel and feeds them through the same session pipeline. Multiple remote speakers
on the tab channel would need diarization (pyannote or Sortformer), which is a later addition.

## 5. Model options and defaults

Default active model: **Parakeet TDT 0.6B v2** (best English accuracy and by far the fastest; word
timestamps). Recommended alternates: Parakeet v3 for non-English, Whisper large-v3-turbo as the
robust all-rounder, distil-large-v3.5 for speed with Whisper behaviour, Whisper small for CPU
fallback.

## 6. Testing

- **Server** (pytest): catalogue, state machine, model swap under concurrent requests, API
  contract, exports. A `gpu` marker runs real models end to end (download, activate, transcribe
  a known clip, swap, transcribe again) with WER checks against the reference text.
- **Bot** (vitest): segmentation, downmix/resample, WAV encoding; and an end-to-end replay test
  that feeds a WAV through the real bot pipeline (Opus encode, `VoiceSource`, decode, segment,
  HTTP) into a running server and checks the transcript.
- **GUI**: driven in a headless browser: switch model, upload a clip, see the transcript.
- **Live Discord**: a soak test in a real voice channel (several speakers, mid-call joins, bot
  restart). Needs a bot token and a channel (section 9).

## 7. Phased plan

| Phase | Scope | Needs Harry |
| --- | --- | --- |
| 0 (MVP) | Server, catalogue, model swap, GUI, bot with replay-tested pipeline, all local on harry-box | No |
| 1 | Live Discord: bot app, invite to a server, soak test; systemd units; auto-shutdown gate | Bot token decision |
| 2 | Public ingress via the tools host + tailnet ACL | Yes: domain, exposure, Tailscale key |
| 3 | Chrome extension for Meet / any browser call | No (tailnet), Yes if off-tailnet |
| 4 | Diarization for single-stream sources, LLM summaries, always-on bot host, extra engines (NeMo/BF16 models) | Per item |

## 8. Risks

- **DAVE receive regressions.** Discord changes voice protocol; discord.js is the most maintained
  path. Pin versions, keep #11653 patched, and soak-test before relying on it.
- **Consent and policy.** Recording people is ours to get right: the bot announces itself in the
  channel when it starts, `/scribe stop` is available to anyone in the call, and audio is not kept.
  Not legal advice; Harry decides where it is used.
- **GPU stack fragility.** CTranslate2 and onnxruntime both need CUDA 12 / cuDNN 9; pin
  onnxruntime-gpu 1.24.4 and the pip CUDA libraries together. A future ORT bump to CUDA 13 must
  move both.
- **harry-box availability.** It sleeps when idle; covered in 3.6.
- **Public exposure (phase 2).** The edge is the only public surface; tailnet ACL and two layers
  of auth limit it.
- **Licences.** Parakeet and Canary are CC-BY-4.0 (attribution), Whisper MIT; fine for personal
  use.

## 9. Decisions needed from Harry

- **H1 Bot identity.** A new Discord application for scribe (recommended, keeps its permissions
  separate), or reuse an existing one. Either way the token goes in `secrets/api-keys.json`.
- **H2 Where it may record.** Which server/channels the bot is invited to, and whether the
  in-channel notice is enough consent for the people in them.
- **H3 Public ingress.** Tailnet-only (recommended until the Chrome extension needs off-tailnet
  use), or `scribe.<domain>` on the tools host. If public: the domain, a tagged Tailscale auth key
  for the tools host, and the ACL change.
- **H4 Audio retention.** Off (recommended), or keep audio for N days.
