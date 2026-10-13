# ASR server / model / meeting-capture research (2026-10-12)

Confidence: [V] = read from the source this session; [U] = uncertain or inferred.

## 0. Bottom line
- **speaches already does what you want**: faster-whisper (CTranslate2) and Parakeet (onnx-asr) in one process, an OpenAI-compatible API, and models loaded on demand and unloaded after a TTL. [V: src/speaches/executors/parakeet.py, config.py]
  - It pins `requires-python ==3.12.*`, so use its Docker image or `uv python install 3.12`, not the system 3.13.
  - Its Parakeet registry only lists HF repos matching `istupakov/parakeet-tdt*`. Canary, Moonshine and Voxtral are not covered.
- Handy's engine is now **transcribe.cpp** (MIT, GGUF/ggml, CUDA/Vulkan). It covers Whisper, Parakeet, Canary, Voxtral, Granite, Moonshine and Qwen3-ASR in one engine, but it has no server and no published Python wheels yet.
- For a single user, a **Chrome extension is easier than a bot**.

## 1. Servers

| Project | Licence | Models | Runtime switching | Turing GPU | Diarization / word timestamps | Maintenance |
|---|---|---|---|---|---|---|
| **speaches** (speaches-ai/speaches) | MIT | faster-whisper (large-v3, turbo, distil), Parakeet TDT via onnx-asr, Piper/Kokoro TTS | `model` field per request. Lazy load, TTL unload (`stt_model_ttl` default 300 s, -1 = never), `preload_models`. `GET/POST/DELETE /v1/models/{id}` downloads and removes models. Has a Gradio UI. | Yes: CT2 on CUDA 12 and ORT CUDA provider (priority 100). TensorRT excluded. | Diarization router (added in v0.9.0-rc.3). Word timestamps via faster-whisper. | Last release v0.9.0-rc.3 (2025-12-27). Last commit I saw was 2026-04-18 (realtime API work). Alive but slowing. |
| **whisper.cpp server** | MIT [U, not shown on page] | Whisper only (ggml) | `/load` endpoint swaps the model file. Native route is `/inference`. An OpenAI-style path can be set with `--inference-path` [U]. | CUDA, fine | `--diarize` is stereo-only. tinydiarize needs a special model. | Active |
| **WhisperLive** (collabora) | MIT | faster_whisper, TensorRT, OpenVINO backends | Loads a model per client connection, or one shared model with `-fw`. `--enable_rest` adds an OpenAI-style REST route. Ships a Chrome extension. | Yes (faster_whisper) | Optional pyannote diarization | Active [U] |
| **parascribe** (hadsie) | MIT | Parakeet only, via onnx-asr (default `istupakov/parakeet-tdt-0.6b-v3-onnx`) | `PARASCRIBE_MODELS` allow-list, `MAX_RESIDENT_MODELS`, `MODEL_TTL_S`. OpenAI `/v1/audio/transcriptions` and `/v1/models`. | Yes (`EXECUTION_PROVIDER=cuda`) | pyannote diarization, opt-in per request | Small project [U] |
| **Scriberr** | MIT | Whisper, Parakeet, Canary, pyannote/NeMo diarization | Chosen in the web UI [U] | CUDA image covers Pascal through Ada, so Turing is fine | Built-in diarization | Author says development is paused |
| **WhisperX** | BSD-2 | faster-whisper plus wav2vec2 alignment | Library/CLI, not a server | Yes | Best word-level alignment, pyannote diarization (needs an HF token; the community-1 model is CC-BY-4.0) | Active. Wants CUDA 12.8 toolkit [V] |
| **insanely-fast-whisper** | Apache-2.0 | HF Transformers Whisper | CLI only | Works with `--flash False`; FA2 is not available on Turing | pyannote | Community, 0.0.15, effectively stale |
| **Wyoming (faster-whisper)** | MIT | Whisper, plus sherpa Parakeet v3 and onnx-asr backends | Set by CLI flag | Whisper on GPU; sherpa stays on CPU | No | Built for Home Assistant voice commands, not long-form |
| **NeMo** | Apache-2.0 | Parakeet, Canary, Canary-Qwen | A library with no server. Riva/NIM is the server route. | See below | Timestamps yes | v3.0.0, torch>=2.6 |
| **Riva/NIM ASR** | NVIDIA AI Enterprise terms | Parakeet 0.6b | Profile per container | I found no Turing statement. Matrix says Blackwell is not supported for TDT, and the profile uses ~14 GB GPU. | Yes | Heavy; not recommended |
| **Vexa's STT unit** (`deploy/transcription`) | Apache-2.0 | faster-whisper, set by `MODEL_SIZE` | Restart to change | Yes | n/a | Part of Vexa |

Sources:
- https://github.com/speaches-ai/speaches
- https://github.com/hadsie/parascribe
- https://github.com/collabora/WhisperLive
- https://github.com/ggml-org/whisper.cpp/tree/master/examples/server
- https://github.com/rishikanthc/Scriberr
- https://github.com/m-bain/whisperX
- https://github.com/rhasspy/wyoming-faster-whisper
- https://github.com/Vexa-ai/vexa/tree/main/deploy/transcription
- https://docs.nvidia.com/nim/speech/26.05.0/reference/support-matrix/asr.html

Also useful:
- sherpa-onnx (Apache-2.0) runs Parakeet v2, Whisper and Moonshine, has a websocket server and diarization, and has CUDA builds.
- It ships no model-switching GUI. https://github.com/k2-fsa/sherpa-onnx

## 2. Handy (cjpais/Handy)
- **What it is**: a Tauri v2 push-to-talk dictation app. Rust backend, TypeScript/Bun frontend. Licence is **MIT**, but the name and logo are not open. Latest release is v0.9.8 (2026-10-03). https://github.com/cjpais/Handy
- **Model selection**: Settings > Models lists catalogue entries, downloads them with progress, and lets you pick one. Dropping a custom Whisper `.bin` into the models folder is auto-detected.
- **Engines (Cargo.toml for 0.9.8)**:
  - `transcribe-rs 0.3.8` with only the `onnx` feature, covering Parakeet, Moonshine, SenseVoice, GigaAM, Canary and Cohere.
  - `transcribe-cpp 0.4.0` for GGUF models (Whisper, Parakeet, Voxtral, Qwen3-ASR and others), with Vulkan on Linux, Metal on macOS and dynamic backends.
  - There is **no whisper-rs** now. The README recommends "Parakeet Unified EN 0.6B" in GGUF.
- **Where the model-picker logic lives**: `src-tauri/src/managers/model.rs` (ModelInfo, EngineType, ModelSource, QuantFile, GGUF header capability prober). It downloads from URL (sha256) or via hf-hub, and it is tied to Tauri.
- **Reuse in a server**:
  - The `ModelInfo` and catalogue schema is a good template, but the code is Tauri-coupled.
  - `transcribe-rs` (MIT) is a Rust library. It wants 16 kHz mono WAV and has CUDA/ORT providers. You could wrap it in an axum server, but that is more work than Python.
  - **transcribe.cpp** (https://github.com/handy-computer/transcribe.cpp, MIT) is the best reusable engine:
    - 16 model families. Features per family: Granite does diarize and word timestamps, Parakeet gives token timestamps.
    - It also includes a Sortformer diarizer, VAD and langid.
    - CUDA is a build flag (`-DTRANSCRIBE_CUDA=ON`, needs nvcc).
    - The Python binding is "in development"; build from source. No server.
  - Its benchmark and catalogue site (FLEURS, Q8_0): https://models.handy.computer
- **Verdict**: copy the UX idea, not the code. Build the picker over speaches' `/v1/models` API.

## 3. English meeting-speech model comparison
Numbers are from the HF Open ASR Leaderboard (mean WER over 8 English sets, then RTFx). AMI is the meeting set.

| Model | Mean WER | RTFx | AMI | Licence | Notes |
|---|---|---|---|---|---|
| Granite-speech-4.1-2b | 5.33 | 231 | 8.09 | Apache-2.0 | 2B BF16 LLM-style. Timestamps and diarization only in the "-plus" variant. In transcribe.cpp. Long-form handling unclear. |
| Canary-Qwen-2.5B | 5.63 | 418 | 10.18 | CC-BY-4.0 | **Requires NeMo/PyTorch.** Trained on <=40 s clips. No timestamps. Turing/fp16 behaviour [U]. |
| **Parakeet-TDT-0.6B-v2** (EN) | **6.05** | **3386** | 11.16 | CC-BY-4.0 | 24 min per pass; longer with local attention. Word, segment and char timestamps. Tiny (~2 GB). |
| **Parakeet-TDT-0.6B-v3** (25 languages) | 6.34 | 3333 | 11.31 | CC-BY-4.0 | Same strengths, plus language auto-detect. Up to 3 h with local attention. |
| Canary-1B-flash | 6.35 | 1046 (A100) | 13.11 | CC-BY-4.0 | NeMo. Under 40 s per input, so chunking needed. Timestamps are experimental. |
| Kyutai STT 2.6B-en | 6.4 | 88 | 12.17 | CC-BY-4.0 | BF16 weights, up to 2 h, moshi server, 2.5 s delay. |
| Moonshine v2 medium streaming (245M) | 6.65 (vendor-reported) | n/a | n/a | MIT | CPU-friendly. Streaming. |
| Distil-large-v3.5 | 7.21 | 202 | n/a | MIT | About 1.5x faster than turbo. CT2 repo `distil-whisper/distil-large-v3.5-ct2`. Long-form WER 10.04. |
| Voxtral-Mini-3B-2507 | 7.05 | n/a | 16.3 | Apache-2.0 | About 9.5 GB in fp16 (vLLM). Up to 30 min of audio. No timestamps. |
| Whisper large-v3 | 7.44 | 146 | n/a | MIT | |
| Whisper large-v3-turbo | 7.83 | 200 | n/a | MIT | |
| Voxtral-Mini-4B-Realtime-2602 | 7.68 | n/a | n/a | Apache-2.0 | Needs >=16 GB and vLLM nightly. Too tight on 16 GB; skip. |

Sources:
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/nvidia/canary-qwen-2.5b
- https://huggingface.co/nvidia/canary-1b-flash
- https://huggingface.co/kyutai/stt-2.6b-en
- https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- https://huggingface.co/distil-whisper/distil-large-v3.5
- https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- https://arxiv.org/pdf/2510.06961 (whisper numbers)
- https://moonshine-voice.readthedocs.io/en/latest/moonshine-vs-whisper/

Notes:
- I did not find reliable Open ASR Leaderboard numbers for Moonshine v2. The paper table was unreadable through the fetcher, so verify the Whisper figures and Moonshine on the live leaderboard.
- **Runs without NeMo (onnxruntime / onnx-asr):** Parakeet TDT v2/v3, Canary v1/v2, Whisper (ONNX), and via sherpa-onnx also Moonshine. Parakeet v2/v3 also run through transcribe.cpp.
- **Needs NeMo/PyTorch:** Canary-Qwen-2.5B and Canary-1B-flash in full form. The Parakeet-ONNX route covers only TDT/CTC/Canary v1/v2-style graphs.
- **Turing caveats:** no bf16 and no FA2.
  - Use `float16` or `int8_float16` for CT2 and fp16/int8 ONNX for Parakeet.
  - NVIDIA's Parakeet v2 card lists Ampere, Blackwell, Hopper and Volta, not Turing, though a T4 appears in its test hardware.
  - The BF16-only LLM-style models (Granite, Voxtral, Kyutai, Canary-Qwen) may need fp16 casting and may overflow. Treat them as experimental on this card [U].
- **Recommendation:** Parakeet-TDT-0.6B-v2 (or v3) as the default meeting model; whisper-large-v3-turbo or distil-large-v3.5 as the fallback for non-English or proper-noun-heavy audio.

## 4. One process, both families, CUDA (simplest)
**Option A, a server (recommended): speaches in Docker (CUDA tag) or `uv venv --python 3.12`.** It bundles `faster-whisper`, `ctranslate2`, `onnx-asr` and `onnxruntime-gpu`.

**Option B, direct in Python:**
```
uv venv --python 3.12
uv pip install faster-whisper "onnx-asr[gpu,hub]" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
uv pip install "onnxruntime-gpu>=1.24.4,<1.27"
```
- `faster-whisper` is 1.2.1 (2025-10-31), but commits continued through 2026-10-10. `ctranslate2` is 4.8.2 (2026-08-31).
- `onnx-asr` is 0.12.0 (2026-07-15), MIT, Python 3.10-3.14. Excluded ORT versions are !=1.24.1, 1.25.*, 1.26.0.
- **CT2 needs CUDA 12 cuBLAS and cuDNN 9.** Set `LD_LIBRARY_PATH` to the pip `nvidia/*/lib` directories before starting Python. https://github.com/SYSTRAN/faster-whisper
- **Library-clash risk:** from ORT 1.27, the PyPI `onnxruntime-gpu` is built for CUDA 13.0. Separate CUDA 12.8 packages exist. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
  - Both stacks use cuDNN 9 (`libcudnn.so.9`) and could collide in one process.
  - So pin ORT below 1.27, or use the CUDA 12.8 build, to keep everything on cu12. Your driver (CUDA 13.0) runs cu12 userspace fine.
  - I did not test this; run a smoke test loading both models.
- **Usage:**
  - `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", providers=["CUDAExecutionProvider"])`; call `.with_timestamps()` and `.with_vad(...)` for long audio. The model accepts only ~20-30 s per chunk without VAD.
  - `WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")`.
- sherpa-onnx or NeMo are heavier alternatives. NeMo 3.0.0 pulls torch>=2.6 and is only needed for Canary-Qwen.
- Docs: https://github.com/istupakov/onnx-asr

## 5. Meeting capture
| Project | Licence | Notes |
|---|---|---|
| **Vexa** (Vexa-ai/vexa, v0.12) | **Apache-2.0** | Gateway plus per-bot containers (Docker, K8s or process). Meet, Teams, Zoom. `make all` is GPU-free; you point `TRANSCRIPTION_SERVICE_URL` and `TRANSCRIPTION_SERVICE_TOKEN` at an STT service. The README calls it "faster-whisper, OpenAI-compatible". **Whether speaches fits that contract is unverified.** `POST /bots` returns 503 when STT is missing. Large (2.6k commits), so heavy for one user. https://github.com/Vexa-ai/vexa |
| **Attendee** (attendee-labs) | **Elastic License 2.0** (source-available; no managed-service resale) | Django + Postgres + Redis in one image. Meet, Zoom, Teams. Providers in code: Deepgram (base URL overridable), AssemblyAI, Gladia, Sarvam, OpenAI, Kyutai (`server_url`), `custom_async`, and Meet closed captions. **No native OpenAI-compatible self-hosted ASR**, so use the Kyutai or `custom_async` hooks [U]. Guest join by default; `use_bot_login` for sign-in, needs support to enable per docs. Very active (push 2026-10-12). |
| **Meeting BaaS** (Meeting-BaaS/meet-teams-bot) | licence not detected by GitHub (NOASSERTION) | Hosted BaaS plus a bot repo. Not clearly self-hostable, so I would skip it. |
| **Meetily** (Zackriya-Solutions/meetily) | MIT | Not a bot: a Tauri desktop app that mixes mic and system audio. Whisper or Parakeet ONNX locally, Ollama summaries. Linux has no installer and must be built; ASR endpoint is not pluggable [V]. |
| **Recall.ai** (commercial reference) | n/a | $0.50/h recording plus $0.15/h built-in transcription, 5 free hours. https://www.recall.ai/pricing |

**Google Meet bot restrictions:**
- Bots joining as guests wait in the lobby, and only a host sees the admit prompt.
- If the host disables "anyone with the link can ask to join", guests are auto-declined.
- Google added a "potential risk" queue in 2026 (reported starting 2026-03-24) where suspicious joins, such as bots, default to auto-deny. https://www.neowin.net/news/google-meet-gets-a-new-update-to-stop-bots-from-joining-meetings/
- A signed-in bot whose account email is on the invite skips the lobby. Recall documents this.
- A bot account is a Google account for automated use, which may breach ToS. Treat with caution.
- The Meet Media API remains developer preview as of the June 2026 sources; all participants must be enrolled. Not viable.

**Chrome extension (recommended for one user):**
- MV3 flow: the service worker calls `chrome.tabCapture.getMediaStreamId({targetTabId})` after a user gesture (Chrome 116+). It creates an offscreen document with reason `USER_MEDIA`, which calls `getUserMedia({audio:{mandatory:{chromeMediaSource:'tab', chromeMediaSourceId:id}}})`. https://developer.chrome.com/docs/extensions/how-to/web-platform/screen-capture
- Captured tab audio is silenced for you, so route it back to `AudioContext.destination`.
- Mix the mic in the offscreen doc with a second `getUserMedia` into a `MediaStreamDestination`. Mic permission inside an offscreen doc is a known problem; grant it first from an extension page.
- Stream 16 kHz PCM from an AudioWorklet over WebSocket to your server.
- Keep tab and mic on separate channels, which gives you free "you vs. them" diarization.
- Open-source references: Recall's `recallai/chrome-recording-transcription-extension` (tab + mic mix, offscreen doc; records locally rather than streaming), WhisperLive's `Audio-Transcription-Chrome` (WebSocket streaming to a Whisper server), and `asimons81/chrome-video-recorder` (mix code).
- No ready-made project I found does tabCapture + mic + WebSocket + speaches. Expect to write about 300 lines.

**Recommendation:** extension for Meet in a browser (no lobby, no bot account, works on any call). Use a bot (Vexa) only if you need unattended meetings.
