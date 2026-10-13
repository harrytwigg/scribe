# scribe

Self-hosted call transcription where you pick the speech-recognition model. A Discord bot
transcribes each speaker in a voice channel, and a Chrome extension transcribes Google Meet (or
any call in a browser tab) without anything joining the call. The transcription runs on a local GPU with open-source
models (NVIDIA Parakeet and Canary, OpenAI Whisper and Distil-Whisper) chosen in a web GUI.

- `server/`: Python ASR server and GUI (FastAPI; faster-whisper and onnx-asr on CUDA).
- `bot/`: Discord bot (discord.js, `@discordjs/voice` with DAVE).
- `extension/`: Chrome extension for Meet and other browser calls (tab audio + mic, no bot).
- `deploy/`: systemd user units and the install script.
- `docs/spec.md`: design, decisions, phases. `docs/research/`: the research behind it.

## Using it

1. Open the GUI (`http://harry-box:8787` on the tailnet) and sign in with `SCRIBE_API_TOKEN`.
2. **Models**: download a model and press **Use**. Parakeet TDT 0.6B v2 is the default.
3. In a Discord voice channel, run `/scribe join`. Lines appear under **Live** as people speak.
   `/scribe stop` ends it (anyone in the call can); `/scribe optout` excludes you for good.
4. **History** keeps every transcript, with Markdown, SRT and JSON export.
5. **Try it** transcribes a file with the active model, to compare models on the same clip.

### Google Meet (or any call in a browser tab)

1. In Chrome, open `chrome://extensions`, turn on **Developer mode**, **Load unpacked**, and pick
   the `extension/` folder.
2. In the extension's **Settings**: the server (`http://harry-box:8787`), the API token and your
   name. Press **Allow microphone** once.
3. In the Meet tab, click the scribe icon, then **Transcribe this tab**. Lines from the call
   ("Others") and your mic (your name) appear under **Live** in the GUI. **Stop**, or closing the
   tab, ends it.

Tell people you are transcribing: unlike the Discord bot, nothing in the call announces it.

## Developing

```sh
cd server && uv sync              # Python 3.12, CUDA 12 wheels
uv run pytest                     # unit tests (fake engines, no GPU)
uv run pytest -m gpu -s           # every catalogue model on the GPU (downloads ~16 GB once)
uv run playwright install chromium && uv run pytest -m gui -s   # the GUI in a browser

cd bot && pnpm install            # Node 22.12+
pnpm test                         # unit tests
pnpm e2e                          # two replayed speakers through the bot pipeline into a real
                                  # server, with a model swap mid-call
pnpm build && node dist/src/smoke.js   # live: join an empty voice channel (DAVE handshake), leave

cd extension && npm test          # unit tests (Node 22, no dependencies)
cd server && uv run pytest -m extension -s   # the extension in Chromium into a real GPU server
```

`pnpm e2e` replays audio after `receiver.subscribe`: it does not exercise Discord's transport, DAVE
decryption or SSRC mapping. Only a real call does.

## Deploying

```sh
deploy/install.sh     # first run writes ~/.config/scribe/env from deploy/env.example; edit, rerun
```

The script copies the code to `~/.local/opt/scribe`, builds it and (re)starts the
`scribe-server` and `scribe-bot` user units. The server listens on the addresses in `SCRIBE_HOST`
and refuses to listen beyond loopback without an API token.
