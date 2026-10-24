# scribe

Self-hosted call transcription where you pick the speech-recognition model. A Discord bot
transcribes each speaker in a voice channel, and a Chrome extension transcribes Google Meet (or
any call in a browser tab) without anything joining the call. The transcription runs on a local GPU with open-source
models (NVIDIA Parakeet and Canary, OpenAI Whisper and Distil-Whisper) chosen in a web GUI.

- `server/`: Python ASR server and GUI (FastAPI; faster-whisper and onnx-asr on CUDA).
- `bot/`: Discord bot (discord.js, `@discordjs/voice` with DAVE).
- `extension/`: Chrome extension for Meet and other browser calls (tab audio + mic, no bot).
- `server/scribe_server/cli.py`, `podcast.py`: the `scribe` command and the podcast link resolver (`docs/cli.md`).
- `deploy/`: systemd user units and the install script.
- `docs/spec.md`: design, decisions, phases. `docs/research/`: the research behind it.

## Using it

1. Open the GUI (`http://harry-box:8787` on the tailnet). There is no sign-in: whoever can reach the server can use it.
2. **Models**: download a model and press **Use**. Parakeet TDT 0.6B v2 is the default.
3. In a Discord voice channel, run `/scribe join`. Lines appear under **Live** as people speak.
   `/scribe stop` ends it (anyone in the call can); `/scribe optout` excludes you for good.
4. **History** keeps every transcript, with Markdown, SRT and JSON export. Click a speaker's name
   to rename them for that session.
5. **Try it** transcribes a file with the active model, to compare models on the same clip. Tick
   **Separate speakers** to have a recording split into Speaker 1, 2, ...

### Podcasts and recordings

- **Transcribe** tab: upload audio files, or paste Spotify, Apple Podcasts, RSS or mp3 links (one per
  line). They queue; each row has a progress bar for the download, decoding and transcribing, and
  shows its transcript with Markdown/SRT/JSON export when done. Transcripts are kept in **History**.
- **CLI** (for scripts and agents; `deploy/install.sh` puts it at `~/.local/bin/scribe`):

  ```sh
  scribe transcribe https://open.spotify.com/episode/<id>            # transcript on stdout
  scribe transcribe talk.m4a --speakers --format json --out talk.json
  scribe transcribe LINK1 LINK2 LINK3 --out transcripts/            # a queue, one file each
  ```

  Spotify's own audio is DRM-protected, so a Spotify link is resolved to the show's public RSS copy
  (like podcastmp3.com); a Spotify exclusive has none and fails with a clear message. See
  [docs/cli.md](docs/cli.md) for sources, options, JSON output and exit codes.

### Google Meet (or any call in a browser tab)

1. In Chrome, open `chrome://extensions`, turn on **Developer mode**, **Load unpacked**, and pick
   an `extension/` folder that will stay where it is: Chrome forgets an unpacked extension whose
   folder is deleted or moved. On the machine that runs scribe, `deploy/install.sh` keeps a copy at
   `~/.local/opt/scribe/extension`; from any other machine, the dashboard's **Extension** tab
   downloads it as a zip to unzip somewhere permanent. Repeat for each Chrome profile.
2. In the extension's **Settings**: the server (`http://harry-box:8787`) and your
   name. Press **Allow microphone** once.
3. In the Meet tab, click the scribe icon, then **Transcribe this tab**. Lines from the call
   (as Speaker 1, Speaker 2, ...: click a name to rename them) and your mic (your name) appear
   under **Live** in the GUI. **Stop**, closing or
   leaving the tab, or 10 minutes with no speech ends it.

Use headphones: on speakers, the others may hear themselves echo. Tell people you are
transcribing: unlike the Discord bot, nothing in the call announces it.

## Developing

```sh
cd server && uv sync --reinstall-package onnxruntime-gpu   # Python 3.12, CUDA 12 wheels; the
                                  # reinstall repairs a venv that still had the CPU onnxruntime
                                  # wheel, which shares onnxruntime-gpu's directory
uv run pytest                     # unit tests (fake engines, no GPU; the network is faked too)
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
and refuses to listen on every interface (`0.0.0.0`): there is no API token, so name the addresses
(loopback and the tailnet's) and keep the server off any network you do not trust.
