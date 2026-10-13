#!/usr/bin/env bash
# Install or update scribe for this user: copy the code to ~/.local/opt/scribe, build it, and
# (re)start the systemd user units. Run from a checkout. Needs uv, pnpm and Node 22.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
dest="$HOME/.local/opt/scribe"
units="$HOME/.config/systemd/user"

if [ ! -f "$HOME/.config/scribe/env" ]; then
  mkdir -p "$HOME/.config/scribe"
  cp "$repo/deploy/env.example" "$HOME/.config/scribe/env"
  echo "Wrote ~/.config/scribe/env from the example: edit it, then run this again." >&2
  exit 1
fi

mkdir -p "$dest" "$units"
rsync -a --delete --exclude .venv --exclude node_modules --exclude dist --exclude __pycache__ \
  --exclude .pytest_cache "$repo/server/" "$dest/server/"
rsync -a --delete --exclude node_modules --exclude dist "$repo/bot/" "$dest/bot/"

# Reinstalling onnxruntime-gpu each time is cheap (uv's cache) and repairs a venv where the CPU
# onnxruntime package, which shares its directory, was installed or removed over it.
(cd "$dest/server" && uv sync --no-dev --frozen --reinstall-package onnxruntime-gpu)
(cd "$dest/bot" && pnpm install --frozen-lockfile && pnpm build)

cp "$repo/deploy/scribe-server.service" "$repo/deploy/scribe-bot.service" "$units/"
systemctl --user daemon-reload
systemctl --user enable scribe-server.service scribe-bot.service
systemctl --user restart scribe-server.service scribe-bot.service
systemctl --user --no-pager status scribe-server.service scribe-bot.service | head -20
