"""Run the server: `python -m scribe_server` (settings from the environment, see config.py)."""

import asyncio
import logging
import socket
from functools import partial

import uvicorn

from .app import Services, create_app
from .auth import Auth
from .catalogue import CATALOGUE
from .config import Settings
from .models import ModelManager
from .store import Store
from .worker import GpuWorker


def build(settings: Settings) -> Services:
    from . import cuda
    from .engines import load_engine
    from .vad import SileroVad

    if settings.device == "cuda":
        cuda.preload()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    models = ModelManager(
        CATALOGUE,
        settings.models_dir,
        loader=partial(_load, load_engine, settings.device),
        state_file=settings.data_dir / "state.json",
    )
    services = Services(
        models=models,
        store=Store(settings.data_dir / "scribe.db"),
        worker=GpuWorker(settings.queue_depth),
        vad=SileroVad(settings.models_dir),
        auth=Auth(settings.api_token, settings.trust_localhost, settings.extension_origins),
        audio_dir=settings.data_dir / "audio" if settings.keep_audio else None,
        allowed_hosts=settings.host_allowlist(),
        open_sessions_file=settings.data_dir / "open-sessions",
        browser_idle_s=settings.browser_idle_s,
    )
    models.restore(settings.default_model)
    return services


def _load(load_engine, device, entry, path):
    return load_engine(entry, path, device)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    settings.check()
    app = create_app(build(settings))
    sockets = []
    for host in settings.hosts:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, settings.port))
        sockets.append(sock)
    server = uvicorn.Server(uvicorn.Config(app, log_level="info", proxy_headers=False))
    logging.getLogger("scribe").info("listening on %s port %d", ", ".join(settings.hosts), settings.port)
    asyncio.run(server.serve(sockets=sockets))


if __name__ == "__main__":
    main()
