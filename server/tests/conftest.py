import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scribe_server.app import Services, create_app
from scribe_server.audio import RATE, wav_bytes
from scribe_server.auth import Auth
from scribe_server.catalogue import CatalogueEntry
from scribe_server.models import ModelManager
from scribe_server.store import Store
from scribe_server.worker import GpuWorker

TOKEN = "test-token"


def entry(model_id: str) -> CatalogueEntry:
    return CatalogueEntry(
        id=model_id, name=model_id, engine="faster-whisper", repo=f"test/{model_id}", description="",
        languages="English", licence="MIT", size_mb=1, accuracy=3, speed=3, files=["*"],
    )


CATALOGUE = [entry("alpha"), entry("beta"), entry("gamma")]


class FakeEngine:
    """Answers `<model id>: <seconds of audio>`; `delay` holds the worker to test swaps."""

    def __init__(self, model_id: str, delay: float = 0.0):
        self.model_id = model_id
        self.delay = delay
        self.closed = False
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> str:
        assert not self.closed, "transcribe on an unloaded model"
        self.calls += 1
        time.sleep(self.delay)
        return f"{self.model_id}: {len(audio) / RATE:.1f}s"

    def close(self) -> None:
        self.closed = True


class FakeLoader:
    def __init__(self):
        self.engines: dict[str, FakeEngine] = {}
        self.delay = 0.0
        self.load_gate: threading.Event | None = None

    def __call__(self, e: CatalogueEntry, path: Path) -> FakeEngine:
        if self.load_gate:
            self.load_gate.wait(5)
        engine = FakeEngine(e.id, self.delay)
        self.engines[e.id] = engine
        return engine


class FakeDownloader:
    """Writes a file; `gate` holds it mid-download, `fail` makes it raise."""

    def __init__(self):
        self.gate: threading.Event | None = None
        self.fail = False

    def __call__(self, e, dest: Path, download) -> None:
        download.total = 100
        (dest / "model.bin").write_bytes(b"x" * 40)
        if self.gate:
            while not self.gate.wait(0.02):
                if download.cancelled:
                    raise RuntimeError("cancelled")
        if self.fail:
            raise RuntimeError("network down")
        (dest / "model.bin").write_bytes(b"x" * 100)


def energy_vad(audio: np.ndarray) -> list[tuple[int, int]]:
    """Spans where 20 ms frames are louder than a threshold: enough to stand in for Silero."""
    frame = RATE // 50
    loud = [np.sqrt(np.mean(audio[i : i + frame] ** 2)) > 0.01 for i in range(0, len(audio), frame)]
    spans, start = [], None
    for index, on in enumerate(loud + [False]):
        if on and start is None:
            start = index * frame
        elif not on and start is not None:
            spans.append((start, min(index * frame, len(audio))))
            start = None
    limit = 25 * RATE
    out = []
    for a, b in spans:
        while b - a > limit:
            out.append((a, a + limit))
            a += limit
        out.append((a, b))
    return out


def tone(seconds: float, rate: int = RATE, channels: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([wave] * channels, axis=1) if channels > 1 else wave


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def wav(audio: np.ndarray, rate: int = RATE) -> bytes:
    return wav_bytes(audio, rate)


class Harness:
    def __init__(self, tmp_path: Path, token: str | None = TOKEN):
        self.loader = FakeLoader()
        self.downloader = FakeDownloader()
        self.models = ModelManager(
            CATALOGUE, tmp_path / "models", loader=self.loader, downloader=self.downloader,
            state_file=tmp_path / "state.json",
        )
        self.worker = GpuWorker(depth=4)
        self.services = Services(
            models=self.models, store=Store(tmp_path / "scribe.db"), worker=self.worker,
            vad=energy_vad, auth=Auth(token),
        )
        self.app = create_app(self.services)
        self.client = TestClient(self.app)

    def ready(self, model_id: str = "alpha") -> None:
        self.models.download(model_id)
        self.models.wait_for_download(model_id)
        self.models.activate(model_id, wait=True)


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.worker.stop()
