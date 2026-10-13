"""Real models on the GPU, through the HTTP API: download, activate, transcribe, swap.

    uv run pytest -m gpu            # every catalogue model (~16 GB of downloads the first time)
    SCRIBE_GPU_MODELS=parakeet-tdt-0.6b-v2,whisper-tiny.en uv run pytest -m gpu

Models are kept in SCRIBE_MODELS_DIR (default ~/.local/share/scribe/models) between runs.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile
from fastapi.testclient import TestClient

from scribe_server.__main__ import build
from scribe_server.app import create_app
from scribe_server.audio import RATE, wav_bytes
from scribe_server.catalogue import CATALOGUE
from scribe_server.config import Settings

from .wer import wer

pytestmark = pytest.mark.gpu

JFK = Path(__file__).parent / "fixtures" / "jfk.wav"
JFK_TEXT = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
WANTED = [m for m in os.environ.get("SCRIBE_GPU_MODELS", "").split(",") if m] or [e.id for e in CATALOGUE]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    models_dir = Path(os.environ.get("SCRIBE_MODELS_DIR", "~/.local/share/scribe/models")).expanduser()
    settings = Settings(
        hosts=["127.0.0.1"], port=0, data_dir=data, models_dir=models_dir, api_token=None,
        device="cuda", keep_audio=False, default_model="", queue_depth=32,
    )
    services = build(settings)
    # The host allowlist (DNS-rebinding guard) admits localhost, not TestClient's "testserver".
    yield TestClient(create_app(services), base_url="http://localhost"), services
    services.worker.stop()


def wait_for(client, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        models = {m["id"]: m for m in client.get("/api/models").json()}
        if predicate(models):
            return models
        time.sleep(1)
    raise AssertionError("timed out")


def transcribe(client, data: bytes) -> dict:
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", data, "audio/wav")}, data={"response_format": "verbose_json"})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("model_id", WANTED)
def test_model_end_to_end(client, model_id):
    c, services = client
    models = {m["id"]: m for m in c.get("/api/models").json()}
    if models[model_id]["status"] == "not_downloaded":
        assert c.post(f"/api/models/{model_id}/download").status_code == 202
        models = wait_for(c, lambda ms: ms[model_id]["status"] != "downloading", timeout=1800)
    assert models[model_id]["status"] == "downloaded", models[model_id]["error"]

    assert c.post(f"/api/models/{model_id}/activate").status_code == 202
    models = wait_for(c, lambda ms: not ms[model_id]["loading"], timeout=600)
    assert models[model_id]["active"], models[model_id]["error"]

    data = JFK.read_bytes()
    transcribe(c, data)  # warm-up
    result = transcribe(c, data)
    score = wer(JFK_TEXT, result["text"])
    print(f"\n{model_id}: wer={score:.3f} asr={result['asr_ms']}ms text={result['text']!r}")
    assert result["model"] == model_id
    assert score <= 0.15, result["text"]

    silent = transcribe(c, wav_bytes(np.zeros(5 * RATE, dtype=np.float32)))
    assert silent["text"] == ""
    noise = transcribe(c, wav_bytes((np.random.default_rng(0).standard_normal(3 * RATE) * 0.003).astype(np.float32)))
    assert noise["text"] == "", noise["text"]


def test_long_recording_is_chunked(client):
    """A 75 s recording (the clip seven times) is cut at <= 25 s and transcribed in full."""
    c, _ = client
    audio, rate = soundfile.read(JFK, dtype="float32")
    gap = np.zeros(int(0.8 * rate), dtype=np.float32)
    long = np.concatenate([np.concatenate([audio, gap]) for _ in range(7)])
    result = transcribe(c, wav_bytes(long, rate))
    assert wer(" ".join([JFK_TEXT] * 7), result["text"]) <= 0.15, result["text"]
