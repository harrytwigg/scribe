"""Diarization with the real models (sherpa-onnx on the CPU) and Parakeet on the GPU, on a
three-speaker conversation built from LibriSpeech (fixtures/README.md).

    uv run pytest -m gpu tests/test_diarize_gpu.py -s
"""

import json
import os
import time
from pathlib import Path

import pytest
import soundfile
from fastapi.testclient import TestClient

from scribe_server.__main__ import build
from scribe_server.app import create_app
from scribe_server.audio import RATE, wav_bytes
from scribe_server.config import Settings

from .wer import wer

pytestmark = pytest.mark.gpu

FIXTURES = Path(__file__).parent / "fixtures"
TRUTH = json.loads((FIXTURES / "conversation.json").read_text())
MODEL = "parakeet-tdt-0.6b-v2"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    models_dir = Path(os.environ.get("SCRIBE_MODELS_DIR", "~/.local/share/scribe/models")).expanduser()
    settings = Settings(
        hosts=["127.0.0.1"], port=0, data_dir=data, models_dir=models_dir, api_token=None,
        device="cuda", keep_audio=False, default_model=MODEL, queue_depth=32,
    )
    services = build(settings)
    client = TestClient(create_app(services), base_url="http://localhost")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if status["active_model"] == MODEL and status["diarization"] not in ("loading", "off"):
            break
        time.sleep(1)
    assert client.get("/api/status").json()["diarization"] == "ready"
    yield client
    # Free the VRAM for the next GPU test module (the live server may hold a model too).
    services.models.shutdown()
    services.worker.stop()


def who_spoke(lines, start_key, end_key, speaker_key):
    """For each reference turn, the hypothesis speaker covering most of it."""
    out = []
    for turn in TRUTH:
        overlap = {}
        for line in lines:
            o = min(turn["end"], line[end_key]) - max(turn["start"], line[start_key])
            if o > 0:
                overlap[line[speaker_key]] = overlap.get(line[speaker_key], 0) + o
        out.append(max(overlap, key=overlap.get) if overlap else None)
    return out


def assert_consistent(found):
    """Each reference speaker always gets the same label, and no two share one."""
    mapping = {}
    for turn, label in zip(TRUTH, found):
        assert label is not None, f"no speaker found for {turn['speaker']} at {turn['start']} s"
        mapping.setdefault(turn["speaker"], set()).add(label)
    assert all(len(labels) == 1 for labels in mapping.values()), mapping
    assert len({next(iter(labels)) for labels in mapping.values()}) == 3, mapping


def use(client, model_id):
    r = client.post(f"/api/models/{model_id}/activate")
    assert r.status_code == 202, r.text
    deadline = time.monotonic() + 600
    while client.get("/api/status").json()["active_model"] != model_id:
        assert time.monotonic() < deadline, f"{model_id} did not activate"
        time.sleep(1)


# Parakeet and Whisper give word timings (one ASR call per utterance, words shared out by time);
# Canary gives none, so it transcribes each turn on its own.
@pytest.mark.parametrize("model_id", ["whisper-large-v3-turbo", "canary-1b-v2", MODEL])
def test_whole_recording(client, model_id):
    use(client, model_id)
    started = time.monotonic()
    r = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json", "diarize": "true"},
        files={"file": ("c.wav", (FIXTURES / "conversation.wav").read_bytes())},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    print(f"\nwhole: {time.monotonic() - started:.1f} s for 42 s, {body['speakers']} speakers")
    for seg in body["segments"]:
        print(f"  {seg['start']:5.1f}-{seg['end']:5.1f} {seg['speaker']}: {seg['text'][:70]}")
    assert body["speakers"] == 3
    found = who_spoke(body["segments"], "start", "end", "speaker")
    assert_consistent(found)
    reference = " ".join(t["text"] for t in TRUTH)
    hypothesis = " ".join(seg["text"] for seg in body["segments"])
    # Speech the segmenter misses never reaches the ASR, which a plain transcription would hear:
    # splitting by speaker must not cost words.
    plain = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "json"},
        files={"file": ("c.wav", (FIXTURES / "conversation.wav").read_bytes())},
    ).json()["text"]
    diarized_wer, plain_wer = wer(reference, hypothesis), wer(reference, plain)
    print(f"{model_id}: WER diarized {diarized_wer:.3f}, undiarized {plain_wer:.3f}")
    assert diarized_wer < 0.1 and diarized_wer <= plain_wer + 0.02


def test_live_utterances_keep_speakers(client):
    """The call arrives as two utterances (the extension cuts at 25 s): a speaker heard in both
    must keep one label."""
    audio, rate = soundfile.read(FIXTURES / "conversation.wav", dtype="float32")
    assert rate == RATE
    sid = client.post("/api/sessions", json={"source": "browser", "title": "diarization"}).json()["id"]
    lines = []
    cut = 25 * RATE
    for index, (a, b) in enumerate([(0, cut), (cut, len(audio))]):
        r = client.post(
            f"/api/sessions/{sid}/utterances",
            data={"speaker_id": "tab", "speaker_name": "Others", "started_at": str(a / RATE), "client_key": f"k{index}", "diarize": "true"},
            files={"file": ("u.wav", wav_bytes(audio[a:b]))},
        )
        assert r.status_code == 201, r.text
        lines += r.json()["lines"]
    for line in lines:
        line["end"] = line["started_at"] + line["duration_ms"] / 1000
        print(f"  {line['started_at']:5.1f}-{line['end']:5.1f} {line['speaker_id']} {line['speaker_name']}: {line['text'][:60]}")
    found = who_spoke(lines, "started_at", "end", "speaker_id")
    assert_consistent(found)
    assert all(line["model_id"] == MODEL for line in lines)
    # The same two utterances undiarized: the 25 s cut splits a word for both, so this is the fair
    # comparison.
    plain = " ".join(
        client.post("/v1/audio/transcriptions", files={"file": ("u.wav", wav_bytes(audio[a:b]))}).json()["text"]
        for a, b in [(0, cut), (cut, len(audio))]
    )
    reference = " ".join(t["text"] for t in TRUTH)
    live_wer, plain_wer = wer(reference, " ".join(line["text"] for line in lines)), wer(reference, plain)
    print(f"WER live diarized {live_wer:.3f}, undiarized {plain_wer:.3f}")
    assert live_wer < 0.1 and live_wer <= plain_wer + 0.02


def test_one_voice_stays_one_speaker(client):
    """Short turns of one older, noisier voice score lower against each other (0.38-0.61) than
    LibriSpeech does: JFK cut into three utterances must still be one speaker."""
    audio, _ = soundfile.read(FIXTURES / "jfk.wav", dtype="float32")
    sid = client.post("/api/sessions", json={"source": "browser"}).json()["id"]
    speakers = set()
    for index, (a, b) in enumerate([(0, 4 * RATE), (4 * RATE, 8 * RATE), (8 * RATE, len(audio))]):
        r = client.post(
            f"/api/sessions/{sid}/utterances",
            data={"speaker_id": "tab", "speaker_name": "Others", "started_at": str(a / RATE), "client_key": f"j{index}", "diarize": "true"},
            files={"file": ("u.wav", wav_bytes(audio[a:b]))},
        )
        speakers |= {line["speaker_id"] for line in r.json().get("lines", [])}
    assert speakers == {"tab#1"}
