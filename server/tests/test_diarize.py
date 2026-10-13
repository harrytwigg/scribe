"""Diarization logic without the models: turn tidying, speaker matching, and the API flow with a
fake diarizer that tells voices apart by pitch."""

import numpy as np

from scribe_server.audio import RATE
from scribe_server.diarize import DiarizerSlot, SpeakerBook, Turn, tidy

from .conftest import Harness, energy_vad, silence, wav


def pitch(hz: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class FakeDiarizer:
    """Turns are loud spans; a turn's voice is a one-hot vector of its pitch (in 100 Hz bands)."""

    def _turns(self, audio):
        out = []
        for a, b in energy_vad(audio):
            spectrum = np.abs(np.fft.rfft(audio[a:b]))
            band = int(np.fft.rfftfreq(b - a, 1 / RATE)[spectrum.argmax()] // 100)
            out.append(Turn(a / RATE, b / RATE, band))
        return tidy(out)

    @staticmethod
    def voice(band: int) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        v[band] = 1
        return v

    def turns(self, audio, book):
        pairs = []
        for turn in self._turns(audio):
            pairs.append((turn, book.match(self.voice(turn.label), turn.duration)))
        return pairs

    def whole(self, audio):
        numbers = {}
        return [(t, numbers.setdefault(t.label, len(numbers) + 1)) for t in self._turns(audio)]


def diarized(tmp_path) -> Harness:
    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(FakeDiarizer())
    h.ready()
    return h


def test_tidy_joins_and_folds():
    turns = [Turn(0, 2, 0), Turn(2.2, 3, 0), Turn(3, 3.3, 1), Turn(3.3, 6, 2), Turn(6.5, 9, 0), Turn(10, 11, 0)]
    got = [(t.start, t.end, t.label) for t in tidy(turns)]
    # 0+0 joined across a 0.2 s gap; the 0.3 s sliver of 1 folds into the turn before it; the last
    # two turns of 0 stay apart across their 1 s gap (someone else may have spoken in it).
    assert got == [(0, 3.3, 0), (3.3, 6, 2), (6.5, 9, 0), (10, 11, 0)]


def test_book_keeps_voices_apart_and_learns():
    book = SpeakerBook()
    a, b = np.eye(4, dtype=np.float32)[0], np.eye(4, dtype=np.float32)[1]
    assert book.match(a, 3).number == 1
    assert book.match(b, 3).number == 2
    near_a = a + 0.3 * b
    assert book.match(near_a, 3).number == 1  # similarity 0.96
    # A short turn never founds a speaker: it goes to the nearest known one.
    assert book.match(np.eye(4, dtype=np.float32)[2], 0.5).number in (1, 2)
    assert len(book.speakers) == 2


def test_diarized_utterances_keep_speakers_across_the_call(tmp_path):
    h = diarized(tmp_path)
    c = h.client
    sid = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    first = np.concatenate([pitch(300, 2), silence(0.8), pitch(700, 2), silence(0.5)])
    second = np.concatenate([pitch(700, 1.5), silence(0.8), pitch(300, 2.5), silence(0.8), pitch(1100, 2), silence(0.5)])
    form = lambda key: {"speaker_id": "tab", "speaker_name": "Others", "started_at": "100", "client_key": key, "diarize": "true"}  # noqa: E731

    r1 = c.post(f"/api/sessions/{sid}/utterances", data=form("tab:1"), files={"file": ("u.wav", wav(first))}).json()
    assert [(l["speaker_id"], l["speaker_name"]) for l in r1["lines"]] == [("tab#1", "Speaker 1"), ("tab#2", "Speaker 2")]
    assert r1["lines"][1]["started_at"] > 102.5  # offset by the turn's start in the utterance

    # Speaker 2 is renamed mid-call: their past lines change, and so do later ones.
    renamed = c.post(f"/api/sessions/{sid}/speakers/tab%232", json={"name": "Alice"}).json()
    assert renamed == {"speaker_id": "tab#2", "name": "Alice", "lines": 1}

    r2 = c.post(f"/api/sessions/{sid}/utterances", data=form("tab:2"), files={"file": ("u.wav", wav(second))}).json()
    assert [(l["speaker_id"], l["speaker_name"]) for l in r2["lines"]] == [
        ("tab#2", "Alice"), ("tab#1", "Speaker 1"), ("tab#3", "Speaker 3"),
    ]
    # A retry of the same utterance returns the stored lines and stores nothing new.
    again = c.post(f"/api/sessions/{sid}/utterances", data=form("tab:2"), files={"file": ("u.wav", wav(second))}).json()
    assert [l["id"] for l in again["lines"]] == [l["id"] for l in r2["lines"]]
    lines = c.get(f"/api/sessions/{sid}").json()["utterances"]
    assert len(lines) == 5
    assert [l["speaker_name"] for l in lines if l["speaker_id"] == "tab#2"] == ["Alice", "Alice"]


def test_speaker_books_survive_a_restart(tmp_path):
    h = diarized(tmp_path)
    c = h.client
    sid = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": "1", "diarize": "true"}
    c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(300, 2)))})
    c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(700, 2)))})

    from fastapi.testclient import TestClient

    from scribe_server.app import create_app

    restarted = TestClient(create_app(h.services), headers=dict(c.headers))
    r = restarted.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(700, 2)))}).json()
    assert r["lines"][0]["speaker_id"] == "tab#2"


def test_without_diarization_a_mixed_channel_is_one_speaker(harness):
    harness.ready()
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    assert c.get("/api/status").json()["diarization"] == "off"
    form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": "1", "diarize": "true"}
    r = c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(300, 2)))}).json()
    assert (r["speaker_id"], r["speaker_name"]) == ("tab", "Others")


def test_renamed_undiarized_speaker_keeps_the_name(harness):
    harness.ready()
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    form = {"speaker_id": "u1", "speaker_name": "ada_99", "started_at": "1"}
    c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(300, 2)))})
    assert c.post(f"/api/sessions/{sid}/speakers/u1", json={"name": "Ada"}).json()["lines"] == 1
    r = c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(300, 2)))}).json()
    assert r["speaker_name"] == "Ada"
    assert c.post(f"/api/sessions/{sid}/speakers/u1", json={"name": "  "}).status_code == 400


def test_diarized_upload(tmp_path):
    h = diarized(tmp_path)
    audio = np.concatenate([pitch(300, 2), silence(0.8), pitch(700, 2), silence(0.8), pitch(300, 1.5)])
    r = h.client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json", "diarize": "true"},
        files={"file": ("a.wav", wav(audio))},
    ).json()
    assert [seg["speaker"] for seg in r["segments"]] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert r["speakers"] == 2
    assert r["text"].splitlines()[1].startswith("Speaker 2: alpha:")
