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

    def whole(self, audio, num_speakers=None):
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


# -- review round 1 ---------------------------------------------------------------------------

import random  # noqa: E402
import threading  # noqa: E402

from scribe_server.diarize import Diarizer, Speaker, untangle  # noqa: E402
from scribe_server.worker import QueueFull  # noqa: E402


def spans(turns):
    return [(round(t.start, 2), round(t.end, 2), t.label) for t in turns]


def test_tidy_survives_overlapping_slivers():
    # Used to raise IndexError: a leading sliver folded into a "next" turn that did not exist.
    assert spans(tidy([Turn(0, 0.3, 0), Turn(0.1, 0.4, 1)])) == [(0, 0.4, 1)]
    assert spans(tidy([Turn(0, 0.2, 0)])) == [(0, 0.2, 0)]
    assert tidy([]) == []


def test_tidy_fuzz():
    rng = random.Random(7)
    for _ in range(20_000):
        turns = []
        for _ in range(rng.randint(0, 6)):
            a = rng.uniform(0, 10)
            turns.append(Turn(a, a + rng.choice([rng.uniform(0, 0.6), rng.uniform(0, 4)]), rng.randint(0, 2)))
        out = tidy(turns)
        for t in out:
            assert t.end > t.start
        for a, b in zip(out, out[1:]):
            assert a.end <= b.start + 1e-9, (turns, out)  # ordered, no overlap


def test_crosstalk_is_transcribed_once():
    # Two speakers overlapping by 0.28 s (as real sherpa output does): the one who started later
    # has the overlap, so the audio is cut at their start.
    assert spans(untangle([Turn(0.03, 5.80, 1), Turn(5.52, 9.28, 2)])) == [(0.03, 5.52, 1), (5.52, 9.28, 2)]
    # A short interjection inside someone's turn splits it.
    assert spans(untangle([Turn(0, 10, 0), Turn(5, 6, 1)])) == [(0, 5, 0), (5, 6, 1), (6, 10, 0)]


def test_new_speaker_numbers_never_collide():
    book = SpeakerBook([Speaker(1, np.eye(8, dtype=np.float32)[0], 3), Speaker(2, np.eye(8, dtype=np.float32)[1], 3),
                        Speaker(4, np.eye(8, dtype=np.float32)[3], 3)])
    assert book.match(np.eye(8, dtype=np.float32)[5], 3).number == 5


class StubbedDiarizer(Diarizer):
    """The real `turns` (per-turn matching, short-turn pooling, merging), with the models stubbed:
    `_turns` returns the clusterer output given, and a voice is a one-hot of the audio's pitch."""

    def __init__(self, clusterer_turns):
        self._live_lock = threading.Lock()
        self._given = clusterer_turns

    def _turns(self, audio, live=False):
        return tidy(self._given)

    def voice(self, audio):
        return FakeDiarizer.voice(int(np.fft.rfftfreq(len(audio), 1 / RATE)[np.abs(np.fft.rfft(audio)).argmax()] // 100))


def test_real_turns_matches_each_long_turn_on_its_own_voice():
    # The clusterer gives one label to two different voices (as it did to a short turn and the
    # next speaker); per-turn matching against the book tells them apart.
    audio = np.concatenate([pitch(300, 2.5), silence(1), pitch(700, 2.5)])
    d = StubbedDiarizer([Turn(0, 2.5, 0), Turn(3.5, 6, 0)])
    got = d.turns(audio, SpeakerBook())
    assert [(spans([t])[0][:2], sp.number) for t, sp in got] == [((0, 2.5), 1), ((3.5, 6), 2)]


def test_real_turns_pools_only_short_turns_by_label():
    # Two short turns (300 Hz) share a label with a long turn of someone else (700 Hz): pooled on
    # their own audio they match speaker 1, not the long turn's voice.
    audio = np.concatenate([pitch(300, 0.7), silence(0.8), pitch(300, 0.7), silence(0.8), pitch(700, 3)])
    book = SpeakerBook()
    book.match(FakeDiarizer.voice(3), 3)  # speaker 1: 300 Hz
    book.match(FakeDiarizer.voice(7), 3)  # speaker 2: 700 Hz
    d = StubbedDiarizer([Turn(0, 0.7, 5), Turn(1.5, 2.2, 5), Turn(3.0, 6.0, 5)])
    got = d.turns(audio, book)
    # The two short turns are one speaker-1 turn after merging; the long one is speaker 2.
    assert [sp.number for _, sp in got] == [1, 2]


def test_a_failing_diarizer_never_loses_speech(tmp_path):
    h = diarized(tmp_path)

    class Broken(FakeDiarizer):
        def turns(self, audio, book):
            raise RuntimeError("model exploded")

        def whole(self, audio):
            raise RuntimeError("model exploded")

    h.services.diarization = DiarizerSlot(Broken())
    c = h.client
    sid = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": "1", "diarize": "true", "client_key": "k"}
    r = c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(pitch(300, 2)))})
    assert r.status_code == 201 and r.json()["speaker_id"] == "tab" and r.json()["text"]
    up = c.post("/v1/audio/transcriptions", data={"response_format": "verbose_json", "diarize": "true"},
                files={"file": ("a.wav", wav(pitch(300, 2)))})
    assert up.status_code == 200 and up.json()["speakers"] == 1


def test_a_retry_after_a_busy_gpu_reuses_the_turns(tmp_path, monkeypatch):
    h = diarized(tmp_path)
    c = h.client
    sid = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": "1", "diarize": "true", "client_key": "k1"}
    audio = wav(np.concatenate([pitch(300, 2.5), silence(1), pitch(700, 2.5), silence(0.5)]))
    real_run = h.worker.run
    calls = {"n": 0}

    async def busy_once(fn):
        calls["n"] += 1
        if calls["n"] == 1:
            raise QueueFull("transcription queue is full")
        return await real_run(fn)

    monkeypatch.setattr(h.worker, "run", busy_once)
    assert c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", audio)}).status_code == 503
    voices = {sid_: w for sid_, _, w in h.services.store.voices(sid, "tab#")}
    r = c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", audio)}).json()
    assert [l["speaker_id"] for l in r["lines"]] == ["tab#1", "tab#2"]
    # The retry did not diarize again: the book learned this audio once.
    assert {sid_: w for sid_, _, w in h.services.store.voices(sid, "tab#")} == voices == {"tab#1": 2.5, "tab#2": 2.5}


def test_diarized_upload_numbers_speakers_without_gaps(tmp_path):
    h = diarized(tmp_path)
    # Speaker 2's turn is silence: it transcribes to nothing and is dropped; speaker 3 becomes 2.
    audio = np.concatenate([pitch(300, 2), silence(2), pitch(1100, 2)])
    h.services.diarization.value.whole = lambda a: [(Turn(0, 2, 0), 1), (Turn(2, 4, 1), 2), (Turn(4, 6, 2), 3)]
    r = h.client.post("/v1/audio/transcriptions", data={"response_format": "verbose_json", "diarize": "true"},
                      files={"file": ("a.wav", wav(audio))}).json()
    assert [seg["speaker"] for seg in r["segments"]] == ["Speaker 1", "Speaker 2"] and r["speakers"] == 2


def test_rename_is_capped(harness):
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    assert c.post(f"/api/sessions/{sid}/speakers/u1", json={"name": "x" * 81}).status_code == 400


def test_words_from_tokens():
    from scribe_server.engines import words_from_tokens

    words = words_from_tokens([" And", " so", ",", " my", " fe", "ll", "ow"], [0.2, 0.5, 0.8, 1.0, 1.3, 1.4, 1.5], 2.0)
    assert [(w.text, w.start, w.end) for w in words] == [(" And", 0.2, 0.5), (" so,", 0.5, 1.0), (" my", 1.0, 1.3), (" fellow", 1.3, 2.0)]


def test_words_go_to_the_turn_they_fall_in(tmp_path, monkeypatch):
    """With word timings the utterance is transcribed once and its words shared out by time."""
    from scribe_server.engines import Word

    from .conftest import FakeEngine

    def transcribe_words(self, audio):
        # One word per half second of audio, named after its time in this chunk.
        return [Word(f" w{i / 2:g}", i / 2, i / 2 + 0.4) for i in range(int(len(audio) / RATE * 2))]

    monkeypatch.setattr(FakeEngine, "transcribe_words", transcribe_words, raising=False)
    monkeypatch.setattr(FakeEngine, "word_timings", True, raising=False)
    h = diarized(tmp_path)
    c = h.client
    sid = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    audio = np.concatenate([pitch(300, 2), silence(1), pitch(700, 2), silence(0.5)])
    form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": "0", "diarize": "true"}
    r = c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(audio))}).json()
    assert [(l["speaker_id"], l["text"]) for l in r["lines"]] == [
        # One ASR call for the 0-5 s chunk. A word in the 2-3 s gap goes to the nearer turn: w2
        # (centre 2.2 s) to the first, w2.5 (centre 2.7 s) to the second.
        ("tab#1", "w0 w0.5 w1 w1.5 w2"),
        ("tab#2", "w2.5 w3 w3.5 w4 w4.5"),
    ]
    assert sum(e.calls for e in h.loader.engines.values()) == 0  # never transcribed per turn


# -- review round 2 ---------------------------------------------------------------------------


def test_a_diarized_upload_with_no_turns_is_still_transcribed(tmp_path):
    h = diarized(tmp_path)
    h.services.diarization.value.whole = lambda audio: []  # pyannote heard nothing; VAD and ASR do
    r = h.client.post("/v1/audio/transcriptions", data={"response_format": "verbose_json", "diarize": "true"},
                      files={"file": ("a.wav", wav(pitch(300, 2)))}).json()
    assert r["text"] == "Speaker 1: alpha: 2.0s" and r["speakers"] == 1


def test_a_retry_while_the_first_attempt_is_diarizing_waits_for_it(tmp_path):
    """The client timed out and retried while the first attempt was still on the CPU worker: the
    retry must reuse the first attempt's turns, not diarize (and learn) the audio again."""
    import asyncio
    import time as clock

    import httpx

    h = diarized(tmp_path)
    fake = h.services.diarization.value
    real_turns = fake.turns
    runs = []

    def slow_turns(audio, book):
        runs.append(1)
        clock.sleep(0.3)
        return real_turns(audio, book)

    fake.turns = slow_turns
    sid = h.client.post("/api/sessions", json={"source": "browser"}).json()["id"]
    audio = wav(np.concatenate([pitch(300, 2.5), silence(1), pitch(700, 2.5), silence(0.5)]))
    form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": "1", "diarize": "true", "client_key": "k1"}

    async def both():
        transport = httpx.ASGITransport(app=h.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", headers=dict(h.client.headers)) as c:
            post = lambda: c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", audio)})  # noqa: E731
            first = asyncio.ensure_future(post())
            await asyncio.sleep(0.1)  # the first is now diarizing
            return await asyncio.gather(first, post())

    first, retry = asyncio.run(both())
    assert first.status_code == retry.status_code == 201
    assert len(runs) == 1
    assert [l["id"] for l in first.json()["lines"]] == [l["id"] for l in retry.json()["lines"]]
    assert {sid_: w for sid_, _, w in h.services.store.voices(sid, "tab#")} == {"tab#1": 2.5, "tab#2": 2.5}


# -- whole recordings: scraps are folded into the real speakers ---------------------------------


def band_of(clip) -> int:
    return int(np.fft.rfftfreq(len(clip), 1 / RATE)[np.abs(np.fft.rfft(clip)).argmax()] // 100)


def conversation(*parts):
    """(audio, turns) for consecutive stretches (hz, seconds, label), a gap between each."""
    audio, turns, at = [], [], 0.0
    for hz, seconds, label in parts:
        audio += [pitch(hz, seconds), silence(1)]
        turns.append(Turn(at, at + seconds, label))
        at += seconds + 1
    return np.concatenate(audio), turns


def folded(parts):
    from scribe_server.diarize import absorb_scraps

    audio, turns = conversation(*parts)
    out = absorb_scraps(turns, audio, lambda clip: FakeDiarizer.voice(band_of(clip)))
    return sorted({t.label for t in out}), out


def test_a_scrap_that_sounds_like_a_speaker_joins_them():
    # Two real speakers (about 100 s each) and two scraps: one in the host's voice (300 Hz band),
    # one in nobody's. Both are under 15 s, so both join someone.
    labels, out = folded([(300, 100, 1), (700, 100, 2), (300, 4, 3), (900, 3, 4), (300, 50, 1), (700, 50, 2)])
    assert labels == [1, 2]
    assert [t.label for t in out if t.start == 202] == [1]  # the 300 Hz scrap is the host again


def test_a_real_third_voice_with_a_minute_of_speech_stays():
    # 18 s of a different voice is under 5% of 400 s but over 15 s, and it sounds like nobody else.
    labels, _ = folded([(300, 190, 1), (700, 190, 2), (500, 18, 3)])
    assert labels == [1, 2, 3]


def test_a_longer_scrap_that_sounds_like_a_speaker_joins_them():
    labels, _ = folded([(300, 190, 1), (700, 190, 2), (305, 18, 3)])
    assert labels == [1, 2]


def test_one_speaker_or_two_balanced_speakers_are_left_alone():
    assert folded([(300, 60, 1)])[0] == [1]
    assert folded([(300, 60, 1), (700, 50, 2), (300, 50, 1)])[0] == [1, 2]


def test_a_scrap_whose_voice_cannot_be_measured_goes_to_the_speaker_beside_it():
    from scribe_server.diarize import absorb_scraps

    def voice(clip):
        if band_of(clip) == 5:
            raise RuntimeError("no voice in that")
        return FakeDiarizer.voice(band_of(clip))

    audio, turns = conversation((300, 100, 1), (700, 100, 2), (500, 3, 3), (700, 50, 2))
    out = absorb_scraps(turns, audio, voice)
    assert {t.label for t in out} == {1, 2}
    assert [t.label for t in out if abs(t.start - 202) < 0.01] == [2]  # between two stretches of 2


def test_a_known_speaker_count_is_passed_on_and_skips_the_folding(monkeypatch):
    seen = {}

    class Counted(Diarizer):
        def __init__(self):
            self._whole_lock = threading.Lock()

        def _turns(self, audio, live=False, num_speakers=None):
            seen["n"] = num_speakers
            return [Turn(0, 1, 0), Turn(1, 2, 1)]

    monkeypatch.setattr("scribe_server.diarize.absorb_scraps", lambda turns, *a: [seen.setdefault("folded", True), turns][1])
    d = Counted()
    assert [n for _, n in d.whole(silence(2), num_speakers=2)] == [1, 2]
    assert seen == {"n": 2}  # not folded
    d.whole(silence(2))
    assert seen["n"] is None and seen["folded"]
