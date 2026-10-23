"""The background transcription API: an uploaded recording or a pasted podcast link becomes a job,
and a finished job becomes a session in History."""

import time

import numpy as np

from scribe_server.diarize import DiarizerSlot

from .conftest import Harness, silence, tone, wav
from .podcast_fakes import FakeFetcher, feed
from .test_diarize import FakeDiarizer, pitch


def two_stretches() -> np.ndarray:
    # Two stretches of speech too far apart (over 25 s from start to end) to be one chunk.
    return np.concatenate([silence(1), tone(20), silence(2), tone(20), silence(1)])


def wait(client, job_id: str, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/api/transcriptions/{job_id}").json()
        if job["status"] not in ("queued", "fetching", "decoding", "transcribing"):
            return job
        assert time.monotonic() < deadline, job
        time.sleep(0.02)


def upload(client, audio=None, **params):
    return client.post(
        "/api/transcriptions", params=params, content=wav(two_stretches() if audio is None else audio),
        headers={"Content-Type": "application/octet-stream"},
    )


def test_an_upload_becomes_a_timestamped_session(harness):
    harness.ready("alpha")
    with harness.client as c:
        started = upload(c, filename="my talk.wav")
        assert started.status_code == 202
        job = wait(c, started.json()["id"])
        assert job["status"] == "done" and job["title"] == "my talk"
        result = job["result"]
        assert result["model"] == "alpha" and result["speakers"] == 1
        assert [(s["start"], s["end"]) for s in result["segments"]] == [(1.0, 21.0), (23.0, 43.0)]
        assert result["text"] == " ".join(s["text"] for s in result["segments"])
        # It is a session now: History, exports and renaming all work on it.
        session = c.get(f"/api/sessions/{job['session_id']}").json()
        assert session["source"] == "upload" and session["ended_at"] is not None
        assert [u["text"] for u in session["utterances"]] == [s["text"] for s in result["segments"]]
        srt = c.get(f"/api/sessions/{job['session_id']}/export", params={"format": "srt"}).text
        assert "00:00:01,000 --> 00:00:21,000" in srt and "00:00:23,000 --> 00:00:43,000" in srt


def test_speakers_are_separated_on_request(tmp_path):
    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(FakeDiarizer())
    h.ready()
    audio = np.concatenate([silence(0.5), pitch(300, 2), silence(1), pitch(500, 2), silence(0.5)])
    try:
        with h.client as c:
            job = wait(c, upload(c, audio, diarize="true").json()["id"])
            assert job["status"] == "done" and job["result"]["speakers"] == 2
            assert [s["speaker"] for s in job["result"]["segments"]] == ["Speaker 1", "Speaker 2"]
            session = c.get(f"/api/sessions/{job['session_id']}").json()
            assert {u["speaker_name"] for u in session["utterances"]} == {"Speaker 1", "Speaker 2"}
    finally:
        h.worker.stop()


def test_diarize_before_the_separator_is_ready_says_so(harness):
    harness.ready("alpha")
    with harness.client as c:
        job = wait(c, upload(c, diarize="true").json()["id"])
    assert job["status"] == "done" and "not ready" in job["result"]["note"]


def test_silence_is_a_finished_job_with_no_lines(harness):
    harness.ready("alpha")
    with harness.client as c:
        job = wait(c, upload(c, silence(4)).json()["id"])
        assert job["status"] == "done" and job["result"]["segments"] == [] and job["result"]["speakers"] == 0
    assert harness.loader.engines["alpha"].calls == 0


def test_bad_audio_is_an_error_not_a_session(harness):
    harness.ready("alpha")
    with harness.client as c:
        job = wait(c, c.post("/api/transcriptions", content=b"not audio at all").json()["id"])
        assert job["status"] == "error" and "cannot decode" in job["error"]
        assert c.get("/api/sessions").json() == []


def test_up_front_refusals(harness):
    with harness.client as c:
        assert upload(c).status_code == 503  # no model active
        harness.ready("alpha")
        assert c.post("/api/transcriptions", content=b"").status_code == 400
        harness.services.max_upload_bytes = 100
        assert upload(c).status_code == 413
        assert c.get("/api/transcriptions/nope").status_code == 404
    assert harness.anonymous.post("/api/transcriptions", content=b"x").status_code == 401


def test_a_podcast_link_is_fetched_and_transcribed(harness):
    harness.ready("alpha")
    audio = wav(two_stretches())
    harness.services.fetcher = FakeFetcher({
        "https://feeds.example.com/s.xml": feed("The Show", [("Ep 1: Hello", "Wed, 04 Mar 2026 08:30:00 GMT", "0:10", "https://cdn.example.com/1.wav")]),
        "https://cdn.example.com/1.wav": (audio, "audio/wav"),
    })
    with harness.client as c:
        started = c.post("/api/transcriptions/link", json={"url": "https://feeds.example.com/s.xml"})
        assert started.status_code == 202
        job = wait(c, started.json()["id"])
        assert job["status"] == "done", job
        assert job["title"] == "The Show: Ep 1: Hello"
        assert job["episode"]["audio_url"] == "https://cdn.example.com/1.wav" and job["episode"]["show"] == "The Show"
        assert len(job["result"]["segments"]) == 2
        assert c.get(f"/api/sessions/{job['session_id']}").json()["title"] == "The Show: Ep 1: Hello"


def test_a_link_that_cannot_be_resolved_fails_with_the_reason(harness):
    harness.ready("alpha")
    harness.services.fetcher = FakeFetcher({})
    with harness.client as c:
        job = wait(c, c.post("/api/transcriptions/link", json={"url": "https://example.com/page"}).json()["id"])
    assert job["status"] == "error" and "404" in job["error"]


def test_the_server_will_not_fetch_private_addresses_for_a_link(harness):
    harness.ready("alpha")
    with harness.client as c:
        job = wait(c, c.post("/api/transcriptions/link", json={"url": "http://127.0.0.1:8787/api/status"}).json()["id"])
    assert job["status"] == "error" and "public address" in job["error"]


def test_a_running_job_keeps_the_host_awake_and_can_be_cancelled(tmp_path):
    h = Harness(tmp_path)
    h.services.open_sessions_file = tmp_path / "open"
    h.ready("alpha")
    h.loader.engines["alpha"].delay = 0.3
    try:
        with h.client as c:
            job_id = upload(c).json()["id"]
            assert (tmp_path / "open").read_text().strip() == "1"
            c.delete(f"/api/transcriptions/{job_id}")
            job = wait(c, job_id)
            assert job["status"] == "cancelled"
            assert (tmp_path / "open").read_text().strip() == "0"
            assert c.get("/api/sessions").json() == []
    finally:
        h.worker.stop()


def test_a_busy_queue_makes_a_job_wait_not_fail(tmp_path):
    h = Harness(tmp_path)
    h.ready("alpha")
    h.loader.engines["alpha"].delay = 0.05
    audio = np.concatenate([silence(0.5)] + [np.concatenate([tone(20), silence(1)]) for _ in range(12)])  # 12 chunks > queue depth 4
    try:
        with h.client as c:
            job = wait(c, upload(c, audio).json()["id"], timeout=30)
            assert job["status"] == "done" and len(job["result"]["segments"]) == 12
    finally:
        h.worker.stop()
