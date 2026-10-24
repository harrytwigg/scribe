"""The background transcription API: an uploaded recording or a pasted podcast link becomes a job,
and a finished job becomes a session in History."""

import tempfile
import time

import numpy as np

from scribe_server.diarize import DiarizerSlot

from .conftest import Harness, silence, tone, wav
from .podcast_fakes import FakeFetcher, FakeReply, feed
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


def test_jobs_beyond_max_jobs_wait_their_turn_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    h = Harness(tmp_path / "h")
    h.services.max_jobs = 1
    h.services.max_queued = 2
    h.services.open_sessions_file = tmp_path / "open"
    h.ready("alpha")
    h.loader.engines["alpha"].delay = 0.2
    try:
        with h.client as c:
            a, b, d = (upload(c, title=name).json()["id"] for name in ("A", "B", "D"))
            assert [c.get(f"/api/transcriptions/{i}").json()["queue_position"] for i in (a, b, d)] == [None, 1, 2]
            assert [c.get(f"/api/transcriptions/{i}").json()["status"] for i in (b, d)] == ["queued", "queued"]
            assert c.get("/api/status").json()["transcriptions"] == {"running": 1, "queued": 2, "max": 1}
            assert (tmp_path / "open").read_text().strip() == "3"  # the queue keeps the host awake
            # the queue is bounded; a refused upload leaves no job or file behind
            full = upload(c)
            assert full.status_code == 429 and full.headers["Retry-After"] == "60"
            # cancelling a waiting job moves the ones behind it up
            c.delete(f"/api/transcriptions/{b}")
            assert c.get(f"/api/transcriptions/{b}").json()["status"] == "cancelled"
            assert c.get(f"/api/transcriptions/{d}").json()["queue_position"] == 1
            assert wait(c, a)["status"] == "done" and wait(c, d)["status"] == "done"
            # D ran after A (one at a time): its transcript was stored later
            sessions = {s["title"]: s for s in c.get("/api/sessions").json()}
            assert sessions["A"]["started_at"] < sessions["D"]["started_at"] and "B" not in sessions
            # a body over the limit, streamed without a length, is refused and frees its slot
            h.services.max_upload_bytes = 100
            assert c.post("/api/transcriptions", content=iter([b"x" * 60, b"x" * 60])).status_code == 413
            h.services.max_upload_bytes = 512 * 1024 * 1024
            assert (tmp_path / "open").read_text().strip() == "0"
            assert wait(c, upload(c).json()["id"])["status"] == "done"
        assert list(tmp_path.glob("scribe-job-*")) == [], "the audio is not kept"
    finally:
        h.worker.stop()


def test_one_progress_bar_covers_the_download_and_the_transcription(harness):
    harness.ready("alpha")
    audio = wav(two_stretches())

    class SlowReply(FakeReply):
        def read(self, n=-1):
            time.sleep(0.03)
            return super().read(min(n, 60_000) if n > 0 else n)

    class SlowFetcher(FakeFetcher):
        def request(self, url, headers=None):
            body, content_type, *_ = self.pages[url]
            return SlowReply(url, body, content_type)

    harness.services.fetcher = SlowFetcher({"https://cdn.example.com/talk.wav": (audio, "audio/wav")})
    harness.loader.engines["alpha"].delay = 0.1
    seen = []
    with harness.client as c:
        job_id = c.post("/api/transcriptions/link", json={"url": "https://cdn.example.com/talk.wav"}).json()["id"]
        while True:
            job = c.get(f"/api/transcriptions/{job_id}").json()
            seen.append(job)
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.02)
    assert job["status"] == "done", job
    progress = [j["progress"] for j in seen]
    assert progress == sorted(progress) and progress[-1] == 1.0, "one bar that only moves forward"
    downloading = [j for j in seen if j["status"] == "fetching" and j["download"]["done"]]
    assert downloading and all(j["download"]["total"] == len(audio) for j in downloading)
    assert 0 < max(j["progress"] for j in downloading) <= 0.15, "the download is the first stretch of the bar"
    assert {j["status"] for j in seen} >= {"fetching", "transcribing", "done"}
    transcribing = [j["progress"] for j in seen if j["status"] == "transcribing"]
    assert transcribing and min(transcribing) >= 0.2 and max(transcribing) <= 1.0  # 1.0 while the transcript is being stored


def test_an_uploaded_files_bar_starts_at_decoding_and_has_no_download(harness):
    harness.ready("alpha")
    with harness.client as c:
        first = upload(c).json()
        assert first["download"] is None and first["progress"] <= 0.05
        job = wait(c, first["id"])
        assert job["progress"] == 1.0 and job["stage_progress"] == 1.0


def test_separating_speakers_says_what_the_job_is_doing(tmp_path):
    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(FakeDiarizer())
    h.ready()
    audio = np.concatenate([silence(0.5), pitch(300, 2), silence(1), pitch(500, 2), silence(0.5)])
    try:
        with h.client as c:
            job_id = upload(c, audio, diarize="true").json()["id"]
            steps = set()
            while (job := c.get(f"/api/transcriptions/{job_id}").json())["status"] not in ("done", "error"):
                steps.add(job["step"])
                time.sleep(0.01)
            assert job["status"] == "done"
            assert "separating speakers" in steps or job["step"] == "separating speakers"
    finally:
        h.worker.stop()


def test_cancelling_a_fetch_stops_the_download(harness):
    harness.ready("alpha")
    reads = []

    class SlowReply(FakeReply):
        def read(self, n=-1):
            time.sleep(0.02)
            reads.append(1)
            return super().read(min(n, 1000) if n > 0 else n)

    class SlowFetcher(FakeFetcher):
        def request(self, url, headers=None):
            body, content_type, *_ = self.pages[url]
            return SlowReply(url, body, content_type)

    harness.services.fetcher = SlowFetcher({"https://cdn.example.com/long.mp3": (b"x" * 200_000, "audio/mpeg")})
    with harness.client as c:
        job_id = c.post("/api/transcriptions/link", json={"url": "https://cdn.example.com/long.mp3"}).json()["id"]
        time.sleep(0.3)
        assert c.delete(f"/api/transcriptions/{job_id}").status_code == 200
        assert c.get(f"/api/transcriptions/{job_id}").json()["status"] == "cancelled"  # at once, not eventually
        time.sleep(0.2)
        settled = len(reads)
        time.sleep(0.5)
        assert len(reads) == settled and settled < 100, "the download thread must stop, not run on to the end"
        assert c.get("/api/sessions").json() == []


def test_a_known_speaker_count_is_passed_to_the_separator_and_implies_separating(tmp_path):
    counts = []

    class Counting(FakeDiarizer):
        def whole(self, audio, num_speakers=None, cancel=None):
            counts.append(num_speakers)
            return super().whole(audio, num_speakers)

    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(Counting())
    h.ready()
    audio = np.concatenate([silence(0.5), pitch(300, 2), silence(1), pitch(500, 2), silence(0.5)])
    try:
        with h.client as c:
            job = wait(c, upload(c, audio, speakers="2").json()["id"])  # no `diarize`: asking for a count is enough
            assert job["status"] == "done" and job["result"]["speakers"] == 2
            wait(c, upload(c, audio, diarize="true").json()["id"])
            assert counts == [2, None]
            assert upload(c, audio, speakers="0").status_code == 422
            assert upload(c, audio, speakers="21").status_code == 422
            assert c.post("/api/transcriptions/link", json={"url": "https://x.test/a.mp3", "speakers": 99}).status_code == 422
    finally:
        h.worker.stop()


def test_cancelling_a_job_stops_its_work_in_the_separator(tmp_path):
    import threading

    seen = {}

    class Slow(FakeDiarizer):
        def whole(self, audio, num_speakers=None, cancel=None):
            seen["started"] = True
            seen["stopped"] = cancel.wait(10)  # a real separator would be busy for minutes
            raise RuntimeError("cancelled")

    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(Slow())
    h.ready()
    audio = np.concatenate([silence(0.5), pitch(300, 2), silence(0.5)])
    try:
        with h.client as c:
            job_id = upload(c, audio, diarize="true").json()["id"]
            deadline = time.monotonic() + 5
            while not seen.get("started") and time.monotonic() < deadline:
                time.sleep(0.02)
            c.delete(f"/api/transcriptions/{job_id}")
            assert wait(c, job_id)["status"] == "cancelled"
            deadline = time.monotonic() + 5
            while "stopped" not in seen and time.monotonic() < deadline:
                time.sleep(0.02)
            assert seen["stopped"] is True  # the cancel event reached the separator
    finally:
        h.worker.stop()
