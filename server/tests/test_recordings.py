"""Call recordings (docs/spec.md, 4.4): the extension's audio kept on disk whether or not live
transcription kept up, and a call's gaps filled from it afterwards."""

import io
import json
import time

import av
import numpy as np
import soxr

from scribe_server.audio import RATE
from scribe_server.diarize import DiarizerSlot
from scribe_server.recordings import Recordings

from .conftest import Harness, silence, tone, wav
from .test_diarize import FakeDiarizer, pitch

T0 = 1_800_000_000.0  # a call that began at this wall-clock time


def webm(audio: np.ndarray) -> bytes:
    """Opus in WebM, as Chrome's MediaRecorder writes it."""
    pcm = (np.clip(soxr.resample(audio, RATE, 48000), -1, 1) * 32767).astype(np.int16)
    out = io.BytesIO()
    with av.open(out, "w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.layout, stream.bit_rate = "mono", 24000
        for i in range(0, len(pcm) - 959, 960):
            frame = av.AudioFrame.from_ndarray(pcm[i : i + 960][None, :], format="s16", layout="mono")
            frame.sample_rate, frame.pts = 48000, i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return out.getvalue()


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0


def put(client, sid, channel, index, audio, at, mime="audio/webm", encode=webm, **params):
    return client.put(
        f"/api/sessions/{sid}/recording/{channel}/{index}",
        params={"started_at": at, "duration_ms": int(len(audio) / RATE * 1000), **params},
        content=encode(audio) if encode else audio,
        headers={"Content-Type": mime},
    )


def new_call(client, source="browser") -> str:
    return client.post("/api/sessions", json={"source": source, "title": "Meet"}).json()["id"]


def marker(client, sid, speaker_id, name, at, seconds, text=None):
    return client.post(
        f"/api/sessions/{sid}/markers",
        json={
            "speaker_id": speaker_id, "speaker_name": name, "started_at": at, "duration_ms": int(seconds * 1000),
            "text": text or f"[unavailable: {round(seconds)} s of audio from {name}]",
        },
    ).json()


def wait(client, job_id, timeout=15):
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/api/transcriptions/{job_id}").json()
        if job["status"] not in ("queued", "fetching", "decoding", "transcribing"):
            return job
        assert time.monotonic() < deadline, job
        time.sleep(0.02)


def lines_of(client, sid):
    return client.get(f"/api/sessions/{sid}").json()["utterances"]


# -- storing -----------------------------------------------------------------------------------


def test_a_segment_is_stored_once_however_often_it_is_sent(harness, tmp_path):
    c = harness.client
    sid = new_call(c)
    audio = tone(5)
    first = put(c, sid, "tab", 0, audio, T0)
    assert first.status_code == 201 and first.json()["created"] is True
    path = tmp_path / "recordings" / sid / "tab" / "00000.webm"
    stored = path.read_bytes()
    assert stored[:4] == b"\x1aE\xdf\xa3"  # a WebM file, as sent

    again = put(c, sid, "tab", 0, tone(5, ) * 0.5, T0 + 99)  # a retry (even with other bytes) changes nothing
    assert again.status_code == 200 and again.json()["created"] is False
    assert path.read_bytes() == stored
    chunks = c.get(f"/api/sessions/{sid}/recording").json()["channels"]["tab"]
    assert chunks["chunks"] == 1 and chunks["started_at"] == T0


def test_an_upload_the_manifest_never_heard_of_is_registered_by_the_retry(harness, tmp_path):
    """A crash after the file was written but before the manifest was: the retry must not be
    skipped as 'already stored'."""
    c = harness.client
    sid = new_call(c)
    folder = tmp_path / "recordings" / sid / "mic"
    folder.mkdir(parents=True)
    (folder / "00000.webm").write_bytes(b"left over from a crash")
    assert put(c, sid, "mic", 0, tone(5), T0).status_code == 201
    assert (folder / "00000.webm").read_bytes()[:4] == b"\x1aE\xdf\xa3"
    assert c.get(f"/api/sessions/{sid}/recording").json()["channels"]["mic"]["chunks"] == 1


def test_bad_uploads_are_refused(harness):
    c = harness.client
    sid = new_call(c)
    assert put(c, sid, "speaker", 0, tone(1), T0).status_code == 400
    assert put(c, sid, "tab", 100_000, tone(1), T0).status_code == 400
    assert c.put(f"/api/sessions/{sid}/recording/tab/0", params={"started_at": T0}, content=b"").status_code == 400
    assert put(c, "000000000000", "tab", 0, tone(1), T0).status_code == 404
    assert c.put(f"/api/sessions/{sid}/recording/tab/0", content=b"x").status_code == 422  # no started_at


def test_manifest_and_finalise(harness, tmp_path):
    c = harness.client
    sid = new_call(c)
    for index in (0, 1, 3):  # 2 never arrived
        assert put(c, sid, "tab", index, tone(30.4), T0 + index * 30, sample_rate=48000).status_code == 201
    put(c, sid, "mic", 0, tone(30.4), T0)
    recording = c.get(f"/api/sessions/{sid}/recording").json()
    assert recording["finalised"] is False
    assert recording["channels"]["tab"]["missing"] == [2]
    assert recording["channels"]["mic"]["missing"] == []

    c.post(f"/api/sessions/{sid}/end")
    # Segments still arriving after the end are accepted, and the summary follows them.
    assert put(c, sid, "tab", 2, tone(30.4), T0 + 60).status_code == 201
    recording = c.get(f"/api/sessions/{sid}/recording").json()
    assert recording["finalised"] is True
    assert recording["channels"]["tab"]["missing"] == [] and recording["channels"]["tab"]["chunks"] == 4
    assert recording["channels"]["tab"]["ended_at"] == T0 + 90 + 30.4
    assert recording["bytes"] == sum(f.stat().st_size for f in (tmp_path / "recordings" / sid).rglob("*.webm"))

    manifest = Recordings(tmp_path / "recordings").manifest(sid)
    assert manifest["session_id"] == sid and manifest["version"] == 1
    assert manifest["channels"]["tab"]["sample_rate"] == 48000
    chunk = manifest["channels"]["tab"]["chunks"]["3"]
    assert chunk == {"file": "tab/00003.webm", "started_at": T0 + 90, "duration_ms": 30400, "bytes": chunk["bytes"]}
    assert chunk["bytes"] == (tmp_path / "recordings" / sid / "tab" / "00003.webm").stat().st_size
    assert c.get(f"/api/sessions/{sid}").json()["recording"]["gaps"] == 0


def test_a_session_with_no_recording_says_so(harness):
    c = harness.client
    sid = new_call(c)
    assert c.get(f"/api/sessions/{sid}/recording").status_code == 404
    assert c.get(f"/api/sessions/{sid}").json()["recording"] is None


def test_recording_can_be_switched_off(tmp_path):
    h = Harness(tmp_path)
    h.services.recordings = None
    try:
        sid = new_call(h.client)
        assert put(h.client, sid, "tab", 0, tone(1), T0).status_code == 403
        assert h.client.get("/api/status").json()["recording"] is False
    finally:
        h.worker.stop()


def test_the_browser_opt_out_stops_recording(harness):
    c = harness.client
    sid = new_call(c)
    assert c.get("/api/status").json()["recording"] is True
    c.post("/api/optouts", json={"source": "browser", "user_id": "recording", "opted_out": True})
    assert put(c, sid, "tab", 0, tone(1), T0).status_code == 403
    assert c.get("/api/status").json()["recording"] is False
    c.post("/api/optouts", json={"source": "browser", "user_id": "recording", "opted_out": False})
    assert put(c, sid, "tab", 0, tone(1), T0).status_code == 201


def test_deleting_a_session_deletes_its_audio(harness, tmp_path):
    c = harness.client
    sid = new_call(c)
    put(c, sid, "tab", 0, tone(2), T0)
    assert (tmp_path / "recordings" / sid).is_dir()
    c.delete(f"/api/sessions/{sid}")
    assert not (tmp_path / "recordings" / sid).exists()


def test_storing_needs_no_model_and_no_queue(harness):
    """The reason recording exists: a server whose transcription queue is full (503 to every live
    utterance) still keeps the audio."""
    import threading

    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    gate = threading.Event()
    try:
        harness.worker.submit(lambda: gate.wait(10))
        while harness.worker.depth:  # it is running, not waiting
            time.sleep(0.01)
        for _ in range(harness.worker._queue.maxsize):  # and the queue is full behind it
            harness.worker.submit(lambda: gate.wait(10))
        busy = live_line_raw(c, sid, "mic", "Alice", T0, tone(2), "mic:0")
        assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        assert put(c, sid, "mic", 0, tone(2), T0).status_code == 201
        assert put(c, sid, "tab", 0, tone(2), T0).status_code == 201
    finally:
        gate.set()


# -- reading back --------------------------------------------------------------------------------


def test_segments_are_placed_by_time_and_a_missing_one_is_silence(harness):
    c = harness.client
    sid = new_call(c)
    put(c, sid, "mic", 0, tone(10), T0)  # 0 to 10 s
    put(c, sid, "mic", 2, tone(10), T0 + 20)  # 20 to 30 s: 10 to 20 s never arrived
    audio = c.get(f"/api/sessions/{sid}/recording/mic/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert float(audio.headers["x-recording-started-at"]) == T0
    import soundfile

    samples, rate = soundfile.read(io.BytesIO(audio.content), dtype="float32")
    assert rate == RATE and abs(len(samples) / RATE - 30) < 0.1
    assert rms(samples[2 * RATE : 8 * RATE]) > 0.15
    assert rms(samples[11 * RATE : 19 * RATE]) < 0.001
    assert rms(samples[22 * RATE : 28 * RATE]) > 0.15
    assert c.get(f"/api/sessions/{sid}/recording/mic/audio", params={"start": T0, "end": T0 + 1e9}).status_code == 400
    assert c.get(f"/api/sessions/{sid}/recording/speaker/audio").status_code == 404
    got = c.get(f"/api/sessions/{sid}/recording/mic/audio", params={"start": T0 + 8, "end": T0 + 12}).content
    cut, _ = soundfile.read(io.BytesIO(got), dtype="float32")
    assert abs(len(cut) / RATE - 4) < 0.01 and rms(cut[: 2 * RATE]) > 0.15 and rms(cut[2 * RATE + 400 :]) < 0.001


def test_overlapping_segments_do_not_double_or_drop_audio(harness):
    """The extension overlaps neighbouring segments so nothing falls between them."""
    c = harness.client
    sid = new_call(c)
    put(c, sid, "tab", 0, tone(30.4), T0)
    put(c, sid, "tab", 1, tone(30.4), T0 + 30)
    import soundfile

    samples, _ = soundfile.read(io.BytesIO(c.get(f"/api/sessions/{sid}/recording/tab/audio").content), dtype="float32")
    assert abs(len(samples) / RATE - 60.4) < 0.1
    assert rms(samples[29 * RATE : 31 * RATE]) > 0.15  # across the seam


# -- filling a call's gaps -----------------------------------------------------------------------


def live_line_raw(client, sid, speaker, name, at, audio, key):
    return client.post(
        f"/api/sessions/{sid}/utterances",
        data={"speaker_id": speaker, "speaker_name": name, "started_at": str(at), "client_key": key},
        files={"file": ("u.wav", wav(audio), "audio/wav")},
    )


def live_line(client, sid, *args):
    return live_line_raw(client, sid, *args).json()


def test_a_gap_is_replaced_with_what_the_recording_says(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    # The call: Alice spoke at 5 s (heard live), at 20 s (lost), at 40 s (heard live).
    live_line(c, sid, "mic", "Alice", T0 + 5, tone(3), "mic:5")
    live_line(c, sid, "mic", "Alice", T0 + 40, tone(2), "mic:40")
    marker(c, sid, "mic", "Alice", T0 + 20, 4)
    # The recording has all of it: speech at 5, 20 and 40 s, in two segments.
    track = np.concatenate([silence(5), tone(3), silence(12), tone(4), silence(16), tone(2), silence(3)])
    put(c, sid, "mic", 0, track[: 30 * RATE], T0)
    put(c, sid, "mic", 1, track[30 * RATE :], T0 + 30)
    c.post(f"/api/sessions/{sid}/end")
    assert c.get(f"/api/sessions/{sid}").json()["recording"]["gaps"] == 1

    with c:
        started = c.post(f"/api/sessions/{sid}/reprocess")
        assert started.status_code == 202 and started.json()["kind"] == "reprocess"
        job = wait(c, started.json()["id"])
    assert job["status"] == "done", job
    result = job["result"]
    assert (result["gaps"], result["filled"], result["unrecoverable"], result["failed"]) == (1, 1, 0, 0)
    assert (result["unheard"], result["lines"], result["remaining"]) == (0, 1, 0)  # the rest was already in the transcript

    lines = lines_of(c, sid)
    assert [(u["speaker_id"], u["speaker_name"]) for u in lines] == [("mic", "Alice")] * 3
    assert [round(u["started_at"] - T0) for u in lines] == [5, 20, 40]  # in order, the new one between
    assert lines[1]["text"].startswith("alpha: 4.")  # transcribed from the 4 s of audio
    export = c.get(f"/api/sessions/{sid}/export").text
    assert "[unavailable" not in export and export.count("alpha:") == 3
    assert c.get(f"/api/sessions/{sid}").json()["recording"]["gaps"] == 0

    # Doing it again changes nothing.
    with c:
        again = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert again["result"]["gaps"] == 0 and len(lines_of(c, sid)) == 3


def test_a_gap_in_a_split_tab_gets_its_speakers_back(tmp_path):
    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(FakeDiarizer())
    h.ready()
    c = h.client
    try:
        sid = new_call(c)
        form = lambda key, at: {"speaker_id": "tab", "speaker_name": "Others", "started_at": str(at), "client_key": key, "diarize": "true"}  # noqa: E731
        first = np.concatenate([pitch(300, 2), silence(0.8), pitch(700, 2), silence(0.5)])
        c.post(f"/api/sessions/{sid}/utterances", data=form("tab:1", T0 + 2), files={"file": ("u.wav", wav(first))})
        c.post(f"/api/sessions/{sid}/speakers/tab%232", json={"name": "Alice"})
        # Lost: Alice and then the first speaker again.
        lost = np.concatenate([pitch(700, 1.5), silence(0.8), pitch(300, 2.5), silence(0.4)])
        marker(c, sid, "tab", "Others", T0 + 20, 5.2)
        track = np.concatenate([silence(2), first, silence(14), lost, silence(5)])
        put(c, sid, "tab", 0, track, T0)
        c.post(f"/api/sessions/{sid}/end")
        with c:
            job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
        assert job["status"] == "done" and job["result"]["filled"] == 1, job
        lines = lines_of(c, sid)
        assert [(u["speaker_id"], u["speaker_name"]) for u in lines] == [
            ("tab#1", "Speaker 1"), ("tab#2", "Alice"), ("tab#2", "Alice"), ("tab#1", "Speaker 1"),
        ]
        assert lines[2]["started_at"] < lines[3]["started_at"] and lines[1]["started_at"] < lines[2]["started_at"]
    finally:
        h.worker.stop()


def test_a_marker_for_an_utterance_the_server_did_serve_after_all_is_just_removed(harness):
    """The extension gave up on a request that the server then finished: the real line is kept and
    the marker goes, so the utterance is not in the transcript twice."""
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    live_line(c, sid, "mic", "Alice", T0 + 10, tone(3), f"mic:{T0 + 10}")
    c.post(f"/api/sessions/{sid}/markers", json={
        "speaker_id": "mic", "speaker_name": "Alice", "started_at": T0 + 10, "duration_ms": 3000, "client_key": f"mic:{T0 + 10}",
        "text": "[unavailable: 3 s of audio from Alice]",
    })
    put(c, sid, "mic", 0, np.concatenate([silence(10), tone(3), silence(7)]), T0)
    assert len(lines_of(c, sid)) == 2
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["filled"] == 1 and job["result"]["lines"] == 0
    assert [u["client_key"] for u in lines_of(c, sid)] == [f"mic:{T0 + 10}"]


def test_a_gap_with_no_audio_behind_it_is_left_alone(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    marker(c, sid, "mic", "Alice", T0 + 100, 4)  # recorded only to 30 s
    live_line(c, sid, "mic", "Alice", T0, tone(25), "mic:0")
    put(c, sid, "mic", 0, np.concatenate([tone(25), silence(5)]), T0)
    c.post(f"/api/sessions/{sid}/end")
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["unrecoverable"] == 1 and job["result"]["remaining"] == 1
    assert [u["text"] for u in lines_of(c, sid)] == ["alpha: 25.0s", "[unavailable: 4 s of audio from Alice]"]


def test_speech_the_transcript_never_heard_of_is_found_in_the_recording(harness):
    """The server was down for a while: no line, and no marker either, since the marker could not
    be posted. The recording still has the speech, and gap filling finds it."""
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    live_line(c, sid, "mic", "Alice", T0 + 2, tone(3), "mic:2")
    live_line(c, sid, "mic", "Alice", T0 + 60, tone(3), "mic:60")
    # Speech at 2 s and 60 s were heard live; the stretch at 20 s and the one at 40 s were not.
    track = np.concatenate([silence(2), tone(3), silence(15), tone(4), silence(16), tone(2), silence(18), tone(3), silence(2)])
    put(c, sid, "mic", 0, track[: 30 * RATE], T0)
    put(c, sid, "mic", 1, track[30 * RATE :], T0 + 30)
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["gaps"] == 0 and job["result"]["unheard"] == 2 and job["result"]["lines"] == 2
    lines = lines_of(c, sid)
    assert [round(u["started_at"] - T0) for u in lines] == [2, 20, 40, 60]
    assert {u["speaker_id"] for u in lines} == {"mic"} and {u["speaker_name"] for u in lines} == {"Alice"}
    # Nothing is left to find, so another pass adds nothing.
    with c:
        again = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert again["result"]["unheard"] == 0 and len(lines_of(c, sid)) == 4


def test_an_idle_server_goes_through_a_call_again_only_when_more_audio_has_arrived(harness, tmp_path):
    harness.ready("alpha")
    c, app = harness.client, harness.app
    sid = new_call(c)
    put(c, sid, "mic", 0, np.concatenate([silence(2), tone(3), silence(2)]), T0)
    c.post(f"/api/sessions/{sid}/end")
    harness.services.store._exec("update sessions set ended_at = ended_at - 120 where id = ?", sid)
    manifest = lambda: json.loads((tmp_path / "recordings" / sid / "manifest.json").read_text())  # noqa: E731

    def pass_over():
        """Let an idle server decide whether to go through the recording, and wait for it."""
        age_last_chunk(tmp_path, sid, 400)
        data = manifest()
        data["auto_last"] = 0
        (tmp_path / "recordings" / sid / "manifest.json").write_text(json.dumps(data))
        c.portal.call(app.state.auto_reprocess)
        deadline = time.monotonic() + 10
        while (t := c.get("/api/status").json()["transcriptions"])["running"] + t["queued"]:
            assert time.monotonic() < deadline
            time.sleep(0.02)

    with c:
        c.post(f"/api/sessions/{sid}/end")
        pass_over()
        assert len(lines_of(c, sid)) == 1 and manifest()["auto_attempts"] == 1 and manifest()["reprocess_complete"] is True
        pass_over()  # nothing new: it does not go through the same audio again
        assert manifest()["auto_attempts"] == 1
        put(c, sid, "mic", 1, np.concatenate([tone(3), silence(2)]), T0 + 30)  # the extension caught up
        assert manifest()["reprocess_complete"] is False
        pass_over()
        assert manifest()["auto_attempts"] == 2 and len(lines_of(c, sid)) == 2


def test_an_old_marker_without_a_duration_is_found_by_its_text(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    c.post(f"/api/sessions/{sid}/markers", json={
        "speaker_id": "mic", "speaker_name": "Alice", "started_at": T0 + 10, "text": "[unavailable: 3 s of audio from Alice]",
    })
    put(c, sid, "mic", 0, np.concatenate([silence(10), tone(3), silence(7)]), T0)
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["filled"] == 1
    assert lines_of(c, sid)[0]["text"].startswith("alpha:")


def test_reprocess_needs_a_recording_a_model_and_one_at_a_time(harness):
    c = harness.client
    sid = new_call(c)
    assert c.post(f"/api/sessions/{sid}/reprocess").status_code == 404  # nothing recorded
    put(c, sid, "mic", 0, tone(3), T0)
    assert c.post(f"/api/sessions/{sid}/reprocess").status_code == 503  # no model
    assert c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "everything"}).status_code == 422
    assert c.post("/api/sessions/000000000000/reprocess").status_code == 404


def test_the_whole_recording_can_be_transcribed_again(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    live_line(c, sid, "mic", "Alice", T0 + 5, tone(3), "mic:5")
    marker(c, sid, "mic", "Alice", T0 + 30, 4)
    track = np.concatenate([silence(5), tone(3), silence(22), tone(4), silence(5)])
    put(c, sid, "mic", 0, track, T0)
    c.post(f"/api/sessions/{sid}/speakers/mic", json={"name": "Alice Example"})  # a name given in the GUI
    c.post(f"/api/sessions/{sid}/end")
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "all"}).json()["id"])
    assert job["status"] == "done" and job["result"]["remaining"] == 0, job
    lines = lines_of(c, sid)
    assert [round(u["started_at"] - T0) for u in lines] == [5, 30]
    assert {u["speaker_name"] for u in lines} == {"Alice Example"}
    assert all(u["client_key"].startswith("re") for u in lines)  # all new; the old ones are gone


def test_a_failed_reprocess_all_leaves_the_transcript_as_it_was(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    live_line(c, sid, "mic", "Alice", T0 + 5, tone(3), "mic:5")
    put(c, sid, "mic", 0, np.concatenate([silence(5), tone(3), silence(30), tone(3), silence(2)]), T0)  # two utterances
    c.post(f"/api/sessions/{sid}/end")
    before = lines_of(c, sid)

    calls = {"n": 0}
    engine = harness.loader.engines["alpha"]
    original = engine.transcribe

    def breaks_on_the_second(audio):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the model fell over")
        return original(audio)

    engine.transcribe = breaks_on_the_second
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "all"}).json()["id"])
    assert job["status"] == "error"
    assert lines_of(c, sid) == before


# -- on its own, when idle ---------------------------------------------------------------------


def age_last_chunk(tmp_path, sid, seconds):
    path = tmp_path / "recordings" / sid / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["last_chunk_at"] -= seconds
    path.write_text(json.dumps(manifest))


def test_an_ended_call_with_gaps_is_filled_when_the_server_is_idle(harness, tmp_path):
    harness.ready("alpha")
    c = harness.client
    app = harness.app
    sid = new_call(c)
    marker(c, sid, "mic", "Alice", T0 + 10, 3)
    put(c, sid, "mic", 0, np.concatenate([silence(10), tone(3), silence(7)]), T0)

    with c:
        c.portal.call(app.state.auto_reprocess)  # the call is still open: nothing happens
        assert lines_of(c, sid)[0]["text"].startswith("[unavailable")
        c.post(f"/api/sessions/{sid}/end")
        harness.services.store._exec("update sessions set ended_at = ended_at - 120 where id = ?", sid)
        age_last_chunk(tmp_path, sid, 120)
        c.portal.call(app.state.auto_reprocess)
        deadline = time.monotonic() + 10
        while lines_of(c, sid)[0]["text"].startswith("[unavailable"):
            assert time.monotonic() < deadline
            time.sleep(0.02)
    assert lines_of(c, sid)[0]["text"].startswith("alpha:")


# -- retention ---------------------------------------------------------------------------------


def test_retention(tmp_path):
    root = tmp_path / "recordings"
    rec = Recordings(root, keep_days=14, after_reprocess_days=3, max_bytes=10_000)
    ids = {name: f"{i:012x}" for i, name in enumerate(["open", "old", "fresh", "done", "gone", "big1", "big2"])}
    for name, sid in ids.items():
        rec.put(sid, "tab", 0, T0, b"x" * (6_000 if name.startswith("big") else 100), mime="audio/webm")
    day, now = 86400, T0 + 100 * 86400
    rec.note_reprocess(ids["done"], complete=True, now=now - 4 * day)
    ended = {
        ids["open"]: None, ids["old"]: now - 15 * day, ids["fresh"]: now - 1 * day, ids["done"]: now - 5 * day,
        ids["big1"]: now - 2 * day, ids["big2"]: now - 1 * day,
        # ids["gone"] is not a session any more
    }
    deleted = set(rec.purge(ended, now))
    # Past 14 days; reprocessed clean more than 3 days ago; its session deleted; and the oldest
    # of the two big ones, which together are over the cap.
    assert deleted == {ids["old"], ids["done"], ids["gone"], ids["big1"]}
    assert set(rec.sessions()) == {ids["open"], ids["fresh"], ids["big2"]}


def test_a_reprocess_that_leaves_gaps_does_not_shorten_retention(tmp_path):
    rec = Recordings(tmp_path / "r", after_reprocess_days=3)
    sid = "0" * 12
    rec.put(sid, "tab", 0, T0, b"x")
    rec.note_reprocess(sid, complete=False, now=T0)
    assert rec.purge({sid: T0 + 1}, T0 + 10 * 86400) == []


# -- review findings: what a reprocess must never do -------------------------------------------


def test_transcribing_all_again_refuses_to_lose_lines_the_recording_does_not_cover(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    live_line(c, sid, "mic", "Alice", T0 + 5, tone(3), "mic:5")
    live_line(c, sid, "mic", "Alice", T0 + 100, tone(3), "mic:100")  # the recording stops at 30 s
    put(c, sid, "mic", 0, np.concatenate([silence(5), tone(3), silence(22)]), T0)
    before = lines_of(c, sid)
    assert c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "all"}).status_code == 409  # the call is open
    c.post(f"/api/sessions/{sid}/end")
    refused = c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "all"})
    assert refused.status_code == 409 and "does not cover 1 line" in refused.json()["detail"]
    assert lines_of(c, sid) == before
    # Filling gaps is still on offer.
    with c:
        assert wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])["status"] == "done"
    assert [round(u["started_at"] - T0) for u in lines_of(c, sid)] == [5, 100]


def test_transcribing_all_again_keeps_a_marker_the_recording_cannot_answer(harness):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    live_line(c, sid, "mic", "Alice", T0 + 5, tone(3), "mic:5")
    marker(c, sid, "mic", "Alice", T0 + 100, 4)  # nothing recorded there
    put(c, sid, "mic", 0, np.concatenate([silence(5), tone(3), silence(22)]), T0)
    c.post(f"/api/sessions/{sid}/end")
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "all"}).json()["id"])
    assert job["status"] == "done" and job["result"]["remaining"] == 1 and job["result"]["complete"] is False
    assert [u["text"].startswith("[unavailable") for u in lines_of(c, sid)] == [False, True]


def test_a_gap_whose_fill_failed_is_not_found_again_as_unheard_speech(harness):
    """One pass fails on a marker; the next must fill that marker once, not leave two lines."""
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    marker(c, sid, "mic", "Alice", T0 + 10, 4)
    put(c, sid, "mic", 0, np.concatenate([silence(10), tone(4), silence(6)]), T0)
    engine = harness.loader.engines["alpha"]
    original, calls = engine.transcribe, {"n": 0}

    def fails_once(audio):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the model fell over")
        return original(audio)

    engine.transcribe = fails_once
    with c:
        first = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
        assert first["result"]["failed"] == 1 and first["result"]["complete"] is False
        assert [u["text"].startswith("[unavailable") for u in lines_of(c, sid)] == [True]  # nothing doubled
        second = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert second["result"]["filled"] == 1 and second["result"]["complete"] is True
    assert [u["text"] for u in lines_of(c, sid)] == ["alpha: 4.0s"]


def test_a_segment_that_cannot_be_decoded_leaves_the_marker_and_the_pass_incomplete(harness, tmp_path):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    marker(c, sid, "mic", "Alice", T0 + 10, 4)
    assert put(c, sid, "mic", 0, b"this is not audio at all", T0, encode=None, duration_ms=30000).status_code == 201
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["unrecoverable"] == 1 and job["result"]["remaining"] == 1 and job["result"]["complete"] is False
    assert [u["text"].startswith("[unavailable") for u in lines_of(c, sid)] == [True]
    manifest = json.loads((tmp_path / "recordings" / sid / "manifest.json").read_text())
    assert manifest["reprocess_complete"] is False and manifest["scanned_bytes"] is None  # tried again later


def test_a_failed_search_for_unheard_speech_is_not_a_complete_pass(harness, tmp_path):
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    put(c, sid, "mic", 0, np.concatenate([silence(2), tone(3), silence(2)]), T0)  # no line, no marker
    engine = harness.loader.engines["alpha"]
    engine.transcribe = lambda audio: (_ for _ in ()).throw(RuntimeError("the model fell over"))
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["failed"] == 1 and job["result"]["complete"] is False
    manifest = json.loads((tmp_path / "recordings" / sid / "manifest.json").read_text())
    assert manifest["reprocess_complete"] is False and manifest["scanned_bytes"] is None


def test_a_fill_shares_the_key_of_the_utterance_it_stands_for(harness):
    """So that if the live request the client gave up on is served late, the index keeps one."""
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    key = f"mic:{T0 + 10}"
    c.post(f"/api/sessions/{sid}/markers", json={
        "speaker_id": "mic", "speaker_name": "Alice", "started_at": T0 + 10, "duration_ms": 3000, "client_key": key,
        "text": "[unavailable: 3 s of audio from Alice]",
    })
    put(c, sid, "mic", 0, np.concatenate([silence(10), tone(3), silence(7)]), T0)
    with c:
        wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert [u["client_key"] for u in lines_of(c, sid)] == [key]
    assert live_line_raw(c, sid, "mic", "Alice", T0 + 10, tone(3), key).json()["id"] == lines_of(c, sid)[0]["id"]
    assert len(lines_of(c, sid)) == 1  # the late live request found the line there


# -- what the existing tests did not pin -------------------------------------------------------


def test_where_two_segments_overlap_the_later_one_is_used(tmp_path):
    rec = Recordings(tmp_path / "r")
    sid = "1" * 12
    rec.put(sid, "tab", 0, T0, wav(tone(10) * 0.3), mime="audio/wav", duration_ms=10_000)  # 0 to 10 s, quiet
    rec.put(sid, "tab", 1, T0 + 8, wav(tone(10)), mime="audio/wav", duration_ms=10_000)  # 8 to 18 s, loud
    samples, origin, bad = rec.read(sid, "tab")
    assert origin == T0 and bad == []
    assert abs(rms(samples[4 * RATE : 6 * RATE]) - rms(tone(2) * 0.3)) < 0.01
    assert abs(rms(samples[9 * RATE : 10 * RATE]) - rms(tone(1))) < 0.01  # inside the overlap: the later one


def test_a_failed_reprocess_all_puts_the_speaker_names_back(tmp_path):
    h = Harness(tmp_path)
    h.services.diarization = DiarizerSlot(FakeDiarizer())
    h.ready()
    c = h.client
    try:
        sid = new_call(c)
        form = {"speaker_id": "tab", "speaker_name": "Others", "started_at": str(T0 + 2), "client_key": "tab:1", "diarize": "true"}
        first = np.concatenate([pitch(300, 2), silence(0.8), pitch(700, 2), silence(0.5)])
        c.post(f"/api/sessions/{sid}/utterances", data=form, files={"file": ("u.wav", wav(first))})
        c.post(f"/api/sessions/{sid}/speakers/tab%232", json={"name": "Alice"})
        put(c, sid, "tab", 0, np.concatenate([silence(2), first, silence(2)]), T0)
        c.post(f"/api/sessions/{sid}/end")
        before = (lines_of(c, sid), c.get(f"/api/sessions/{sid}").json()["speakers"])
        rows = h.services.store.speaker_rows(sid, "tab")
        engine = h.loader.engines["alpha"]
        engine.transcribe = lambda audio: (_ for _ in ()).throw(RuntimeError("the model fell over"))
        with c:
            job = wait(c, c.post(f"/api/sessions/{sid}/reprocess", json={"mode": "all"}).json()["id"])
        assert job["status"] == "error"
        assert (lines_of(c, sid), c.get(f"/api/sessions/{sid}").json()["speakers"]) == before
        assert before[1] == {"tab#2": "Alice"}
        assert h.services.store.speaker_rows(sid, "tab") == rows  # the voices came back too, not only the names
    finally:
        h.worker.stop()


def test_a_gap_an_earlier_attempt_already_filled_is_not_transcribed_again(harness):
    """An attempt stored the text and stopped before removing the marker: the next only removes it."""
    harness.ready("alpha")
    c = harness.client
    sid = new_call(c)
    made = marker(c, sid, "mic", "Alice", T0 + 10, 3)  # an older marker: no key of its own
    harness.services.store.add_utterance(sid, "mic", "Alice", T0 + 10, 3000, "alpha: 3.0s", "alpha", 5, f"fill:{made['id']}")
    put(c, sid, "mic", 0, np.concatenate([silence(10), tone(3), silence(7)]), T0)
    engine = harness.loader.engines["alpha"]
    before = engine.calls
    with c:
        job = wait(c, c.post(f"/api/sessions/{sid}/reprocess").json()["id"])
    assert job["result"]["filled"] == 1 and job["result"]["lines"] == 0
    assert engine.calls == before  # nothing was sent to the model
    assert [u["text"] for u in lines_of(c, sid)] == ["alpha: 3.0s"]
