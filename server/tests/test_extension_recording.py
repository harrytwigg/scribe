"""A call recorded by the real extension code in a real Chromium, with the server failing it
(docs/spec.md, 4.4): the audio must reach the server's disk whole, and the gaps in the transcript
must be fillable from it.

Two ways to fail the live transcript, both ones that really happened or can:

- starved: ASR is slow, the queue fills, every utterance past it gets 503 (the 40-minute call that
  lost a third of its lines);
- down: the server is stopped in the middle of the call and started again.

The fake engine stands in for the model, so this needs no GPU; Chromium does the recording.

    uv run playwright install chromium   # once
    uv run pytest -m extension tests/test_extension_recording.py -s
"""

import base64
import contextlib
import io
import socket
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile
import uvicorn

from scribe_server.audio import RATE
from scribe_server.auth import Auth
from scribe_server.config import EXTENSION_ORIGIN

from .conftest import Harness
from .test_extension import EXTENSION, EXTENSION_ID

pytestmark = pytest.mark.extension

CALL_S = 30  # how long the call lasts
SEGMENT_MS = 4000  # recorded in segments this long, so a 30 s call has several


def bursts(hz: float, on: float, off: float, seconds: float = 60) -> bytes:
    """A WAV of tone bursts with silence between: a stand-in for people taking turns."""
    t = np.arange(int(on * 16000)) / 16000
    cycle = np.concatenate([0.3 * np.sin(2 * np.pi * hz * t), np.zeros(int(off * 16000))])
    samples = np.tile(cycle, int(seconds / (on + off)) + 1)[: int(seconds * 16000)]
    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes((samples * 32767).astype("<i2").tobytes())
    return out.getvalue()


class Served:
    """The harness's app under uvicorn on a fixed port, which can be stopped and started again."""

    def __init__(self, harness: Harness):
        self.harness = harness
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = None

    def start(self) -> None:
        self.server = uvicorn.Server(uvicorn.Config(self.harness.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline
            time.sleep(0.01)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(10)


@pytest.fixture
def served(tmp_path):
    harness = Harness(tmp_path)
    harness.services.auth = Auth((EXTENSION_ORIGIN,))  # the extension's own origin is admitted
    harness.loader.delay = 0.0
    harness.ready("alpha")
    server = Served(harness)
    server.start()
    yield server
    if server.thread.is_alive():
        server.stop()
    harness.worker.stop()


@contextlib.contextmanager
def chromium(tmp_path):
    """Chromium with the extension loaded and a fake mic playing tone bursts."""
    from playwright.sync_api import sync_playwright

    mic_clip = tmp_path / "mic.wav"
    mic_clip.write_bytes(bursts(330, 0.9, 1.1))
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(tmp_path / "profile"), channel="chromium", headless=True,
            args=[
                f"--disable-extensions-except={EXTENSION}", f"--load-extension={EXTENSION}",
                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={mic_clip}", "--autoplay-policy=no-user-gesture-required",
            ],
        )
        worker = context.service_workers[0] if context.service_workers else context.wait_for_event("serviceworker")
        assert worker.url.startswith(f"chrome-extension://{EXTENSION_ID}/")
        try:
            yield context
        finally:
            context.close()


def offscreen(context):
    """The extension's offscreen document, loaded as a page, ready to start a call."""
    page = context.new_page()
    page.goto(f"chrome-extension://{EXTENSION_ID}/src/offscreen.html")
    page.locator("script[src='offscreen.js']").wait_for(state="attached")
    for _ in range(100):
        if page.evaluate("Boolean(globalThis.scribe)"):
            return page
        time.sleep(0.1)
    raise AssertionError("the offscreen document never loaded")


def begin(page, served, retry_ms=(200, 400)):
    """Start a call (the tab is a looping tone-burst clip, the mic Chromium's fake one); returns when."""
    config = {"server": served.url, "myName": "Alice", "othersName": "Meet", "useMic": True, "separateSpeakers": False, "record": True}
    began = time.time()
    page.evaluate(
        """async ([config, clip, retry, segmentMs]) => {
            const bytes = Uint8Array.from(atob(clip), (c) => c.charCodeAt(0))
            const audio = new Audio(URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })))
            audio.loop = true
            await audio.play()
            const tab = audio.captureStream()
            const mic = await navigator.mediaDevices.getUserMedia({ audio: true })
            await globalThis.scribe.start(config, 'Meet - test', { tab, mic }, {
              clientOptions: { retryDelaysMs: retry },
              recorderOptions: { segmentMs },
            })
        }""",
        [config, base64.b64encode(bursts(220, 1.2, 0.9)).decode(), list(retry_ms), SEGMENT_MS],
    )
    return began


def uploaded(page, timeout_s=50):
    """Wait for the document to have delivered every segment."""
    for _ in range(timeout_s * 4):
        if page.evaluate("globalThis.scribe.uploader.pending") == 0:
            return
        time.sleep(0.25)
    raise AssertionError("the recording never finished uploading")


def browser_session(served):
    import httpx

    return next(s for s in httpx.Client(base_url=served.url).get("/api/sessions").json() if s["source"] == "browser")["id"]


def call_in_chromium(tmp_path, served, during, *, retry_ms=(200, 400)):
    """Run a call of CALL_S seconds in the extension's offscreen document. `during(elapsed)` is
    called about every second, to break things. Returns the session id and when the call began and ended."""
    with chromium(tmp_path) as context:
        page = offscreen(context)
        began = begin(page, served, retry_ms)
        while (elapsed := time.time() - began) < CALL_S:
            during(elapsed)
            time.sleep(1)
        ended = time.time()
        page.evaluate("globalThis.scribe.stop()")  # returns once the call has drained, which can take a while
        uploaded(page)  # the document stays until every segment has reached the server
    return browser_session(served), began, ended


def recorded(api, sid, channel):
    """A recorded channel as samples, and the longest silence in it."""
    audio = api.get(f"/api/sessions/{sid}/recording/{channel}/audio")
    assert audio.status_code == 200
    samples, rate = soundfile.read(io.BytesIO(audio.content), dtype="float32")
    assert rate == RATE
    frame = RATE // 50
    loud = [float(np.sqrt(np.mean(samples[i : i + frame] ** 2))) > 0.01 for i in range(0, len(samples) - frame, frame)]
    longest = run = 0
    for on in loud:
        run = 0 if on else run + 1
        longest = max(longest, run)
    return samples, longest * 0.02, sum(loud) / len(loud)


def check_recording_whole(served, sid, began, ended):
    """Both channels are on disk from start to end with nothing missing: no silence longer than
    the pause between bursts (1.1 s at the longest), and the burst share the source has."""
    import httpx

    api = httpx.Client(base_url=served.url, timeout=60)
    info = api.get(f"/api/sessions/{sid}/recording").json()
    for channel, duty in (("tab", 1.2 / 2.1), ("mic", 0.9 / 2.0)):
        got = info["channels"][channel]
        assert got["missing"] == [] and got["chunks"] >= CALL_S // (SEGMENT_MS // 1000) - 1, got
        samples, longest_silence, share = recorded(api, sid, channel)
        seconds = len(samples) / RATE
        assert seconds >= (ended - began) - 3, f"{channel}: {seconds:.1f} s recorded of a {ended - began:.1f} s call"
        assert longest_silence <= 1.4, f"{channel}: a {longest_silence:.1f} s hole in the recording"
        assert abs(share - duty) < 0.12, f"{channel}: {share:.2f} of it is sound, the source is {duty:.2f}"
    print(
        f"\nrecorded {info['bytes'] / 1024:.0f} KiB for {ended - began:.0f} s of two channels: "
        f"{info['bytes'] / (ended - began) * 3600 / 1048576:.1f} MiB an hour"
    )
    return info


def fill_and_export(served, sid):
    import httpx

    api = httpx.Client(base_url=served.url, timeout=60)
    served.harness.loader.engines["alpha"].delay = 0.0
    before = api.get(f"/api/sessions/{sid}").json()["utterances"]
    job = api.post(f"/api/sessions/{sid}/reprocess").json()
    for _ in range(300):
        job = api.get(f"/api/transcriptions/{job['id']}").json()
        if job["status"] not in ("queued", "fetching", "decoding", "transcribing"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job
    after = api.get(f"/api/sessions/{sid}").json()["utterances"]
    export = api.get(f"/api/sessions/{sid}/export").text
    return before, after, job["result"], export


def test_a_starved_server_loses_no_audio_and_the_transcript_is_filled_afterwards(tmp_path, served):
    # Each transcription takes 4 s and two channels talk every couple of seconds: the queue fills
    # and the server answers 503 to the rest.
    served.harness.loader.engines["alpha"].delay = 4.0
    sid, began, ended = call_in_chromium(tmp_path, served, lambda elapsed: None)
    info = check_recording_whole(served, sid, began, ended)
    assert info["finalised"] is True

    served.harness.loader.engines["alpha"].delay = 0.0
    before, after, result, export = fill_and_export(served, sid)
    markers = [u for u in before if u["text"].startswith("[unavailable")]
    assert markers, "the server was not starved: nothing to fill"
    assert not [u for u in after if u["text"].startswith("[unavailable")]
    assert len(after) > len(before) - len(markers)
    assert "[unavailable" not in export and export.count("alpha:") == len(after)
    assert result["remaining"] == 0
    print(f"{len(markers)} of {len(before)} lines were markers; after the reprocess {len(after)} lines, none missing")
    # In order, and each from the right side of the call.
    assert [u["started_at"] for u in after] == sorted(u["started_at"] for u in after)
    assert {u["speaker_id"] for u in after} == {"tab", "mic"}
    assert {u["speaker_name"] for u in after if u["speaker_id"] == "mic"} == {"Alice"}


def test_a_server_stopped_mid_call_loses_no_audio_and_the_transcript_is_filled_afterwards(tmp_path, served):
    state = {"down": False, "up": False}

    def during(elapsed):
        if elapsed > 8 and not state["down"]:
            served.stop()  # the server goes away (a restart, a crash)...
            state["down"] = True
        if elapsed > 20 and not state["up"]:
            served.start()  # ...and comes back, with its database, while the call is still going
            state["up"] = True

    sid, began, ended = call_in_chromium(tmp_path, served, during)
    assert state["down"] and state["up"]
    check_recording_whole(served, sid, began, ended)

    before, after, result, export = fill_and_export(served, sid)
    # Nothing was transcribed for 12 s of a 30 s call, and no marker could be posted to say so.
    assert result["unheard"] > 0, result
    assert len(after) > len(before)
    covered = [u for u in after if 8 < u["started_at"] - began < 20]
    assert covered, "the transcript still has a hole where the server was down"
    assert "[unavailable" not in export


def test_a_server_down_when_the_call_ends_gets_the_recording_from_a_later_document(tmp_path, served):
    """The segments the server could not take are kept in the browser's own storage. If the
    document that held them goes away (Chrome closes it, the browser quits), the next one delivers them."""
    with chromium(tmp_path) as context:
        page = offscreen(context)
        began = begin(page, served)
        time.sleep(10)
        served.stop()  # down for the rest of the call and the end of it
        time.sleep(CALL_S - (time.time() - began))
        ended = time.time()
        page.evaluate("globalThis.scribe.stop()")
        left = page.evaluate("globalThis.scribe.uploader.pending")
        assert left >= 3, f"only {left} segments were waiting: the server was not down long enough"
        page.close()  # the document is gone, and with it everything in memory
        served.start()
        later = offscreen(context)  # loading it picks up what the first left in IndexedDB
        uploaded(later)
    check_recording_whole(served, browser_session(served), began, ended)
