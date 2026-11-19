"""The Chrome extension (docs/spec.md, 4) in a real Chromium against a real server on the GPU.

Chromium loads the unpacked extension. A page of the extension runs the same capture code the
offscreen document runs, fed the reference clip twice: as the "tab" (an audio element's stream,
standing in for the captured Meet tab) and as the "mic" (Chromium's fake capture device). Both
channels must come back as transcript lines, attributed, in a session the extension ended.

Not covered, because it needs a person: `chrome.tabCapture` itself (it only works from a click
on the extension's button), and the mic permission prompt.

    uv run playwright install chromium   # once
    uv run pytest -m extension -s
"""

import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import httpx
import pytest

from .wer import words

pytestmark = pytest.mark.extension

JFK = Path(__file__).parent / "fixtures" / "jfk.wav"
EXTENSION = Path(__file__).parents[2] / "extension"
EXTENSION_ID = "mcnpmckibkddnnenfondlblnddldonfm"
SHOTS = Path(os.environ.get("SCRIBE_GUI_SHOTS", Path(__file__).parents[2] / "evidence" / "extension"))
PHRASE = "ask not what your country can do for you"


def padded_clip(lead_s: float, tail_s: float) -> bytes:
    """The reference clip with silence around it, so each pass is one clean utterance."""
    with wave.open(str(JFK)) as source:
        params, frames = source.getparams(), source.readframes(source.getnframes())
    pad = lambda seconds: b"\0" * int(params.framerate * seconds) * params.sampwidth  # noqa: E731
    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setparams(params)
        target.writeframes(pad(lead_s) + frames + pad(tail_s))
    return out.getvalue()


@pytest.fixture(scope="module")
def server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    data_dir = tempfile.mkdtemp(prefix="scribe-ext-")
    env = {
        **os.environ,
        "SCRIBE_PORT": str(port),
        "SCRIBE_DATA_DIR": data_dir,
        "SCRIBE_MODELS_DIR": os.environ.get("SCRIBE_MODELS_DIR", str(Path("~/.local/share/scribe/models").expanduser())),
        "SCRIBE_DEFAULT_MODEL": "parakeet-tdt-0.6b-v2",
    }
    env.pop("SCRIBE_EXTENSION_ORIGINS", None)  # the built-in default must admit the extension
    process = subprocess.Popen([sys.executable, "-m", "scribe_server"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    api = httpx.Client(base_url=base)
    for _ in range(240):
        try:
            status = api.get("/api/status").json()
            if status.get("active_model") and status.get("diarization") == "ready":
                break
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    yield base
    process.terminate()
    process.wait(10)
    shutil.rmtree(data_dir, ignore_errors=True)


def test_extension_transcribes_tab_and_mic(server, tmp_path):
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    mic_clip = tmp_path / "mic.wav"
    mic_clip.write_bytes(padded_clip(1.0, 14.0))  # the fake mic loops this file
    tab_clip = base64.b64encode(padded_clip(1.5, 2.0)).decode()
    api = httpx.Client(base_url=server)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(tmp_path / "profile"),
            channel="chromium",
            headless=True,
            args=[
                f"--disable-extensions-except={EXTENSION}",
                f"--load-extension={EXTENSION}",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={mic_clip}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        worker = context.service_workers[0] if context.service_workers else context.wait_for_event("serviceworker")
        # The manifest's public key pins the ID, and so the origin the server admits.
        assert worker.url.startswith(f"chrome-extension://{EXTENSION_ID}/")
        config = {
            "server": server, "myName": "Alice", "othersName": "Meet", "useMic": True, "separateSpeakers": True, "record": True,
        }

        # Settings: saved through the options page, which tests the connection.
        options = context.new_page()
        options.goto(f"chrome-extension://{EXTENSION_ID}/src/options.html")
        options.fill("[name=server]", server + "/")
        options.fill("[name=myName]", "Alice")
        options.fill("[name=othersName]", "Meet")
        options.click("text=Save and test")
        options.get_by_text("Connected: active model parakeet-tdt-0.6b-v2").wait_for()
        options.screenshot(path=str(SHOTS / "00-options.png"))
        assert worker.evaluate("chrome.storage.local.get(null)") == config
        options.close()

        # The popup's start path through the background and offscreen document: a stream ID that
        # tab capture never issued fails cleanly, with the error shown and no session left open.
        popup = context.new_page()
        popup.goto(f"chrome-extension://{EXTENSION_ID}/src/popup.html")
        popup.set_viewport_size({"width": 340, "height": 260})
        result = popup.evaluate("chrome.runtime.sendMessage({ type: 'start', tabStreamId: 'bogus', title: 'x' })")
        assert result["ok"] is False and result["error"], result
        assert worker.evaluate("chrome.storage.session.get('status')")["status"]["state"] == "stopped"
        assert api.get("/api/status").json()["open_sessions"] == 0
        assert worker.evaluate("chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })") == []

        # The captured tab: Meet rewrites its URL mid-call (history.pushState, #fragment), which must
        # not end the call; a new document loading in the tab (reload, leaving) must.
        call_tab = context.new_page()
        call_tab.goto(server + "/")
        tab_id = worker.evaluate(f"chrome.tabs.query({{ url: '{server}/*' }}).then((tabs) => tabs[0].id)")
        worker.evaluate(
            "id => chrome.storage.session.set({ capturedTab: id, status: { state: 'recording', lines: 1 } })", tab_id
        )
        state = lambda: worker.evaluate("chrome.storage.session.get('status')")["status"]["state"]  # noqa: E731
        call_tab.evaluate("history.pushState({}, '', '/abc-defg-hij?authuser=0')")
        call_tab.evaluate("history.replaceState({}, '', '/abc-defg-hij')")
        call_tab.evaluate("location.hash = 'joined'")
        time.sleep(1.5)
        assert state() == "recording", "a same-document URL change ended the call"
        call_tab.reload()
        for _ in range(50):
            if state() == "stopped":
                break
            time.sleep(0.1)
        assert state() == "stopped", "a new document in the captured tab did not end the call"
        call_tab.close()

        # A status left saying "recording" by a document that died is cleared by Stop.
        worker.evaluate("chrome.storage.session.set({ status: { state: 'recording', lines: 3 } })")
        assert popup.evaluate("chrome.runtime.sendMessage({ type: 'stop' })")["ok"] is True
        assert worker.evaluate("chrome.storage.session.get('status')")["status"]["state"] == "stopped"

        page = context.new_page()
        logs = []
        page.on("console", lambda message: logs.append(message.text))
        page.goto(f"chrome-extension://{EXTENSION_ID}/src/offscreen.html")
        # (No wait_for_function here: the extension's CSP forbids the eval it polls with.)
        page.locator("script[src='offscreen.js']").wait_for(state="attached")
        for _ in range(100):
            if page.evaluate("Boolean(globalThis.scribe)"):
                break
            time.sleep(0.1)
        page.evaluate(
            """async ([config, clip]) => {
                const bytes = Uint8Array.from(atob(clip), (c) => c.charCodeAt(0))
                const audio = new Audio(URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })))
                audio.loop = true  // a call keeps going; a clip that ended would end the capture
                await audio.play()
                const tab = audio.captureStream()
                const mic = await navigator.mediaDevices.getUserMedia({ audio: true })
                await globalThis.scribe.start(config, 'Meet - abc-defg-hij', { tab, mic })
            }""",
            [config, tab_clip],
        )

        def lines():
            session = next(s for s in api.get("/api/sessions").json() if s["source"] == "browser")
            return session, api.get(f"/api/sessions/{session['id']}").json()["utterances"]

        deadline = time.time() + 120
        while time.time() < deadline:
            session, got = lines()
            if {line["speaker_id"].split("#")[0] for line in got} >= {"tab", "mic"}:
                break
            time.sleep(1)
        popup.locator("#detail", has_text="lines").wait_for()
        popup.screenshot(path=str(SHOTS / "01-popup-recording.png"))
        assert session["ended_at"] is None and session["title"] == "Meet - abc-defg-hij"

        # While the call drains, a new start is refused rather than racing it.
        refused = page.evaluate(
            """async ([config]) => {
                const stopping = globalThis.scribe.stop()
                const error = await globalThis.scribe.start(config, 'again', { tab: null, mic: null }).then(() => null, (e) => e.message)
                await stopping
                return error
            }""",
            [config],
        )
        assert refused == "still finishing the last call: try again in a moment"
        session, got = lines()
        assert session["ended_at"] is not None, "stopping ends the session"
        assert api.get("/api/status").json()["open_sessions"] == 0
        popup.locator("#state", has_text="stopped").wait_for()
        context.close()

    print(json.dumps([{k: line[k] for k in ("speaker_id", "speaker_name", "text", "model_id")} for line in got], indent=1))
    by_speaker = {}
    for line in got:
        by_speaker.setdefault(line["speaker_id"].split("#")[0], []).append(line)
    # The tab mixes everyone on the far side, so it is split into speakers; the mic is just you.
    assert {line["speaker_id"] for line in by_speaker["tab"]} == {"tab#1"}
    assert {line["speaker_name"] for line in by_speaker["tab"]} == {"Speaker 1"}
    assert {line["speaker_name"] for line in by_speaker["mic"]} == {"Alice"}
    for speaker in ("tab", "mic"):
        text = " ".join(line["text"] for line in by_speaker[speaker]).lower()
        # Words only: punctuation varies with where an utterance was cut ("ask not. what ...").
        assert " ".join(words(PHRASE)) in " ".join(words(text)), f"{speaker}: {text}"
        assert all(line["model_id"] == "parakeet-tdt-0.6b-v2" for line in by_speaker[speaker])
    (SHOTS / "transcript.json").write_text(json.dumps(got, indent=1))
