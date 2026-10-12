"""The GUI in a real browser against a real server on the GPU: sign in, pick models, transcribe,
download, watch a live session, read history.

    uv run playwright install chromium   # once
    uv run pytest -m gui -s

Screenshots land in SCRIBE_GUI_SHOTS (default ../evidence/gui).
"""

import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.gui

TOKEN = "gui-token"
JFK = Path(__file__).parent / "fixtures" / "jfk.wav"
SHOTS = Path(os.environ.get("SCRIBE_GUI_SHOTS", Path(__file__).parents[2] / "evidence" / "gui"))
# The browser looks like a proxied, non-local client, so it has to sign in like one would over
# the tailnet or the public edge.
REMOTE = {"X-Forwarded-For": "100.64.0.7"}


@pytest.fixture(scope="module")
def server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    env = {
        **os.environ,
        "SCRIBE_PORT": str(port),
        "SCRIBE_DATA_DIR": tempfile.mkdtemp(prefix="scribe-gui-"),
        "SCRIBE_MODELS_DIR": os.environ.get("SCRIBE_MODELS_DIR", str(Path("~/.local/share/scribe/models").expanduser())),
        "SCRIBE_API_TOKEN": TOKEN,
        "SCRIBE_DEFAULT_MODEL": "",
    }
    process = subprocess.Popen([sys.executable, "-m", "scribe_server"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(240):
        try:
            httpx.get(f"{base}/api/whoami")
            break
        except httpx.HTTPError:
            time.sleep(0.5)
    yield base
    process.terminate()
    process.wait(10)


@pytest.fixture(scope="module")
def page(server):
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(extra_http_headers=REMOTE, viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        yield page
        browser.close()


def shot(page, name):
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)


def use_model(page, model_id, timeout=600_000):
    card = page.locator(f'[data-model="{model_id}"]')
    card.get_by_role("button", name="Use").click()
    page.locator(f'[data-model="{model_id}"].active').wait_for(timeout=timeout)


def try_clip(page):
    page.get_by_role("tab", name="Try it").click()
    page.evaluate("document.querySelector('#try-result').replaceChildren()")
    page.locator("#try-file").set_input_files(str(JFK))
    page.locator("#try-text").wait_for(timeout=120_000)
    return page.locator("#try-result").inner_text()


def test_gui(server, page):
    api = httpx.Client(base_url=server, headers={"Authorization": f"Bearer {TOKEN}"})

    # Sign-in is required off-host; a wrong token is refused.
    page.goto(server)
    page.locator("#login").wait_for()
    shot(page, "01-sign-in")
    page.fill("#token", "wrong")
    page.click("#login-form button")
    page.get_by_text("Wrong token.").wait_for()
    page.fill("#token", TOKEN)
    page.click("#login-form button")
    page.locator(".card").first.wait_for()
    assert page.locator(".card").count() == 8

    # Pick Parakeet, transcribe the clip.
    use_model(page, "parakeet-tdt-0.6b-v2")
    assert "Active: Parakeet TDT 0.6B v2" in page.inner_text("#status")
    shot(page, "02-models-parakeet-active")
    result = try_clip(page)
    assert "Parakeet TDT 0.6B v2" in result
    assert "ask not what your country can do for you" in result.lower()
    shot(page, "03-try-parakeet")

    # Swap to Whisper turbo through the GUI; the same clip now goes through Whisper.
    page.get_by_role("tab", name="Models").click()
    use_model(page, "whisper-large-v3-turbo")
    result = try_clip(page)
    assert "Whisper large-v3 turbo" in result
    assert "ask not what your country can do for you" in result.lower()
    shot(page, "04-try-whisper-turbo")

    # Delete a model and download it again, watching the progress bar finish.
    page.get_by_role("tab", name="Models").click()
    tiny = page.locator('[data-model="whisper-tiny.en"]')
    tiny.get_by_role("button", name="Delete").click()
    tiny.get_by_role("button", name=re.compile("^Download")).wait_for()
    tiny.get_by_role("button", name=re.compile("^Download")).click()
    tiny.locator("progress").wait_for(timeout=30_000)
    shot(page, "05-downloading")
    tiny.get_by_role("button", name="Use").wait_for(timeout=600_000)

    # The active model cannot be deleted: the card offers no Delete.
    assert page.locator('[data-model="whisper-large-v3-turbo"]').get_by_role("button", name="Delete").count() == 0

    # Live: a session started elsewhere (the bot) streams into the page.
    session = api.post("/api/sessions", json={"source": "discord", "title": "GUI live test"}).json()
    page.get_by_role("tab", name="Live").click()
    page.get_by_role("button", name="GUI live test").wait_for(timeout=15_000)
    page.get_by_role("button", name="GUI live test").click()
    api.post(
        f"/api/sessions/{session['id']}/utterances",
        files={"file": ("u.wav", JFK.read_bytes(), "audio/wav")},
        data={"speaker_id": "1", "speaker_name": "Ada", "started_at": str(time.time())},
    ).raise_for_status()
    page.locator("#live-transcript .line").first.wait_for(timeout=30_000)
    assert "Ada" in page.inner_text("#live-transcript")
    shot(page, "06-live")

    # History: the session, its export links, deleting it.
    api.post(f"/api/sessions/{session['id']}/end").raise_for_status()
    page.get_by_role("tab", name="History").click()
    page.locator("#history button").first.click()
    page.locator("#history-view .line").first.wait_for()
    assert page.locator('#history-view a[href$="format=md"]').count() == 1
    shot(page, "07-history")
    page.locator("#history-view").get_by_role("button", name="Delete").click()
    page.get_by_text("Deleted.").wait_for()
    assert api.get(f"/api/sessions/{session['id']}").status_code == 404
