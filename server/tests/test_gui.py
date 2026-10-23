"""The GUI in a real browser against a real server on the GPU: sign in, pick models, transcribe,
download, watch a live session, read history.

    uv run playwright install chromium   # once
    uv run pytest -m gui -s

Screenshots land in SCRIBE_GUI_SHOTS (default ../evidence/gui).
"""

import os
import re
import shutil
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
CONVERSATION = Path(__file__).parent / "fixtures" / "conversation.wav"
# Text to type into the next prompt() the page opens (renaming a speaker).
PROMPT_ANSWERS: list[str] = []
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
    data_dir = tempfile.mkdtemp(prefix="scribe-gui-")
    env = {
        **os.environ,
        "SCRIBE_PORT": str(port),
        "SCRIBE_DATA_DIR": data_dir,
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
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def page(server):
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(extra_http_headers=REMOTE, viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept(PROMPT_ANSWERS.pop(0) if dialog.type == "prompt" and PROMPT_ANSWERS else ""))
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

    # Rename a speaker by clicking their name: the page and the stored lines both change.
    PROMPT_ANSWERS.append("Ada Lovelace")
    page.locator("#live-transcript .who").first.click()
    page.locator("#live-transcript .who", has_text="Ada Lovelace").wait_for()
    assert api.get(f"/api/sessions/{session['id']}").json()["utterances"][0]["speaker_name"] == "Ada Lovelace"

    sid = session["id"]

    def say(speaker="Ada"):
        api.post(
            f"/api/sessions/{sid}/utterances",
            files={"file": ("u.wav", JFK.read_bytes(), "audio/wav")},
            data={"speaker_id": "1", "speaker_name": speaker, "started_at": str(time.time())},
        ).raise_for_status()

    # A refresh keeps the tab and the call (they are in the URL).
    assert page.evaluate("location.hash") == f"#live/{sid}"
    page.reload()
    page.locator("#live-transcript .line").first.wait_for(timeout=15_000)
    assert page.get_by_role("tab", name="Live").get_attribute("aria-selected") == "true"
    assert "Ada Lovelace" in page.inner_text("#live-transcript")

    # Rename the call by clicking its title: the header and the call list both change.
    PROMPT_ANSWERS.append("Weekly sync")
    page.locator("#live-transcript button.title").click()
    page.locator("#live-transcript button.title", has_text="Weekly sync").wait_for()
    page.locator("#live-sessions button", has_text="Weekly sync").wait_for(timeout=10_000)
    assert api.get(f"/api/sessions/{sid}").json()["title"] == "Weekly sync"

    # History follows an open call too: a new line appears without a refresh.
    page.get_by_role("tab", name="History").click()
    page.locator("#history button", has_text="Weekly sync").click()
    page.locator("#history-view .line").first.wait_for()
    assert page.evaluate("location.hash") == f"#history/{sid}"
    say()
    page.locator("#history-view .line").nth(1).wait_for(timeout=30_000)
    shot(page, "06b-history-live")

    # With the event stream broken (a buffering proxy, say), the poll still brings lines in.
    context = page.context
    blind = context.new_page()
    blind.add_init_script("window.EventSource = class { addEventListener() {} close() {} }")
    blind.goto(f"{server}/#live/{sid}")
    blind.locator("#live-transcript .line").nth(1).wait_for(timeout=15_000)
    say()
    blind.locator("#live-transcript .line").nth(2).wait_for(timeout=30_000)
    blind.close()

    # Ending the call shows in the open view without a refresh.
    api.post(f"/api/sessions/{sid}/end").raise_for_status()
    page.locator("#history-view .state", has_text="ended").wait_for(timeout=10_000)
    page.locator("#history button", has_text="Weekly sync").wait_for()
    assert "live" not in page.locator("#history button", has_text="Weekly sync").inner_text()

    # History: the session, its export links, deleting it.
    page.locator("#history-view .line").first.wait_for()
    assert page.locator('#history-view a[href$="format=md"]').count() == 1
    shot(page, "07-history")
    page.locator("#history-view").get_by_role("button", name="Delete").click()
    page.get_by_text("Deleted.").wait_for()
    assert api.get(f"/api/sessions/{session['id']}").status_code == 404

    # Try it with speaker separation: the three-speaker conversation comes back as three speakers.
    deadline = time.time() + 300
    while api.get("/api/status").json()["diarization"] == "loading" and time.time() < deadline:
        time.sleep(1)
    assert api.get("/api/status").json()["diarization"] == "ready"
    page.get_by_role("tab", name="Try it").click()
    page.evaluate("document.querySelector('#try-result').replaceChildren()")
    page.locator("#try-speakers").check()
    page.locator("#try-file").set_input_files(str(CONVERSATION))
    page.locator("#try-text .line").first.wait_for(timeout=120_000)
    assert "3 speakers" in page.inner_text("#try-result")
    assert {page.locator("#try-text .who").nth(i).inner_text() for i in range(page.locator("#try-text .who").count())} == {
        "Speaker 1", "Speaker 2", "Speaker 3",
    }
    shot(page, "08-try-speakers")


def test_gui_keeps_its_place(server, page):
    """With two calls live: switching tabs and back keeps each tab's call, a list refresh during a
    slow open does not take the viewer elsewhere, and a call deleted elsewhere is noticed."""
    api = httpx.Client(base_url=server, headers={"Authorization": f"Bearer {TOKEN}"})
    a = api.post("/api/sessions", json={"source": "browser", "title": "Nav A"}).json()["id"]
    time.sleep(0.05)
    b = api.post("/api/sessions", json={"source": "browser", "title": "Nav B"}).json()["id"]  # newest

    def say(sid):
        api.post(
            f"/api/sessions/{sid}/utterances",
            files={"file": ("u.wav", JFK.read_bytes(), "audio/wav")},
            data={"speaker_id": "1", "speaker_name": "Ada", "started_at": str(time.time())},
        ).raise_for_status()

    title = lambda box: page.locator(f"{box} button.title").inner_text()  # noqa: E731

    # Live opens the newest call; pick the other one, go to History and back: still A.
    page.get_by_role("tab", name="Live").click()
    page.locator("#live-sessions button", has_text="Nav A").click()
    page.locator("#live-transcript button.title", has_text="Nav A").wait_for()
    page.get_by_role("tab", name="History").click()
    page.get_by_role("tab", name="Live").click()
    page.locator("#live-transcript button.title", has_text="Nav A").wait_for()
    assert page.evaluate("location.hash") == f"#live/{a}"
    say(a)  # and it follows A again
    page.locator("#live-transcript .line").first.wait_for(timeout=30_000)
    page.wait_for_timeout(5000)  # a list refresh or two
    assert title("#live-transcript") == "Nav A"

    # History: open B, go to Models and back: B is still open, in the URL, and followed.
    page.get_by_role("tab", name="History").click()
    page.locator("#history button", has_text="Nav B").click()
    page.locator("#history-view button.title", has_text="Nav B").wait_for()
    page.get_by_role("tab", name="Models").click()
    page.get_by_role("tab", name="History").click()
    page.locator("#history-view button.title", has_text="Nav B").wait_for()
    assert page.evaluate("location.hash") == f"#history/{b}"
    say(b)
    page.locator("#history-view .line").first.wait_for(timeout=30_000)

    # A slow open (6 s, longer than the 4 s list refresh) still ends on the call clicked.
    page.get_by_role("tab", name="Live").click()
    page.locator("#live-transcript button.title", has_text="Nav A").wait_for()
    page.route(f"**/api/sessions/{b}", lambda route: (time.sleep(6), route.continue_()))
    page.locator("#live-sessions button", has_text="Nav B").click()
    page.wait_for_timeout(9000)
    page.unroute(f"**/api/sessions/{b}")
    page.locator("#live-transcript button.title", has_text="Nav B").wait_for(timeout=10_000)
    assert page.evaluate("location.hash") == f"#live/{b}"

    # B is deleted elsewhere: the viewer notices instead of showing it live forever.
    api.delete(f"/api/sessions/{b}").raise_for_status()
    page.locator("#live-transcript button.title", has_text="Nav B").wait_for(state="detached", timeout=15_000)
    api.post(f"/api/sessions/{a}/end").raise_for_status()


def test_gui_transcribe_tab(server, page):
    """Queue recordings and links in the Transcribe tab, and follow each one's progress."""
    page.goto(server)
    page.locator("#login, .card").first.wait_for()  # signed in already, or asked to
    if page.locator("#login").is_visible():
        page.fill("#token", TOKEN)
        page.click("#login-form button")
    page.locator(".card").first.wait_for()
    if "Active:" not in page.inner_text("#status"):
        use_model(page, "parakeet-tdt-0.6b-v2")
    page.get_by_role("tab", name="Transcribe").click()
    page.evaluate("localStorage.removeItem('scribe-tx-jobs')")

    # Nothing chosen: told what to do, nothing is sent.
    page.click("#tx-go")
    assert "Choose audio files or paste podcast links" in page.inner_text("#tx-status")
    assert page.locator(".tx-item").count() == 0

    # A recording with several voices, separated; alone in the list, its transcript opens by itself.
    page.check("#tx-speakers")
    page.locator("#tx-file").set_input_files(str(CONVERSATION))
    page.click("#tx-go")
    page.locator(".tx-item").first.wait_for()
    page.locator(".tx-item .tx-text .line").first.wait_for(timeout=180_000)
    result = page.inner_text(".tx-item")
    assert "Speaker 2" in result and "Timaeus" in result and "Done:" in result
    assert page.locator(".tx-item a", has_text="SRT").get_attribute("href").endswith("format=srt")
    shot(page, "07-transcribe-upload")

    # The transcript is in History, where it can be renamed and exported.
    page.get_by_role("link", name="Open in History").click()
    page.locator("#history-view .line").first.wait_for(timeout=15_000)
    assert "Timaeus" in page.inner_text("#history-view")

    # A queue: two files and a link the server may not fetch (not a public address), at once.
    page.get_by_role("tab", name="Transcribe").click()
    page.locator(".tx-item").first.get_by_role("button", name="Remove").click()
    page.uncheck("#tx-speakers")
    page.fill("#tx-links", "http://127.0.0.1:9/episode.mp3")
    page.locator("#tx-file").set_input_files([str(JFK), str(JFK)])
    page.click("#tx-go")
    assert page.locator(".tx-item").count() == 3
    # progress is per item: the files show a bar while they wait or run
    assert page.locator(".tx-item progress").count() >= 1
    page.locator(".tx-item .state.error", has_text="not a public address").wait_for(timeout=60_000)
    page.get_by_text("3 of 3 finished").wait_for(timeout=60_000)
    shot(page, "08-transcribe-queue")
    done = page.locator(".tx-item", has_text="Done:")
    assert done.count() >= 2
    done.first.get_by_role("button", name="Show transcript").click()
    page.locator(".tx-item .tx-text .line").first.wait_for()
    assert "ask not what your country" in page.inner_text(".tx-item .tx-text").lower()
    # A finished item can be removed from the list (its transcript stays in History).
    before = page.locator(".tx-item").count()
    page.locator(".tx-item").first.get_by_role("button", name="Remove").click()
    assert page.locator(".tx-item").count() == before - 1
