import threading
import time

import numpy as np

from .conftest import TOKEN, silence, tone, wav

REMOTE = {"X-Forwarded-For": "100.64.0.9"}


def post_utterance(client, session_id, audio, speaker="u1", name="Ada", headers=None, key=None):
    data = {"speaker_id": speaker, "speaker_name": name, "started_at": str(time.time())}
    if key:
        data["client_key"] = key
    return client.post(
        f"/api/sessions/{session_id}/utterances",
        files={"file": ("u.wav", wav(audio), "audio/wav")},
        data=data,
        headers=headers or {},
    )


def test_gui_and_models_listing(harness):
    assert harness.client.get("/").status_code == 200
    models = harness.client.get("/api/models").json()
    assert [m["id"] for m in models] == ["alpha", "beta", "gamma"]
    assert {"status", "progress", "active", "loading", "accuracy", "speed", "licence"} <= set(models[0])


def test_model_lifecycle_over_http(harness):
    c = harness.client
    assert c.post("/api/models/alpha/activate").status_code == 409
    assert c.post("/api/models/alpha/download").status_code == 202
    harness.models.wait_for_download("alpha")
    assert c.post("/api/models/alpha/activate").status_code == 202
    deadline = time.monotonic() + 5
    while c.get("/api/status").json()["active_model"] != "alpha":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert c.delete("/api/models/alpha").status_code == 409
    assert c.post("/api/models/nope/download").status_code == 404


def test_openai_transcription(harness):
    harness.ready("alpha")
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(tone(3)), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["text"].startswith("alpha:")
    r = harness.client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav(tone(3)), "audio/wav")},
        data={"model": "alpha", "response_format": "verbose_json"},
    )
    assert r.json()["model"] == "alpha"
    assert harness.client.get("/v1/models").json()["data"][0]["id"] == "alpha"


def test_transcription_rejects_other_model(harness):
    harness.ready("alpha")
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(tone(1)), "audio/wav")}, data={"model": "beta"})
    assert r.status_code == 409
    assert "alpha" in r.json()["detail"]


def test_no_active_model_is_503(harness):
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(tone(1)), "audio/wav")})
    assert r.status_code == 503


def test_bad_audio_is_400(harness):
    harness.ready("alpha")
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400


def test_silence_never_reaches_the_model(harness):
    harness.ready("alpha")
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(silence(5)), "audio/wav")})
    assert r.json()["text"] == ""
    assert harness.loader.engines["alpha"].calls == 0


def test_long_upload_is_chunked(harness):
    harness.ready("alpha")
    audio = np.concatenate([np.concatenate([tone(9), silence(0.5)]) for _ in range(8)])
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(audio), "audio/wav")})
    assert r.status_code == 200
    assert harness.loader.engines["alpha"].calls >= 3


def test_queue_full_is_503(harness):
    harness.ready("alpha")
    harness.loader.engines["alpha"].delay = 0.5
    results = []

    def call():
        r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(tone(1)), "audio/wav")})
        results.append(r.status_code)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 503 in results
    assert 200 in results


def test_session_flow_and_exports(harness):
    harness.ready("alpha")
    c = harness.client
    session = c.post("/api/sessions", json={"source": "discord", "guild": "g", "channel": "c"}).json()
    sid = session["id"]
    line = post_utterance(c, sid, tone(2), "u1", "Ada")
    assert line.status_code == 201, line.text
    assert line.json()["speaker_name"] == "Ada"
    assert post_utterance(c, sid, silence(2)).json()["skipped"] is True
    post_utterance(c, sid, tone(1), "u2", "Bob")
    full = c.get(f"/api/sessions/{sid}").json()
    assert [u["speaker_name"] for u in full["utterances"]] == ["Ada", "Bob"]
    md = c.get(f"/api/sessions/{sid}/export?format=md").text
    assert "**[00:00:00] Ada:** alpha:" in md
    srt = c.get(f"/api/sessions/{sid}/export?format=srt").text
    assert srt.startswith("1\n00:00:00,")
    assert c.get(f"/api/sessions/{sid}/export?format=json").json()["utterances"][1]["speaker_id"] == "u2"
    assert c.get(f"/api/sessions/{sid}/export?format=doc").status_code == 400
    assert c.get("/api/status").json()["open_sessions"] == 1
    c.post(f"/api/sessions/{sid}/end")
    assert post_utterance(c, sid, tone(1)).status_code == 409
    assert c.get("/api/status").json()["open_sessions"] == 0
    assert c.delete(f"/api/sessions/{sid}").status_code == 200
    assert c.get(f"/api/sessions/{sid}").status_code == 404


def test_optouts(harness):
    c = harness.client
    c.post("/api/optouts", json={"source": "discord", "user_id": "42", "opted_out": True})
    assert c.get("/api/optouts/discord").json() == ["42"]
    c.post("/api/optouts", json={"source": "discord", "user_id": "42", "opted_out": False})
    assert c.get("/api/optouts/discord").json() == []


def test_every_request_needs_the_token(harness):
    c = harness.anonymous
    for headers in ({}, REMOTE):  # loopback is not trusted: agent sessions share the box
        assert c.get("/api/models", headers=headers).status_code == 401
        assert c.get("/api/sessions", headers=headers).status_code == 401
        assert c.post("/v1/audio/transcriptions", headers=headers).status_code == 401
    assert c.get("/api/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert c.get("/").status_code == 200  # the sign-in page itself is public
    assert c.get("/api/whoami").json() == {"authorised": False, "auth_enabled": True}


def test_trust_localhost_is_opt_in(tmp_path):
    from fastapi.testclient import TestClient

    from scribe_server.app import create_app
    from scribe_server.auth import Auth

    from .conftest import Harness

    h = Harness(tmp_path)
    h.services.auth = Auth(TOKEN, trust_localhost=True)
    c = TestClient(create_app(h.services))
    assert c.get("/api/models").status_code == 200
    assert c.get("/api/models", headers=REMOTE).status_code == 401  # proxied: not local
    h.worker.stop()


def test_no_token_means_no_checks(tmp_path):
    from .conftest import Harness

    h = Harness(tmp_path, token=None)
    assert h.client.get("/api/models", headers=REMOTE).status_code == 200
    h.worker.stop()


def test_gui_login_cookie(harness):
    c = harness.anonymous
    assert c.post("/api/login", json={"token": "wrong"}, headers=REMOTE).status_code == 401
    r = c.post("/api/login", json={"token": TOKEN}, headers=REMOTE)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert c.get("/api/models", headers=REMOTE).status_code == 200
    c.post("/api/logout", headers=REMOTE)
    assert c.get("/api/models", headers=REMOTE).status_code == 401


def test_cross_origin_writes_refused_even_locally(harness):
    harness.models.download("alpha")
    harness.models.wait_for_download("alpha")
    c = harness.client
    evil = {"Origin": "https://evil.example"}
    assert c.post("/api/models/alpha/activate", headers=evil).status_code == 403
    assert c.post("/api/login", json={"token": TOKEN}, headers=evil).status_code == 403
    assert c.post("/api/models/alpha/activate", headers={"Origin": "http://testserver"}).status_code == 202


def test_extension_origin_allowed_with_token_only(harness):
    """The Chrome extension's own origin may write (docs/spec.md 4), but still needs the token,
    and cannot sign in for a cookie."""
    from scribe_server.auth import Auth
    from scribe_server.config import EXTENSION_ORIGIN

    harness.services.auth = Auth(TOKEN, extension_origins=(EXTENSION_ORIGIN,))
    ext = {"Origin": EXTENSION_ORIGIN, **REMOTE}
    bearer = {"Authorization": f"Bearer {TOKEN}"}
    c = harness.anonymous
    assert c.post("/api/sessions", json={"source": "browser"}, headers=ext).status_code == 401
    assert c.post("/api/sessions", json={"source": "browser"}, headers={**ext, **bearer}).status_code == 201
    other = {"Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop", **REMOTE, **bearer}
    assert c.post("/api/sessions", json={"source": "browser"}, headers=other).status_code == 403
    assert c.post("/api/login", json={"token": TOKEN}, headers=ext).status_code == 403


def test_abandoned_browser_sessions_are_ended(harness, monkeypatch):
    """A browser that quits mid-call never ends its session; the server does, once it has seen
    no line and no heartbeat for `browser_idle_s` (300 s). Discord sessions are the bot's to end."""
    import asyncio

    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    c = harness.client
    reap = lambda now: asyncio.run(harness.app.state.reap_idle(now))  # noqa: E731
    quiet = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    alive = c.post("/api/sessions", json={"source": "browser"}).json()["id"]
    discord = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    clock[0] = 1250.0
    assert c.post(f"/api/sessions/{alive}/heartbeat").status_code == 200

    assert reap(1299.0) == []
    assert reap(1301.0) == [quiet]
    assert c.get(f"/api/sessions/{quiet}").json()["ended_at"] is not None
    assert c.post(f"/api/sessions/{quiet}/heartbeat").status_code == 409
    assert c.get("/api/status").json()["open_sessions"] == 2
    assert reap(1551.0) == [alive]
    assert c.get(f"/api/sessions/{discord}").json()["ended_at"] is None


def test_marker_lines(harness):
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    r = c.post(f"/api/sessions/{sid}/markers", json={"speaker_id": "u1", "speaker_name": "Ada", "started_at": time.time(), "text": "[unavailable: 3 s of audio from Ada]"})
    assert r.status_code == 201
    assert c.get(f"/api/sessions/{sid}").json()["utterances"][0]["text"].startswith("[unavailable")


def test_live_events_stream(harness):
    """The SSE feed delivers a line posted after the listener connected (real uvicorn, one loop)."""
    import socket

    import httpx
    import uvicorn

    harness.ready("alpha")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(harness.app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        auth = {"Authorization": f"Bearer {TOKEN}"}
        sid = httpx.post(f"{base}/api/sessions", json={"source": "discord"}, headers=auth).json()["id"]
        received = []

        def listen():
            with httpx.stream("GET", f"{base}/api/sessions/{sid}/events", timeout=10, headers=auth) as response:
                for raw in response.iter_lines():
                    if raw.startswith("event:"):
                        received.append((time.monotonic(), raw[6:].strip()))
                        if raw.endswith("end"):
                            return

        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
        time.sleep(0.5)
        r = httpx.post(
            f"{base}/api/sessions/{sid}/utterances",
            files={"file": ("u.wav", wav(tone(1)), "audio/wav")},
            data={"speaker_id": "u1", "speaker_name": "Ada", "started_at": str(time.time())},
            headers=auth,
        )
        assert r.status_code == 201, r.text
        # Both of these are published from handlers that must run on the event loop.
        httpx.post(f"{base}/api/sessions/{sid}/markers", headers=auth,
                   json={"speaker_id": "u1", "speaker_name": "Ada", "started_at": time.time(), "text": "[gap]"})
        ended = time.monotonic()
        httpx.post(f"{base}/api/sessions/{sid}/end", headers=auth)
        listener.join(5)
        assert [name for _, name in received] == ["line", "line", "end"]
        assert received[-1][0] - ended < 1.0, "the end event arrived late"
    finally:
        server.should_exit = True
        thread.join(5)


def test_retried_utterance_is_stored_once(harness):
    harness.ready("alpha")
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    first = post_utterance(c, sid, tone(2), key="u1:100.0").json()
    calls = harness.loader.engines["alpha"].calls
    second = post_utterance(c, sid, tone(2), key="u1:100.0").json()
    assert second["id"] == first["id"]
    assert harness.loader.engines["alpha"].calls == calls, "a stored key must not be transcribed again"
    post_utterance(c, sid, tone(2), key="u1:105.0")
    assert len(c.get(f"/api/sessions/{sid}").json()["utterances"]) == 2


def test_concurrent_duplicate_is_stored_once(harness):
    """The retry arrives while the first attempt is still on the GPU: both run, one row is kept."""
    harness.ready("alpha")
    harness.loader.engines["alpha"].delay = 0.3
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    results = []
    threads = [threading.Thread(target=lambda: results.append(post_utterance(c, sid, tone(1), key="k").json())) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results[0]["id"] == results[1]["id"]
    assert len(c.get(f"/api/sessions/{sid}").json()["utterances"]) == 1


def test_upload_size_cap(harness):
    harness.ready("alpha")
    harness.services.max_upload_bytes = 1000
    r = harness.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav(tone(1)), "audio/wav")})
    assert r.status_code == 413


def test_unknown_host_refused(tmp_path):
    from fastapi.testclient import TestClient

    from scribe_server.app import create_app

    from .conftest import Harness

    h = Harness(tmp_path)
    h.services.allowed_hosts = ["127.0.0.1", "harry-box", "*.ts.net"]
    c = TestClient(create_app(h.services), headers={"Authorization": f"Bearer {TOKEN}"})
    assert c.get("/api/models", headers={"Host": "attacker.example"}).status_code == 400
    assert c.get("/api/models", headers={"Host": "harry-box:8787"}).status_code == 200
    assert c.get("/api/models", headers={"Host": "harry-box.tail1234.ts.net"}).status_code == 200
    h.worker.stop()


def test_store_migrates_old_database(tmp_path):
    import sqlite3

    from scribe_server.store import Store

    db = sqlite3.connect(tmp_path / "old.db")
    db.executescript(
        "create table sessions (id text primary key, source text not null, title text not null, guild text, "
        "channel text, started_at real not null, ended_at real);"
        "create table utterances (id integer primary key autoincrement, session_id text not null, speaker_id text not null, "
        "speaker_name text not null, started_at real not null, duration_ms integer not null, text text not null, "
        "model_id text, asr_ms integer);"
        "insert into sessions values ('s', 'discord', 't', null, null, 1, null);"
        "insert into utterances (session_id, speaker_id, speaker_name, started_at, duration_ms, text) values ('s', 'u', 'U', 1, 1, 'old');"
    )
    db.commit()
    db.close()
    store = Store(tmp_path / "old.db")
    assert store.utterances("s")[0]["text"] == "old"
    _, new = store.add_utterance("s", "u", "U", 2, 1, "new", None, None, "k")
    assert new
    _, again = store.add_utterance("s", "u", "U", 2, 1, "new", None, None, "k")
    assert not again


def test_open_sessions_file(harness, tmp_path):
    from fastapi.testclient import TestClient

    from scribe_server.app import create_app

    harness.services.open_sessions_file = tmp_path / "open-sessions"
    c = TestClient(create_app(harness.services), headers={"Authorization": f"Bearer {TOKEN}"})
    assert harness.services.open_sessions_file.read_text() == "0\n"
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    assert harness.services.open_sessions_file.read_text() == "1\n"
    c.post(f"/api/sessions/{sid}/end")
    assert harness.services.open_sessions_file.read_text() == "0\n"


def test_rename_session(harness):
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "browser", "title": "Meet - abc-defg-hij"}).json()["id"]
    r = c.patch(f"/api/sessions/{sid}", json={"title": "  Weekly sync  "})
    assert r.status_code == 200 and r.json()["title"] == "Weekly sync"
    assert c.get(f"/api/sessions/{sid}").json()["title"] == "Weekly sync"
    assert c.patch(f"/api/sessions/{sid}", json={"title": " "}).status_code == 400
    assert c.patch(f"/api/sessions/{sid}", json={"title": "x" * 121}).status_code == 400
    assert c.patch("/api/sessions/nope", json={"title": "x"}).status_code == 404
    assert harness.anonymous.patch(f"/api/sessions/{sid}", json={"title": "x"}).status_code == 401


def test_names_are_one_plain_line(harness):
    c = harness.client
    sid = c.post("/api/sessions", json={"source": "discord"}).json()["id"]
    assert c.patch(f"/api/sessions/{sid}", json={"title": "Weekly\n# sync\x07\t now"}).json()["title"] == "Weekly # sync now"
    assert c.post(f"/api/sessions/{sid}/speakers/u1", json={"name": " Ada\nLovelace\x00"}).json()["name"] == "Ada Lovelace"
    assert c.get(f"/api/sessions/{sid}").json()["speakers"] == {"u1": "Ada Lovelace"}
    assert c.patch(f"/api/sessions/{sid}", json={"title": "\n\x07"}).status_code == 400
