import threading
import time

import numpy as np

from .conftest import TOKEN, silence, tone, wav

REMOTE = {"X-Forwarded-For": "100.64.0.9"}


def post_utterance(client, session_id, audio, speaker="u1", name="Ada", headers=None):
    return client.post(
        f"/api/sessions/{session_id}/utterances",
        files={"file": ("u.wav", wav(audio), "audio/wav")},
        data={"speaker_id": speaker, "speaker_name": name, "started_at": str(time.time())},
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
                    if raw.startswith("data:"):
                        received.append(raw)
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
        listener.join(5)
        assert received and '"speaker_name": "Ada"' in received[0]
    finally:
        server.should_exit = True
        thread.join(5)
