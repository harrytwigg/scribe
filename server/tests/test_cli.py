"""The `scribe` command against a real HTTP server (the fake-engine harness behind uvicorn)."""

import json
import os
import socket
import threading
import time
import types
import urllib.parse

import numpy as np
import pytest
import uvicorn

from scribe_server import cli, podcast

from .conftest import Harness, silence, tone, wav
from .podcast_fakes import FakeFetcher, feed, itunes
from .test_podcast import EP, FEED_A, search, spotify_pages

AUDIO = wav(np.concatenate([silence(1), tone(20), silence(2), tone(20), silence(1)]))


@pytest.fixture
def served(tmp_path, monkeypatch):
    harness = Harness(tmp_path)
    harness.ready("alpha")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(harness.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    monkeypatch.setenv("SCRIBE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("SCRIBE_API_TOKEN", "test-token")
    monkeypatch.setenv("SCRIBE_ENV_FILE", str(tmp_path / "no-env"))
    harness.url = f"http://127.0.0.1:{port}"
    yield harness
    server.should_exit = True
    thread.join(5)
    harness.worker.stop()


def fake_network(monkeypatch, pages: dict) -> FakeFetcher:
    fake = FakeFetcher(pages)
    monkeypatch.setattr(podcast, "Fetcher", lambda **kwargs: fake)
    return fake


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_transcribe_a_file_prints_text(served, tmp_path, capsys):
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    code, out, err = run(capsys, "transcribe", str(path))
    assert code == 0
    assert out.strip() == "alpha: 20.0s alpha: 20.0s"
    assert "done:" in err and "uploading talk.wav" in err
    sessions = served.client.get("/api/sessions").json()
    assert [s["title"] for s in sessions] == ["talk"]


def test_json_out_file_and_no_save(served, tmp_path, capsys):
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    out_file = tmp_path / "deep" / "talk.json"
    code, out, _ = run(capsys, "transcribe", str(path), "--format", "json", "--out", str(out_file), "--no-save", "-q")
    assert code == 0 and out.strip() == str(out_file)
    data = json.loads(out_file.read_text())
    assert data["model"] == "alpha" and data["episode"] is None and data["session_id"] is None
    assert [(s["start"], s["end"]) for s in data["segments"]] == [(1.0, 21.0), (23.0, 43.0)]
    assert served.client.get("/api/sessions").json() == []


def test_srt_and_timestamped_text(served, tmp_path, capsys):
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    code, out, _ = run(capsys, "transcribe", str(path), "--format", "srt", "-q")
    assert code == 0 and "00:00:01,000 --> 00:00:21,000" in out
    code, out, _ = run(capsys, "transcribe", str(path), "--timestamps", "-q")
    assert out.splitlines()[0].startswith("[00:00:01] alpha:")


def test_a_spotify_link_is_downloaded_and_transcribed(served, tmp_path, monkeypatch, capsys):
    pages = spotify_pages()
    pages["https://cdn.example.com/12.mp3"] = (AUDIO, "audio/wav")
    fake_network(monkeypatch, pages)
    code, out, err = run(capsys, "transcribe", f"https://open.spotify.com/episode/{EP}", "--format", "json", "--keep-audio", str(tmp_path / "kept"))
    assert code == 0, err
    data = json.loads(out)
    assert data["episode"]["title"] == "Ep 12: The Big Idea" and data["episode"]["source"] == "spotify"
    assert data["title"] == "The Show: Ep 12: The Big Idea"
    assert len(data["segments"]) == 2
    assert [p.name for p in (tmp_path / "kept").iterdir()] == ["The-Show-Ep-12-The-Big-Idea.mp3"]


def test_a_spotify_exclusive_exits_3_with_a_json_error(served, monkeypatch, capsys):
    pages = spotify_pages()
    pages[search("The Show")] = itunes([{"collectionName": "The Show", "artistName": "Spotify Studios"}])
    pages["https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"media": "podcast", "entity": "podcastEpisode", "limit": 25, "term": "The Big Idea The Show"})] = itunes([])
    fake_network(monkeypatch, pages)
    code, out, err = run(capsys, "transcribe", f"https://open.spotify.com/episode/{EP}", "--format", "json")
    assert code == 3
    assert "DRM" in err and "exclusive" in err
    assert json.loads(out)["error"] == {"kind": "not_found", "message": err.strip().removeprefix("scribe: error: "), "exit_code": 3}


def test_resolve_and_list_do_not_download(monkeypatch, capsys):
    pages = spotify_pages()
    fake = fake_network(monkeypatch, pages)
    code, out, _ = run(capsys, "resolve", f"https://open.spotify.com/episode/{EP}")
    assert code == 0 and json.loads(out)["audio_url"] == "https://cdn.example.com/12.mp3"
    assert "https://cdn.example.com/12.mp3" not in fake.requested
    code, out, _ = run(capsys, "resolve", FEED_A, "--list")
    assert [e["title"] for e in json.loads(out)] == ["Ep 13: Something Else", "Ep 12: The Big Idea", "Ep 11: Older"]
    code, out, _ = run(capsys, "resolve", FEED_A, "--episode", "3")
    assert json.loads(out)["title"] == "Ep 11: Older"


def test_download_saves_the_audio(monkeypatch, tmp_path, capsys):
    fake_network(monkeypatch, {FEED_A: feed("The Show", [("Ep 1", "Wed, 04 Mar 2026 08:30:00 GMT", "1:00", "https://cdn.example.com/1.mp3")]), "https://cdn.example.com/1.mp3": (b"ID3" * 100, "audio/mpeg")})
    code, out, _ = run(capsys, "download", FEED_A, "--out", str(tmp_path), "-q")
    assert code == 0
    saved = tmp_path / "The-Show-Ep-1.mp3"
    assert out.strip() == str(saved) and saved.read_bytes() == b"ID3" * 100


def test_errors_have_distinct_exit_codes(served, tmp_path, monkeypatch, capsys):
    assert run(capsys, "transcribe", str(tmp_path / "missing.mp3"))[0] == 2
    monkeypatch.setenv("SCRIBE_API_TOKEN", "wrong")
    code, _, err = run(capsys, "status")
    assert code == 5 and "rejected the token" in err
    monkeypatch.setenv("SCRIBE_API_TOKEN", "test-token")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    code, _, err = run(capsys, "transcribe", str(bad), "-q")
    assert code == 6 and "cannot decode" in err
    good_url = os.environ["SCRIBE_URL"]
    monkeypatch.setenv("SCRIBE_URL", "http://127.0.0.1:1")
    code, _, err = run(capsys, "status")
    assert code == 5 and "cannot reach the scribe server" in err
    monkeypatch.setenv("SCRIBE_URL", good_url)
    fake_network(monkeypatch, {})
    assert run(capsys, "transcribe", "https://example.com/dead.mp3", "-q")[0] == 4


def test_settings_come_from_the_env_file_then_the_secrets_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"SCRIBE_API_TOKEN": "from-secrets"}))
    env = tmp_path / "env"
    env.write_text(f"# comment\nSCRIBE_SERVER_URL=http://box:9000/\nSCRIBE_SECRETS_FILE={secrets}\n")
    monkeypatch.setenv("SCRIBE_ENV_FILE", str(env))
    for name in ("SCRIBE_URL", "SCRIBE_API_TOKEN", "SCRIBE_SECRETS_FILE"):
        monkeypatch.delenv(name, raising=False)
    assert cli.server_url(None) == "http://box:9000"
    assert cli.api_token(None) == "from-secrets"
    assert cli.server_url("http://flag:1/") == "http://flag:1"
    monkeypatch.setenv("SCRIBE_API_TOKEN", "from-env")
    assert cli.api_token(None) == "from-env" and cli.api_token("flag") == "flag"


def test_a_wrong_token_is_reported_before_a_large_upload(served, tmp_path, monkeypatch, capsys):
    big = tmp_path / "big.wav"
    big.write_bytes(wav(tone(3)) + b"\0" * (30 * 1024 * 1024))  # the server would reset the connection mid-send
    monkeypatch.setenv("SCRIBE_API_TOKEN", "wrong")
    code, _, err = run(capsys, "transcribe", str(big), "-q")
    assert code == 5 and "rejected the token" in err and "cannot reach" not in err


def test_a_busy_server_queues_the_upload_and_the_cli_says_where_it_stands(served, tmp_path, capsys):
    served.services.max_jobs = 1
    served.loader.engines["alpha"].delay = 0.2
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    first = cli.Server(served.url, "test-token").call("POST", "/api/transcriptions", body=AUDIO)  # takes the only slot
    code, out, err = run(capsys, "transcribe", str(path))
    assert code == 0 and "queued (position 1)" in err and out.count("alpha:") == 2
    assert first["id"]


def two_files(tmp_path):
    paths = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(AUDIO)
        paths.append(str(path))
    return paths


def test_several_sources_are_transcribed_in_turn_with_headers(served, tmp_path, capsys):
    one, two = two_files(tmp_path)
    code, out, err = run(capsys, "transcribe", one, two)
    assert code == 0
    assert out.count("alpha: 20.0s alpha: 20.0s") == 2
    assert "=== one (" in out and "=== two (" in out and out.index("=== one") < out.index("=== two")
    assert "[1/2] uploading one.wav" in err and "[2/2] done:" in err
    assert sorted(s["title"] for s in served.client.get("/api/sessions").json()) == ["one", "two"]


def test_several_sources_as_json_and_a_failure_in_the_middle(served, tmp_path, monkeypatch, capsys):
    one, two = two_files(tmp_path)
    fake_network(monkeypatch, {})  # the link below is dead
    code, out, err = run(capsys, "transcribe", one, "https://example.com/dead.mp3", two, "--format", "json", "-q")
    assert code == 7  # some failed, some did not
    entries = json.loads(out)
    assert [e["ok"] for e in entries] == [True, False, True]
    assert entries[0]["title"] == "one" and entries[2]["title"] == "two" and len(entries[0]["segments"]) == 2
    assert entries[1]["source"] == "https://example.com/dead.mp3" and entries[1]["error"]["exit_code"] == 4


def test_several_sources_into_a_directory(served, tmp_path, capsys):
    one, two = two_files(tmp_path)
    out_dir = tmp_path / "out"
    code, out, _ = run(capsys, "transcribe", one, two, "--out", str(out_dir), "--format", "srt", "-q")
    assert code == 0
    assert out.splitlines() == [str(out_dir / "01-one.srt"), str(out_dir / "02-two.srt")]
    assert "00:00:01,000 --> 00:00:21,000" in (out_dir / "01-one.srt").read_text()


def test_sources_can_come_from_a_file_with_comments(served, tmp_path, capsys):
    one, two = two_files(tmp_path)
    listing = tmp_path / "list.txt"
    listing.write_text(f"# the queue\n{one}\n\n  {two}  \n")
    code, out, _ = run(capsys, "transcribe", "--from", str(listing), "--format", "json", "-q")
    assert code == 0 and [e["title"] for e in json.loads(out)] == ["one", "two"]


def test_all_failing_returns_the_first_failure_and_a_bad_source_is_a_usage_error_up_front(served, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    code, _, err = run(capsys, "transcribe", str(bad), str(bad), "-q")
    assert code == 6 and err.count("cannot decode") == 2
    before = served.client.get("/api/status").json()
    code, _, err = run(capsys, "transcribe", two_files(tmp_path)[0], "nonsense.mp3", "-q")
    assert code == 2 and "nonsense.mp3" in err
    assert served.client.get("/api/sessions").json() == []  # nothing was uploaded first
    assert run(capsys, "transcribe", "-q")[0] == 2
    assert before["transcriptions"]["running"] == 0


def test_no_active_model_is_reported_up_front(served, tmp_path, monkeypatch, capsys):
    served.models._active = None  # as before a model was ever picked
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    code, _, err = run(capsys, "transcribe", str(path), "-q")
    assert code == 5 and "no active model" in err


def test_usage_errors_come_before_the_server_is_contacted(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SCRIBE_URL", "http://127.0.0.1:1")
    assert run(capsys, "transcribe", str(tmp_path / "missing.mp3"))[0] == 2


def test_a_weak_match_is_warned_about_even_when_quiet(served, monkeypatch, capsys):
    pages = spotify_pages()
    # the feed's episode has the right title but no date or length to confirm it
    pages[FEED_A] = feed("The Show", [("The Big Idea", None, "", "https://cdn.example.com/12.mp3")])
    pages["https://cdn.example.com/12.mp3"] = (AUDIO, "audio/wav")
    fake_network(monkeypatch, pages)
    code, _, err = run(capsys, "transcribe", f"https://open.spotify.com/episode/{EP}", "-q")
    assert code == 0 and "warning: weak match" in err
    # a fully confirmed match is not
    fake_network(monkeypatch, {**spotify_pages(), "https://cdn.example.com/12.mp3": (AUDIO, "audio/wav")})
    code, _, err = run(capsys, "transcribe", f"https://open.spotify.com/episode/{EP}", "-q")
    assert code == 0 and "warning" not in err


def test_num_speakers_is_sent_and_implies_speakers(served, tmp_path, capsys):
    seen = []
    from scribe_server.diarize import DiarizerSlot
    from .test_diarize import FakeDiarizer

    class Counting(FakeDiarizer):
        def whole(self, audio, num_speakers=None, cancel=None):
            seen.append(num_speakers)
            return super().whole(audio, num_speakers)

    served.services.diarization = DiarizerSlot(Counting())
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    code, out, err = run(capsys, "transcribe", str(path), "--num-speakers", "2", "-q")
    assert code == 0 and seen == [2]
    assert run(capsys, "transcribe", str(path), "--num-speakers", "0")[0] == 2


def test_a_bad_speaker_count_is_a_usage_error_before_the_server_is_contacted(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SCRIBE_URL", "http://127.0.0.1:9")  # nothing listens: a connection attempt would be exit 5
    monkeypatch.setenv("SCRIBE_ENV_FILE", str(tmp_path / "no-env"))
    path = tmp_path / "talk.wav"
    path.write_bytes(AUDIO)
    for count in ("0", "21", "-3"):
        assert run(capsys, "transcribe", str(path), f"--num-speakers={count}")[0] == 2
