import threading
import time

import pytest

from scribe_server.models import Conflict, ModelManager, NoActiveModel

from .conftest import CATALOGUE, FakeDownloader, FakeLoader


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met")


def test_download_then_activate(harness):
    m = harness.models
    assert m.status("alpha")["status"] == "not_downloaded"
    m.download("alpha")
    m.wait_for_download("alpha")
    assert m.status("alpha")["status"] == "downloaded"
    assert not list(m._dir.glob("*.partial"))
    m.activate("alpha", wait=True)
    assert m.active_id == "alpha"
    assert m.status("alpha")["active"] is True


def test_activate_requires_download(harness):
    with pytest.raises(Conflict):
        harness.models.activate("alpha")


def test_activate_and_download_conflicts_while_downloading(harness):
    harness.downloader.gate = threading.Event()
    m = harness.models
    m.download("alpha")
    wait_until(lambda: m.status("alpha")["progress"] > 0)
    assert m.status("alpha")["status"] == "downloading"
    assert m.status("alpha")["progress"] == pytest.approx(0.4)
    with pytest.raises(Conflict):
        m.activate("alpha")
    with pytest.raises(Conflict):
        m.download("alpha")
    harness.downloader.gate.set()
    m.wait_for_download("alpha")
    assert m.status("alpha")["status"] == "downloaded"


def test_delete_mid_download_cancels_and_cleans(harness):
    harness.downloader.gate = threading.Event()
    m = harness.models
    m.download("alpha")
    wait_until(lambda: (m._dir / "alpha.partial").exists())
    m.delete("alpha")
    assert m.status("alpha")["status"] == "not_downloaded"
    assert m.status("alpha")["error"] is None
    assert not (m._dir / "alpha.partial").exists()
    assert not (m._dir / "alpha").exists()


def test_failed_download_cleans_up_and_reports(harness):
    harness.downloader.fail = True
    m = harness.models
    m.download("alpha")
    m.wait_for_download("alpha")
    status = m.status("alpha")
    assert status["status"] == "not_downloaded"
    assert "network down" in status["error"]
    assert not (m._dir / "alpha.partial").exists()


def test_leftover_partials_removed_at_start(tmp_path):
    (tmp_path / "models" / "beta.partial").mkdir(parents=True)
    ModelManager(CATALOGUE, tmp_path / "models", loader=FakeLoader(), downloader=FakeDownloader())
    assert not (tmp_path / "models" / "beta.partial").exists()


def test_delete_active_refused(harness):
    harness.ready("alpha")
    with pytest.raises(Conflict):
        harness.models.delete("alpha")


def test_delete_downloaded(harness):
    harness.ready("alpha")
    harness.models.download("beta")
    harness.models.wait_for_download("beta")
    harness.models.delete("beta")
    assert harness.models.status("beta")["status"] == "not_downloaded"


def test_no_active_model(harness):
    with pytest.raises(NoActiveModel):
        harness.models.acquire()


def test_swap_waits_for_in_flight_jobs(harness):
    harness.ready("alpha")
    m = harness.models
    held = m.acquire()  # an in-flight job on alpha
    m.download("beta")
    m.wait_for_download("beta")
    m.activate("beta", wait=True)
    assert m.active_id == "beta"
    alpha = harness.loader.engines["alpha"]
    assert not alpha.closed, "alpha unloaded under a running job"
    alpha.transcribe(__import__("numpy").zeros(160, dtype="float32"))
    m.release(held)
    assert alpha.closed
    fresh = m.acquire()
    assert fresh.id == "beta"
    m.release(fresh)
    assert not harness.loader.engines["beta"].closed


def test_swap_unloads_idle_model_immediately(harness):
    harness.ready("alpha")
    harness.models.download("beta")
    harness.models.wait_for_download("beta")
    harness.models.activate("beta", wait=True)
    assert harness.loader.engines["alpha"].closed


def test_second_activate_while_loading_conflicts(harness):
    harness.models.download("alpha")
    harness.models.wait_for_download("alpha")
    harness.loader.load_gate = threading.Event()
    harness.models.activate("alpha")
    assert harness.models.status("alpha")["loading"] is True
    with pytest.raises(Conflict):
        harness.models.activate("alpha")
    harness.loader.load_gate.set()
    wait_until(lambda: harness.models.active_id == "alpha")


def test_restore_reactivates_last_model(harness, tmp_path):
    harness.ready("alpha")
    harness.models.download("beta")
    harness.models.wait_for_download("beta")
    harness.models.activate("beta", wait=True)
    again = ModelManager(CATALOGUE, tmp_path / "models", loader=FakeLoader(), downloader=FakeDownloader(),
                         state_file=tmp_path / "state.json")
    again.restore(default="alpha")
    wait_until(lambda: again.active_id == "beta")


def test_second_manager_on_same_dir_leaves_partials(tmp_path):
    first = ModelManager(CATALOGUE, tmp_path / "models", loader=FakeLoader(), downloader=FakeDownloader())
    (tmp_path / "models" / "beta.partial").mkdir()
    ModelManager(CATALOGUE, tmp_path / "models", loader=FakeLoader(), downloader=FakeDownloader())
    assert (tmp_path / "models" / "beta.partial").exists()
    assert first.status("beta")["status"] == "not_downloaded"
