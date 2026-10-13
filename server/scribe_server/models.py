"""Model state: downloads, the one active model, and swapping it safely (docs/spec.md, 3.3)."""

import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .catalogue import CatalogueEntry, by_id
from .engines import Engine

log = logging.getLogger("scribe.models")

COMPLETE = ".complete"


class Conflict(Exception):
    """The request is valid but clashes with the model's current state (HTTP 409)."""


class NotFound(Exception):
    pass


class NoActiveModel(Exception):
    pass


@dataclass
class LoadedModel:
    id: str
    engine: Engine
    refs: int = 0
    retired: bool = False


@dataclass
class Download:
    process: subprocess.Popen | None = None
    total: int = 0
    cancelled: bool = False
    thread: threading.Thread | None = None


# (entry, destination) -> None; raises on failure. Replaceable in tests.
Downloader = Callable[[CatalogueEntry, Path, Download], None]
Loader = Callable[[CatalogueEntry, Path], Engine]


def subprocess_downloader(entry: CatalogueEntry, dest: Path, download: Download) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "scribe_server.download", entry.repo, str(dest), *entry.files],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    download.process = process
    assert process.stdout is not None
    first = process.stdout.readline()
    if first.startswith("total "):
        download.total = int(first.split()[1])
    _, stderr = process.communicate()
    if download.cancelled:
        raise Conflict("cancelled")
    if process.returncode != 0:
        raise RuntimeError(stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {process.returncode}")


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class ModelManager:
    def __init__(
        self,
        catalogue: list[CatalogueEntry],
        models_dir: Path,
        loader: Loader,
        downloader: Downloader = subprocess_downloader,
        state_file: Path | None = None,
    ):
        self._entries = by_id(catalogue)
        self._order = [entry.id for entry in catalogue]
        self._dir = models_dir
        self._loader = loader
        self._downloader = downloader
        self._state_file = state_file
        self._lock = threading.Lock()
        self._downloads: dict[str, Download] = {}
        self._errors: dict[str, str] = {}
        self._active: LoadedModel | None = None
        self._loading: str | None = None
        self._dir.mkdir(parents=True, exist_ok=True)
        # Leftover partial downloads are cleaned only by the first server on this directory; a
        # second one (a test run beside the installed service) must not delete a live download.
        self._lock_file = open(self._dir / ".scribe.lock", "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._clean_partials()
        except BlockingIOError:
            log.warning("another scribe server uses %s; leaving partial downloads alone", self._dir)

    # -- paths and state ------------------------------------------------------------------------

    def _path(self, model_id: str) -> Path:
        return self._dir / model_id

    def _partial(self, model_id: str) -> Path:
        return self._dir / f"{model_id}.partial"

    def _clean_partials(self) -> None:
        for path in self._dir.glob("*.partial"):
            log.info("removing leftover partial download %s", path.name)
            shutil.rmtree(path, ignore_errors=True)

    def _entry(self, model_id: str) -> CatalogueEntry:
        try:
            return self._entries[model_id]
        except KeyError:
            raise NotFound(f"no model {model_id!r} in the catalogue") from None

    def downloaded(self, model_id: str) -> bool:
        return (self._path(model_id) / COMPLETE).exists()

    def status(self, model_id: str) -> dict:
        entry = self._entry(model_id)
        download = self._downloads.get(model_id)
        if download is not None:
            size = _dir_size(self._partial(model_id))
            progress = min(size / download.total, 0.99) if download.total else 0.0
            state = {"status": "downloading", "progress": round(progress, 3)}
        elif self.downloaded(model_id):
            state = {"status": "downloaded", "progress": 1.0}
        else:
            state = {"status": "not_downloaded", "progress": 0.0}
        active = self._active
        return {
            **entry.public(),
            **state,
            "active": active is not None and active.id == model_id,
            "loading": self._loading == model_id,
            "error": self._errors.get(model_id),
        }

    def list(self) -> list[dict]:
        return [self.status(model_id) for model_id in self._order]

    @property
    def active_id(self) -> str | None:
        return self._active.id if self._active else None

    @property
    def loading_id(self) -> str | None:
        return self._loading

    # -- downloads ------------------------------------------------------------------------------

    def download(self, model_id: str) -> None:
        entry = self._entry(model_id)
        with self._lock:
            if model_id in self._downloads:
                raise Conflict(f"{model_id} is already downloading")
            if self.downloaded(model_id):
                raise Conflict(f"{model_id} is already downloaded")
            download = Download()
            self._downloads[model_id] = download
            self._errors.pop(model_id, None)
        download.thread = threading.Thread(
            target=self._run_download, args=(entry, download), name=f"download-{model_id}", daemon=True
        )
        download.thread.start()

    def _run_download(self, entry: CatalogueEntry, download: Download) -> None:
        partial = self._partial(entry.id)
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        try:
            self._downloader(entry, partial, download)
            (partial / COMPLETE).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            final = self._path(entry.id)
            shutil.rmtree(final, ignore_errors=True)
            partial.rename(final)
            log.info("downloaded %s", entry.id)
        except Exception as error:
            shutil.rmtree(partial, ignore_errors=True)
            if not download.cancelled:
                log.warning("download of %s failed: %s", entry.id, error)
                self._errors[entry.id] = f"download failed: {error}"
        finally:
            with self._lock:
                self._downloads.pop(entry.id, None)

    def wait_for_download(self, model_id: str, timeout: float | None = None) -> None:
        download = self._downloads.get(model_id)
        if download and download.thread:
            download.thread.join(timeout)

    def delete(self, model_id: str) -> None:
        """Delete a downloaded model, or cancel its download. Never the active or loading one."""
        self._entry(model_id)
        with self._lock:
            if self.active_id == model_id or self._loading == model_id:
                raise Conflict(f"{model_id} is in use; activate another model first")
            download = self._downloads.get(model_id)
            if download is not None:
                download.cancelled = True
                if download.process is not None:
                    download.process.kill()
        if download is not None:
            self.wait_for_download(model_id, timeout=30)
            return
        if not self._path(model_id).exists():
            raise Conflict(f"{model_id} is not downloaded")
        shutil.rmtree(self._path(model_id))
        self._errors.pop(model_id, None)

    # -- activation and use ---------------------------------------------------------------------

    def activate(self, model_id: str, wait: bool = False) -> None:
        """Load `model_id` beside the current model, then swap it in.

        In-flight jobs keep the model they started with; the old model is unloaded when the last
        of them finishes (`release`).
        """
        entry = self._entry(model_id)
        with self._lock:
            if self._loading is not None:
                raise Conflict(f"{self._loading} is still loading")
            if model_id in self._downloads or not self.downloaded(model_id):
                raise Conflict(f"{model_id} is not downloaded")
            if self.active_id == model_id:
                return
            self._loading = model_id
            self._errors.pop(model_id, None)
        thread = threading.Thread(target=self._load, args=(entry,), name=f"load-{model_id}", daemon=True)
        thread.start()
        if wait:
            thread.join()

    def _load(self, entry: CatalogueEntry) -> None:
        try:
            started = time.monotonic()
            engine = self._loader(entry, self._path(entry.id))
            log.info("loaded %s in %.1fs", entry.id, time.monotonic() - started)
        except Exception as error:
            log.exception("loading %s failed", entry.id)
            with self._lock:
                self._errors[entry.id] = f"load failed: {error}"
                self._loading = None
            return
        with self._lock:
            old = self._active
            self._active = LoadedModel(entry.id, engine)
            self._loading = None
            if old is not None:
                old.retired = True
                unload = old.refs == 0
            else:
                unload = False
        if unload:
            self._unload(old)
        self._save_state()

    def acquire(self) -> LoadedModel:
        with self._lock:
            if self._active is None:
                raise NoActiveModel("no model is active; pick one in the GUI")
            self._active.refs += 1
            return self._active

    def release(self, model: LoadedModel) -> None:
        with self._lock:
            model.refs -= 1
            unload = model.retired and model.refs == 0
        if unload:
            self._unload(model)

    def _unload(self, model: LoadedModel) -> None:
        log.info("unloading %s", model.id)
        model.engine.close()

    # -- persistence ----------------------------------------------------------------------------

    def _save_state(self) -> None:
        if self._state_file and self.active_id:
            self._state_file.write_text(json.dumps({"active": self.active_id}))

    def restore(self, default: str | None) -> None:
        """Re-activate the model active before the last restart, or the default if downloaded."""
        wanted = None
        if self._state_file and self._state_file.exists():
            wanted = json.loads(self._state_file.read_text()).get("active")
        for model_id in (wanted, default):
            if model_id and model_id in self._entries and self.downloaded(model_id):
                self.activate(model_id)
                return
