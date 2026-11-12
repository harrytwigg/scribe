"""The audio of a browser call, kept on disk as the extension uploads it (docs/spec.md, 4.4).

The extension records each channel (the tab, the mic) as a series of short self-contained Opus/WebM
segments and uploads them as it goes. This is storage and nothing else: no model, no queue, so a
server too busy to transcribe a call can still keep its audio.

    recordings/<session id>/manifest.json
    recordings/<session id>/<tab|mic>/<index>.webm

Each segment says when (wall-clock time) it began, so a stretch of the call is found by time, and a
missing segment is a gap in the timeline, not a break in the file. Neighbouring segments overlap by
a fraction of a second so no audio falls between them; where they overlap the later one is used.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

from .audio import RATE, AudioError, decode

log = logging.getLogger("scribe.recordings")

CHANNELS = ("tab", "mic")
MAX_INDEX = 99_999
MAX_CHUNK_BYTES = 32 * 1024 * 1024
SUFFIXES = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/wav": ".wav", "audio/x-wav": ".wav"}
SESSION_ID = re.compile(r"^[0-9a-f]{12}$")
DAY = 86400.0


class RecordingError(ValueError):
    pass


def _suffix(mime: str) -> str:
    return SUFFIXES.get(mime.split(";")[0].strip().lower(), ".audio")


class Recordings:
    """`keep_days`: how long after a call ends its audio is kept. `after_reprocess_days`: how long
    once a reprocess has left the call with no gaps (the audio did its job). `max_bytes`: the most
    all recordings may take together; the oldest ended calls go first."""

    def __init__(
        self, root: Path, keep_days: float = 14, after_reprocess_days: float = 3, max_bytes: int = 10 * 1024**3
    ):
        self.root = root
        self.keep_days = keep_days
        self.after_reprocess_days = after_reprocess_days
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    # -- manifest -------------------------------------------------------------------------------

    def _dir(self, session_id: str) -> Path:
        if not SESSION_ID.match(session_id):
            raise RecordingError(f"not a session id: {session_id!r}")
        return self.root / session_id

    def _load(self, session_id: str) -> dict | None:
        try:
            return json.loads((self._dir(session_id) / "manifest.json").read_text())
        except FileNotFoundError:
            return None

    def _save(self, session_id: str, manifest: dict) -> None:
        path = self._dir(session_id) / "manifest.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(path)

    def manifest(self, session_id: str) -> dict | None:
        with self._lock:
            return self._load(session_id)

    # -- writing --------------------------------------------------------------------------------

    def put(
        self, session_id: str, channel: str, index: int, started_at: float, data: bytes, *,
        mime: str = "audio/webm", duration_ms: int = 0, sample_rate: int = 0,
    ) -> bool:
        """Store one segment. Returns False if this index was already stored: a retry, which
        changes nothing, so an upload can be repeated until it is acknowledged."""
        if channel not in CHANNELS:
            raise RecordingError(f"unknown channel {channel!r}: expected one of {', '.join(CHANNELS)}")
        if not 0 <= index <= MAX_INDEX:
            raise RecordingError(f"segment index must be between 0 and {MAX_INDEX}")
        if not data:
            raise RecordingError("the segment is empty")
        if len(data) > MAX_CHUNK_BYTES:
            raise RecordingError(f"segment larger than {MAX_CHUNK_BYTES // (1024 * 1024)} MB")
        with self._lock:
            folder = self._dir(session_id)
            manifest = self._load(session_id) or {
                "version": 1, "session_id": session_id, "created_at": time.time(), "finalised_at": None,
                "reprocessed_at": None, "reprocess_complete": False, "scanned_bytes": None, "auto_attempts": 0,
                "last_chunk_at": None, "channels": {},
            }
            entry = manifest["channels"].setdefault(channel, {"mime": mime, "sample_rate": sample_rate, "chunks": {}})
            if str(index) in entry["chunks"]:
                return False
            name = f"{channel}/{index:05d}{_suffix(mime)}"
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            # The manifest is written after the file: a crash between the two leaves a file the
            # manifest does not know, which the retried upload then registers.
            entry["chunks"][str(index)] = {
                "file": name, "started_at": float(started_at), "duration_ms": max(int(duration_ms), 0), "bytes": len(data),
            }
            manifest["last_chunk_at"] = time.time()
            manifest["reprocess_complete"] = False  # there is audio nobody has been through yet
            self._save(session_id, manifest)
            return True

    def finalise(self, session_id: str, now: float | None = None) -> dict | None:
        """The call is over: note it in the manifest and return the summary. Segments that are
        still in flight are accepted afterwards, and the summary follows them."""
        with self._lock:
            manifest = self._load(session_id)
            if manifest is None:
                return None
            manifest["finalised_at"] = time.time() if now is None else now
            self._save(session_id, manifest)
            return self._summary(manifest)

    def note_reprocess(self, session_id: str, complete: bool, scanned_bytes: int | None = None, now: float | None = None) -> None:
        """A pass over the recording finished; `complete` if it left the call with no gap (which
        starts the shorter retention), `scanned_bytes` how much audio it had seen."""
        with self._lock:
            manifest = self._load(session_id)
            if manifest is None:
                return
            manifest["reprocessed_at"] = time.time() if now is None else now
            manifest["reprocess_complete"] = complete
            manifest["scanned_bytes"] = scanned_bytes
            self._save(session_id, manifest)

    def count_auto_attempt(self, session_id: str) -> int:
        with self._lock:
            manifest = self._load(session_id)
            if manifest is None:
                return 0
            manifest["auto_attempts"] = manifest.get("auto_attempts", 0) + 1
            manifest["auto_last"] = time.time()
            self._save(session_id, manifest)
            return manifest["auto_attempts"]

    def delete(self, session_id: str) -> None:
        import shutil

        with self._lock:
            shutil.rmtree(self._dir(session_id), ignore_errors=True)

    # -- reading --------------------------------------------------------------------------------

    @staticmethod
    def _spans(entry: dict) -> list[tuple[float, float, dict]]:
        """(start, end, chunk) of each segment, in time order. A segment with no known duration
        is taken to run to the next one's start (or for one segment length if it is the last)."""
        chunks = sorted(entry["chunks"].items(), key=lambda kv: (kv[1]["started_at"], int(kv[0])))
        spans = []
        for position, (_, chunk) in enumerate(chunks):
            start = chunk["started_at"]
            if chunk["duration_ms"]:
                end = start + chunk["duration_ms"] / 1000
            elif position + 1 < len(chunks):
                end = chunks[position + 1][1]["started_at"]
            else:
                end = start + 30
            spans.append((start, end, chunk))
        return spans

    def _summary(self, manifest: dict) -> dict:
        channels = {}
        for name, entry in manifest["channels"].items():
            spans = self._spans(entry)
            indexes = sorted(int(i) for i in entry["chunks"])
            channels[name] = {
                "mime": entry["mime"],
                "chunks": len(indexes),
                "bytes": sum(c["bytes"] for c in entry["chunks"].values()),
                "started_at": spans[0][0] if spans else None,
                "ended_at": max((end for _, end, _ in spans), default=None),
                # Indexes the extension numbered but this server never received.
                "missing": [i for i in range(indexes[-1] + 1) if i not in set(indexes)] if indexes else [],
            }
        return {
            "session_id": manifest["session_id"],
            "finalised": manifest["finalised_at"] is not None,
            "reprocessed_at": manifest["reprocessed_at"],
            "bytes": sum(c["bytes"] for c in channels.values()),
            "channels": channels,
        }

    def summary(self, session_id: str) -> dict | None:
        with self._lock:
            manifest = self._load(session_id)
            return self._summary(manifest) if manifest else None

    def coverage(self, session_id: str, channel: str, start: float, end: float) -> float:
        """Seconds of [start, end] that some segment covers."""
        with self._lock:
            manifest = self._load(session_id)
        if manifest is None or channel not in manifest["channels"]:
            return 0.0
        covered, cursor = 0.0, start
        for a, b, _ in self._spans(manifest["channels"][channel]):
            a, b = max(a, cursor), min(b, end)
            if b > a:
                covered += b - a
                cursor = b
        return covered

    def read(
        self, session_id: str, channel: str, start: float | None = None, end: float | None = None
    ) -> tuple[np.ndarray, float] | None:
        """The channel's audio between two wall-clock times (all of it by default), 16 kHz mono,
        with silence where no segment covers; and the wall-clock time of its first sample.
        None if the call has no recording of that channel."""
        with self._lock:
            manifest = self._load(session_id)
        if manifest is None or channel not in manifest["channels"]:
            return None
        entry = manifest["channels"][channel]
        spans = self._spans(entry)
        if not spans:
            return None
        origin = spans[0][0] if start is None else start
        stop = max(e for _, e, _ in spans) if end is None else end
        buffer = np.zeros(max(int((stop - origin) * RATE), 0) + RATE, dtype=np.float32)
        top = 0
        for a, b, chunk in spans:
            if a >= stop or b + 1 <= origin:
                continue
            try:
                audio = decode((self._dir(session_id) / chunk["file"]).read_bytes())
            except (AudioError, OSError) as error:
                log.warning("segment %s of %s unreadable: %s", chunk["file"], session_id, error)
                continue
            offset = round((a - origin) * RATE)
            if offset < 0:
                audio, offset = audio[-offset:], 0
            if offset + len(audio) > len(buffer):
                if end is not None:
                    audio = audio[: len(buffer) - offset]
                else:
                    buffer = np.concatenate([buffer, np.zeros(offset + len(audio) - len(buffer), dtype=np.float32)])
            buffer[offset : offset + len(audio)] = audio
            top = max(top, offset + len(audio))
        length = int((stop - origin) * RATE) if end is not None else top
        return buffer[: max(length, 0)], origin

    # -- retention ------------------------------------------------------------------------------

    def sessions(self) -> list[str]:
        try:
            return sorted(p.name for p in self.root.iterdir() if p.is_dir() and SESSION_ID.match(p.name))
        except FileNotFoundError:
            return []

    def purge(self, ended: dict[str, float | None], now: float | None = None) -> list[str]:
        """Delete recordings that are past retention. `ended` maps every session the store knows to
        when it ended (None while open). Audio of a session that no longer exists goes too. An
        open call's audio is never deleted. Returns the sessions whose audio was deleted."""
        now = time.time() if now is None else now
        deleted, kept = [], []
        for session_id in self.sessions():
            manifest = self.manifest(session_id)
            if session_id not in ended:  # the session was deleted
                self.delete(session_id)
                deleted.append(session_id)
                continue
            ended_at = ended[session_id]
            if ended_at is None or manifest is None:  # still open (or nothing written yet)
                continue
            reprocessed = manifest.get("reprocessed_at") or now
            past_reprocess = manifest.get("reprocess_complete") and now - reprocessed > self.after_reprocess_days * DAY
            if past_reprocess or now - ended_at > self.keep_days * DAY:
                self.delete(session_id)
                deleted.append(session_id)
            else:
                kept.append((ended_at, session_id, self._summary(manifest)["bytes"]))
        total = sum(size for _, _, size in kept)
        for _, session_id, size in sorted(kept):  # oldest first
            if total <= self.max_bytes:
                break
            self.delete(session_id)
            deleted.append(session_id)
            total -= size
        if deleted:
            log.info("deleted the recordings of %d session(s) past retention", len(deleted))
        return deleted
