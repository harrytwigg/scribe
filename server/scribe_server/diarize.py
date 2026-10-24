"""Who spoke when, for audio that mixes several people (docs/spec.md, 3.8).

A browser call's tab channel carries everyone on the far side as one stream. sherpa-onnx finds
the speaker turns in it (pyannote segmentation 3.0) and a speaker embedding for each (NVIDIA
TitaNet small), both ONNX on the CPU. Two uses:

- **Live** (`turns` + `SpeakerBook`): each utterance is split into turns, and each turn's voice is
  matched against the speakers already heard in the session, so "Speaker 2" stays the same person
  for the whole call even though utterances arrive one at a time.
- **A whole recording** (`whole`): clustered in one pass, for uploaded files.

Discord needs none of this: it delivers each user's audio separately.
"""

import logging
import multiprocessing
import os
import tarfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from .audio import RATE

log = logging.getLogger("scribe.diarize")

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = f"{RELEASES}/speaker-recongition-models/nemo_en_titanet_small.onnx"

# Cosine similarity of a turn's voice to a known speaker's average voice, at or above which it is
# that speaker. TitaNet small on LibriSpeech test-clean: different speakers' 1.5 s turns scored
# 0.28 at the 95th percentile (max 0.46), the same speaker's 0.36 at the 5th; JFK's 1-2 s turns
# scored 0.38-0.61 against each other (docs/spec.md, 3.8).
MATCH = float(os.environ.get("SCRIBE_DIARIZE_MATCH", "0.35"))
# Turns shorter than this carry too little voice to move a speaker's average.
MIN_EMBED_S = 1.0
# Only a turn this long can found a new speaker; a shorter unmatched one goes to the nearest known
# speaker. Short turns are where same-speaker scores dip, and a split speaker is the worse error.
MIN_FOUND_S = float(os.environ.get("SCRIBE_DIARIZE_MIN_FOUND", "2.0"))
# Whole recordings: a speaker the clusterer found with under SCRAP_SHARE of the speech is usually a
# scrap of someone else (a voice over music, a laugh, the same person on a bad line), not a person.
# With under SCRAP_S seconds it always joins the speaker it sounds most like; with more, only if it
# sounds like them (MATCH), so a real third voice who spoke for a minute stays. A speaker with more
# than SCRAP_SHARE is never touched, however short the recording. EMBED_CAP_S bounds the audio each
# speaker's voice is measured on.
SCRAP_S = float(os.environ.get("SCRIBE_DIARIZE_SCRAP_S", "15"))
SCRAP_SHARE = float(os.environ.get("SCRIBE_DIARIZE_SCRAP_SHARE", "0.05"))
EMBED_CAP_S = 45.0
# The clustering distance for whole recordings (higher merges more): 0.7-0.9 found the right number
# of speakers on the test conversation, 0.5 split speakers. WHOLE_SPLIT_THRESHOLD is what a caller who
# says there are more speakers than the first pass found gets instead: over-split, then merged down.
WHOLE_THRESHOLD = 0.8
WHOLE_SPLIT_THRESHOLD = 0.5
# A turn shorter than this is merged into its neighbour: too short to transcribe alone.
MIN_TURN_S = 0.5
# Same-speaker turns closer than this are one turn.
JOIN_GAP_S = 0.5


@dataclass
class Turn:
    start: float  # seconds into the audio
    end: float
    label: int  # the clusterer's label, local to one call of `turns` or `whole`

    @property
    def duration(self) -> float:
        return self.end - self.start


def untangle(turns: list[Turn]) -> list[Turn]:
    """Turns that do not overlap. Where two speakers overlap (crosstalk), the one who started
    later has the floor until their turn ends, so no stretch of audio is transcribed twice."""
    turns = [t for t in turns if t.end > t.start]
    bounds = sorted({b for t in turns for b in (t.start, t.end)})
    out: list[Turn] = []
    for a, b in zip(bounds, bounds[1:]):
        covering = [t for t in turns if t.start <= a and t.end >= b]
        if not covering:
            continue
        label = max(covering, key=lambda t: (t.start, -t.label)).label
        if out and out[-1].label == label and out[-1].end == a:
            out[-1].end = b
        else:
            out.append(Turn(a, b, label))
    return out


def _join(turns: list[Turn]) -> list[Turn]:
    joined: list[Turn] = []
    for turn in turns:
        if joined and joined[-1].label == turn.label and turn.start - joined[-1].end < JOIN_GAP_S:
            joined[-1].end = max(joined[-1].end, turn.end)
        else:
            joined.append(turn)
    return joined


def tidy(turns: list[Turn]) -> list[Turn]:
    """Non-overlapping turns in order, close same-speaker turns joined, and slivers folded into a
    neighbour (the turn before, or for leading slivers the turn after)."""
    joined = _join(untangle([Turn(t.start, t.end, t.label) for t in turns]))
    out: list[Turn] = []
    lead: list[Turn] = []  # slivers before the first real turn
    for turn in joined:
        if turn.duration >= MIN_TURN_S:
            if lead:
                turn.start = lead[0].start
                lead = []
            out.append(turn)
        elif out:
            out[-1].end = max(out[-1].end, turn.end)
        else:
            lead.append(turn)
    if lead:  # nothing but slivers: one turn, for whoever spoke most of it
        longest = max(lead, key=lambda t: t.duration)
        out.append(Turn(lead[0].start, lead[-1].end, longest.label))
    return _join(out)


@dataclass
class Speaker:
    number: int  # 1-based, in order of first appearance in the session
    centroid: np.ndarray  # unit-length average voice
    weight: float  # seconds of speech behind the centroid


@dataclass
class SpeakerBook:
    """The speakers heard so far in one session's mixed channel."""

    speakers: list[Speaker] = field(default_factory=list)

    def match(self, voice: np.ndarray, seconds: float) -> Speaker:
        """The known speaker this voice belongs to, or a new one; learns from long turns."""
        voice = voice / (np.linalg.norm(voice) or 1.0)
        best, score = None, -1.0
        for speaker in self.speakers:
            similarity = float(voice @ speaker.centroid)
            if similarity > score:
                best, score = speaker, similarity
        if best is not None and (score >= MATCH or seconds < MIN_FOUND_S):
            if seconds >= MIN_EMBED_S:
                mixed = best.centroid * best.weight + voice * seconds
                best.centroid = mixed / np.linalg.norm(mixed)
                best.weight += seconds
            return best
        # max + 1, not len + 1: numbers must stay unique even if the book has a gap.
        speaker = Speaker(max((sp.number for sp in self.speakers), default=0) + 1, voice, seconds)
        self.speakers.append(speaker)
        return speaker


def _fetch(url: str, dest: Path) -> None:
    partial = dest.with_suffix(dest.suffix + ".partial")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(dest)


def fetch_models(models_dir: Path) -> Path:
    """Download the segmentation and embedding models on first use (~45 MB); returns their folder."""
    folder = models_dir / "_diarization"
    folder.mkdir(parents=True, exist_ok=True)
    segmentation = folder / "pyannote-segmentation-3.0.onnx"
    embedding = folder / "nemo_en_titanet_small.onnx"
    if not segmentation.exists():
        archive = folder / "segmentation.tar.bz2"
        _fetch(SEGMENTATION_URL, archive)
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/model.onnx"))
            source = tar.extractfile(member)
            assert source is not None
            partial = segmentation.with_suffix(".partial")
            partial.write_bytes(source.read())
            partial.replace(segmentation)
        archive.unlink()
    if not embedding.exists():
        _fetch(EMBEDDING_URL, embedding)
    return folder


class Sherpa:
    """The sherpa-onnx models and nothing else: raw turns and voice embeddings. Its calls hold
    Python's global lock for as long as they run (a minute or more for a long recording), so the
    server runs them in child processes (`Diarizer`, `isolated`), never next to the web server."""

    def __init__(self, models_dir: Path, threads: int = 8, role: str = "both"):
        """`role`: "recordings" or "calls" load only the clusterer that child uses; "both" loads each."""
        import sherpa_onnx

        folder = fetch_models(models_dir)
        segmentation = folder / "pyannote-segmentation-3.0.onnx"
        embedding = folder / "nemo_en_titanet_small.onnx"
        self._embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=threads)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self._embedding_config)

        def config(threshold: float):
            made = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                    num_threads=threads,
                ),
                embedding=self._embedding_config,
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not made.validate():
                raise RuntimeError("invalid diarization config")
            return made

        self._config = config
        self._threshold = WHOLE_THRESHOLD  # what the whole-recording clusterer is currently set to

        # Its clustering threshold is a distance: higher merges more. For a whole recording, 0.7-0.9
        # found the right speaker count on the test conversation (0.5 split speakers). For live
        # utterances it is set to over-split instead: the session's book merges the pieces back,
        # while a merge here (a short turn lumped in with the next speaker) cannot be undone.
        self._whole = sherpa_onnx.OfflineSpeakerDiarization(config(WHOLE_THRESHOLD)) if role in ("recordings", "both") else None
        self._live = (
            sherpa_onnx.OfflineSpeakerDiarization(config(float(os.environ.get("SCRIBE_DIARIZE_LIVE_THRESHOLD", "0.3"))))
            if role in ("calls", "both")
            else None
        )

    def turns(self, audio: np.ndarray, live: bool, threshold: float | None = None) -> list[tuple[float, float, int]]:
        """`threshold` (whole recordings only): the clustering distance to use instead of the default."""
        if not live and (wanted := threshold or WHOLE_THRESHOLD) != self._threshold:
            self._whole.set_config(self._config(wanted))
            self._threshold = wanted
        result = (self._live if live else self._whole).process(audio).sort_by_start_time()
        return [(s.start, s.end, s.speaker) for s in result]

    def voice(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(RATE, audio)
        stream.input_finished()
        return np.array(self._extractor.compute(stream), dtype=np.float32)


# -- child processes -----------------------------------------------------------------------------
# One for live calls and one for whole recordings, as the two used to have their own models and
# locks: a long upload never holds up a call. Each loads its models once, at start.


class Cancelled(RuntimeError):
    pass


def _watch_parent(parent_pid: int) -> None:
    """A child should not outlive the server, even one that was killed outright. This is a Python
    thread, and a sherpa call holds the global lock, so a child that is busy exits when its call
    ends, not at once (systemd's KillMode=control-group, the default, stops it sooner). PDEATHSIG
    would be immediate, but it fires when the *thread* that started the child exits, and the
    children are started from a loader thread that does."""
    while True:
        time.sleep(2)
        if os.getppid() != parent_pid:
            os._exit(1)


def _child_main(conn, factory, nice: int, parent_pid: int) -> None:
    """The child process: load the models, say so (or why not), then answer calls until told to stop."""
    try:
        if nice:
            os.nice(nice)  # the web server and the GPU feed come first when the machine is busy
        models = factory()
    except BaseException as error:
        conn.send(("error", f"{type(error).__name__}: {error}"))
        return
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    conn.send(("ok", None))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        op, args = message
        try:
            if op == "ping":
                result = True
            elif op == "turns":
                audio, live, *rest = args
                result = models.turns(_open(audio), live, *rest)
            elif op == "voice":
                result = models.voice(_open(args[0]))
            else:
                raise ValueError(f"unknown call {op!r}")
            conn.send(("ok", result))
        except BaseException as error:  # the caller gets the reason; the process carries on
            conn.send(("error", f"{type(error).__name__}: {error}"))


# Audio of at least this many samples goes to a child through shared memory instead of a pipe:
# a recording of hours is hundreds of MB, and sending it would copy it twice over.
SHARE_FROM = RATE * 60


def _share(audio: np.ndarray):
    """The audio as something a child opens with `_open`: itself if small, else a shared block."""
    if len(audio) < SHARE_FROM:
        return audio, None
    block = shared_memory.SharedMemory(create=True, size=audio.nbytes)
    np.ndarray(audio.shape, audio.dtype, buffer=block.buf)[:] = audio
    return (block.name, audio.shape, audio.dtype.str), block


def _open(audio) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio
    name, shape, dtype = audio
    block = shared_memory.SharedMemory(name=name)
    try:
        return np.ndarray(shape, np.dtype(dtype), buffer=block.buf).copy()
    finally:
        block.close()


class ChildProcess:
    """One worker process that runs calls one at a time, with its models loaded at start.

    A call fails with `RuntimeError` if the process dies (killed for memory, say), and with
    `Cancelled` if its `cancel` event is set, which kills the process to stop the work. The next
    call starts a new one. After `close()` nothing is left running and nothing restarts.
    """

    def __init__(self, factory, name: str, nice: int = 0):
        self._factory, self._name, self._nice = factory, name, nice
        self._calls = threading.Lock()  # one call at a time
        self._state = threading.Lock()  # guards what follows, never held while waiting on the child
        self._process = None
        self._conn = None
        self._closed = False

    def _ensure(self):
        with self._state:
            if self._closed:
                raise RuntimeError(f"the {self._name} process is shut down")
            if self._process is not None and self._process.is_alive():
                return self._process, self._conn, False
            context = multiprocessing.get_context("spawn")  # not fork: threads, CUDA, an event loop here
            mine, theirs = context.Pipe()
            process = context.Process(
                target=_child_main, args=(theirs, self._factory, self._nice, os.getpid()), name=self._name, daemon=True
            )
            process.start()
            theirs.close()
            self._process, self._conn = process, mine
            return process, mine, True

    def _discard(self, process) -> None:
        """Stop `process` (if it still runs) and forget it."""
        conn = None
        with self._state:
            if self._process is process:
                self._process, conn, self._conn = None, self._conn, None
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        if conn is not None:
            conn.close()

    def _receive(self, process, conn, cancel):
        """The child's next message, polling so a `cancel` is noticed. A pipe that is closed or at
        its end means the process is gone (a dead process's end reads as EOF)."""
        while True:
            try:
                if conn.poll(0.2):
                    return conn.recv()
            except (EOFError, OSError):
                self._discard(process)
                with self._state:
                    closed = self._closed
                if closed:
                    raise RuntimeError(f"the {self._name} process is shut down") from None
                log.error("the %s process died (exit code %s)", self._name, process.exitcode)
                raise RuntimeError(f"the {self._name} process died") from None
            if cancel is not None and cancel.is_set():
                self._discard(process)
                raise Cancelled("cancelled")

    def _exchange(self, op: str, args: tuple, cancel):
        with self._calls:
            process, conn, fresh = self._ensure()
            if fresh:  # a new process answers once its models are loaded, or says why they would not
                kind, value = self._receive(process, conn, cancel)
                if kind == "error":
                    self._discard(process)
                    raise RuntimeError(f"the {self._name} process could not start: {value}")
            try:
                conn.send((op, args))
            except (OSError, ValueError):
                pass  # it died between the check and the send: _receive reports it
            kind, value = self._receive(process, conn, cancel)
            if kind == "error":
                raise RuntimeError(f"{self._name}: {value}")
            return value

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        """Start the process and wait until its models have loaded."""
        self._exchange("ping", (), None)

    def call(self, op: str, *args, cancel=None):
        return self._exchange(op, args, cancel)

    def close(self) -> None:
        with self._state:
            self._closed = True
            process = self._process
        if process is not None:
            self._discard(process)


class Diarizer:
    """sherpa-onnx diarization and speaker embeddings. Thread-safe: live calls (`turns`) and whole
    recordings (`whole`) use separate models and locks, so a long upload never holds up a call.

    `isolated` (the default) runs the models in child processes. sherpa-onnx holds Python's global
    lock for a whole call, so in this process one recording would freeze every request, the
    dashboard's included, until it was done.
    """

    def __init__(self, models_dir: Path, threads: int = 8, isolated: bool = True, factory=None):
        """`factory(role)` builds the models (a `Sherpa`); a test passes a stand-in."""
        if factory is None:
            fetch_models(models_dir)  # once, here, not in both children at the same time
            factory = partial(Sherpa, models_dir, threads)
        self._sherpa = None
        self._whole_process = self._live_process = None
        if isolated:
            # Recordings run niced: the web server and live calls come first when the machine is busy.
            self._whole_process = ChildProcess(partial(factory, "recordings"), "speaker separation (recordings)", nice=10)
            self._live_process = ChildProcess(partial(factory, "calls"), "speaker separation (calls)")
            try:
                for process in (self._whole_process, self._live_process):
                    process.start()
            except BaseException:
                self.close()  # not the first child left running with nobody to stop it
                raise
        else:
            self._sherpa = factory("both")
        self._live_lock = threading.Lock()
        self._whole_lock = threading.Lock()

    def close(self) -> None:
        """Stop the child processes, busy or not. Nothing restarts them."""
        for process in (self._whole_process, self._live_process):
            if process is not None:
                process.close()

    def _call(self, live: bool, local, op: str, audio: np.ndarray, *args, cancel=None):
        if self._sherpa is not None:
            return local(audio)
        shared, block = _share(audio)
        try:
            process = self._live_process if live else self._whole_process
            return process.call(op, shared, *args, cancel=cancel)
        finally:
            if block is not None:
                block.close()
                block.unlink()

    def _turns(self, audio: np.ndarray, live: bool = False, threshold: float | None = None, cancel=None) -> list[Turn]:
        raw = self._call(live, lambda a: self._sherpa.turns(a, live, threshold), "turns", audio, live, threshold, cancel=cancel)
        return tidy([Turn(start, end, label) for start, end, label in raw])

    def voice(self, audio: np.ndarray, live: bool = True, cancel=None) -> np.ndarray:
        return self._call(live, self._sherpa.voice if self._sherpa is not None else None, "voice", audio, cancel=cancel)

    def turns(self, audio: np.ndarray, book: SpeakerBook) -> list[tuple[Turn, Speaker]]:
        """Split one utterance into turns and name each turn's speaker from the session's book.

        A turn long enough to carry a voice is matched on its own: the book's averages, built over
        the whole call, are a better judge than one utterance's clustering, which can lump a short
        turn in with the next speaker. Shorter turns are pooled by clusterer label (short turns
        only, so a long turn of someone else under the same label cannot outvote them) and matched
        together.
        """
        with self._live_lock:
            turns = self._turns(audio, live=True)
            clip = lambda ts: np.concatenate([audio[int(t.start * RATE) : int(t.end * RATE)] for t in ts])  # noqa: E731
            by_label: dict[int, Speaker] = {}
            out: list[tuple[Turn, Speaker]] = []
            for turn in turns:
                if turn.duration >= MIN_EMBED_S:
                    speaker = book.match(self.voice(clip([turn])), turn.duration)
                else:
                    if turn.label not in by_label:
                        group = clip([t for t in turns if t.label == turn.label and t.duration < MIN_EMBED_S])
                        by_label[turn.label] = book.match(self.voice(group), len(group) / RATE)
                    speaker = by_label[turn.label]
                out.append((turn, speaker))
            return _merge_same_speaker(out)

    def whole(self, audio: np.ndarray, num_speakers: int | None = None, cancel=None) -> list[tuple[Turn, int]]:
        """Turns of a whole recording, speakers numbered 1.. in order of first appearance.

        The clusterer finds as many speakers as it likes and the scraps among them join the real
        ones (`absorb_scraps`). `num_speakers`, if the caller knows it, then merges the closest
        voices until that many remain (`merge_down`; sherpa's own fixed-count clustering gave one
        huge speaker and one of seconds on a real podcast). Setting the `cancel` event (a threading.Event) stops the work, and frees
        the CPU it was using.
        """
        with self._whole_lock:
            voice = lambda clip: self.voice(clip, live=False, cancel=cancel)  # noqa: E731
            turns = self._turns(audio, cancel=cancel)
            if num_speakers and len({t.label for t in turns}) < num_speakers:
                # Fewer than said: cluster again more finely (more speakers than that, then merged down).
                turns = self._turns(audio, threshold=WHOLE_SPLIT_THRESHOLD, cancel=cancel)
            turns = absorb_scraps(turns, audio, voice)
            if num_speakers:
                turns = merge_down(turns, audio, voice, num_speakers)
        numbers: dict[int, int] = {}
        return [(t, numbers.setdefault(t.label, len(numbers) + 1)) for t in turns]


def speaker_voice(turns: list[Turn], audio: np.ndarray, label: int, voice) -> np.ndarray | None:
    """One speaker's unit-length voice, measured on up to EMBED_CAP_S of their longest turns, or None
    if there is under half a second of them or the measurement fails."""
    clips, got = [], 0.0
    for turn in sorted((t for t in turns if t.label == label), key=lambda t: -t.duration):
        if got >= EMBED_CAP_S:
            break
        clips.append(audio[int(turn.start * RATE) : int(turn.end * RATE)])
        got += turn.duration
    clip = np.concatenate(clips) if clips else np.zeros(0, np.float32)
    if len(clip) < RATE // 2:
        return None
    try:
        vector = np.asarray(voice(clip), dtype=np.float32)
    except Exception:  # a voice that cannot be measured is placed by time instead
        log.exception("measuring a speaker's voice failed")
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else None


def _nearest_in_time(turn: Turn, among: list[Turn]) -> int:
    return min(among, key=lambda k: max(k.start - turn.end, turn.start - k.end, 0.0)).label


def absorb_scraps(turns: list[Turn], audio: np.ndarray, voice) -> list[Turn]:
    """Fold the clusterer's scrap speakers (see SCRAP_S) into the speakers they sound like.

    `voice(clip)` is a speaker embedding. A scrap whose voice cannot be measured (under half a
    second of audio) goes to whoever is speaking nearest to it in time.
    """
    seconds: dict[int, float] = {}
    for turn in turns:
        seconds[turn.label] = seconds.get(turn.label, 0.0) + turn.duration
    if len(seconds) < 2:
        return turns
    total = sum(seconds.values())
    keep = {label for label, sec in seconds.items() if sec / total >= SCRAP_SHARE}
    keep.add(max(seconds, key=lambda label: seconds[label]))  # somebody has to be the speaker
    scraps = [label for label in seconds if label not in keep]
    if not scraps:
        return turns

    known = {label: v for label in keep if (v := speaker_voice(turns, audio, label, voice)) is not None}
    kept_turns = [t for t in turns if t.label in keep]
    move: dict[int, int] = {}
    for label in scraps:
        mine = speaker_voice(turns, audio, label, voice)
        if mine is None or not known:
            move[label] = -1  # by time, turn by turn
            continue
        best = max(known, key=lambda k: float(mine @ known[k]))
        if seconds[label] < SCRAP_S or float(mine @ known[best]) >= MATCH:
            move[label] = best

    def relabelled(turn: Turn) -> int:
        target = move.get(turn.label, turn.label)
        return _nearest_in_time(turn, kept_turns) if target == -1 else target

    out = [Turn(t.start, t.end, relabelled(t)) for t in turns]
    log.info("speakers: %d found, %d after folding in scraps", len(seconds), len({t.label for t in out}))
    return tidy(out)


def merge_down(turns: list[Turn], audio: np.ndarray, voice, count: int) -> list[Turn]:
    """Merge the two speakers who sound most alike, again and again, until `count` remain. A speaker
    whose voice cannot be measured goes to whoever speaks nearest in time."""
    seconds: dict[int, float] = {}
    for turn in turns:
        seconds[turn.label] = seconds.get(turn.label, 0.0) + turn.duration
    if len(seconds) <= count:
        return turns
    voices = {label: speaker_voice(turns, audio, label, voice) for label in seconds}
    measured = {label: v for label, v in voices.items() if v is not None}
    groups: dict[int, set[int]] = {label: {label} for label in measured}  # merged label -> its parts
    weight = {label: seconds[label] for label in measured}
    centre = dict(measured)
    while len(groups) > max(count, 1):
        a, b = max(
            ((x, y) for x in groups for y in groups if x < y), key=lambda pair: float(centre[pair[0]] @ centre[pair[1]])
        )
        mixed = centre[a] * weight[a] + centre[b] * weight[b]
        centre[a] = mixed / (np.linalg.norm(mixed) or 1.0)
        weight[a] += weight[b]
        groups[a] |= groups.pop(b)
        del centre[b], weight[b]
    owner = {part: label for label, parts in groups.items() for part in parts}
    placed = [Turn(t.start, t.end, owner[t.label]) for t in turns if t.label in owner]
    out = [
        Turn(t.start, t.end, owner[t.label] if t.label in owner else _nearest_in_time(t, placed)) for t in turns
    ] if placed else turns
    log.info("speakers: merged %d down to %d as asked", len(seconds), len({t.label for t in out}))
    return tidy(out)


def _merge_same_speaker(pairs: list[tuple[Turn, Speaker]]) -> list[tuple[Turn, Speaker]]:
    """Two local labels can map to one known speaker; their adjacent turns are then one turn."""
    merged: list[tuple[Turn, Speaker]] = []
    for turn, speaker in pairs:
        if merged and merged[-1][1] is speaker:
            merged[-1][0].end = max(merged[-1][0].end, turn.end)
        else:
            merged.append((turn, speaker))
    return merged


class DiarizerSlot:
    """Holds the diarizer once it has loaded (its models download on first start, ~45 MB).

    Until it is ready, or if it fails, mixed channels are transcribed as one speaker.
    """

    def __init__(self, diarizer=None, status: str | None = None):
        self.value = diarizer
        self.status = status or ("ready" if diarizer is not None else "off")

    def load(self, factory) -> None:
        self.status = "loading"

        def run():
            try:
                self.value = factory()
                self.status = "ready"
            except Exception as error:  # surfaced in /api/status; transcription carries on
                log.exception("diarization unavailable")
                self.status = f"failed: {error}"

        threading.Thread(target=run, name="diarizer-load", daemon=True).start()

    def ready(self):
        return self.value if self.status == "ready" else None
