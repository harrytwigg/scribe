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
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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
# Whole recordings: a speaker the clusterer found who spoke for less than SCRAP_S seconds, or less
# than SCRAP_SHARE of the speech, is usually a scrap of someone else (a voice over music, a laugh,
# the same person on a bad line), not a person. Under SCRAP_S it always joins the speaker it sounds
# most like; between that and SCRAP_SHARE only if it sounds like them (MATCH), so a real third voice
# who spoke for a minute stays. EMBED_CAP_S bounds the audio each speaker's voice is measured on.
SCRAP_S = float(os.environ.get("SCRIBE_DIARIZE_SCRAP_S", "15"))
SCRAP_SHARE = float(os.environ.get("SCRIBE_DIARIZE_SCRAP_SHARE", "0.05"))
EMBED_CAP_S = 45.0
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

    def __init__(self, models_dir: Path, threads: int = 8):
        import sherpa_onnx

        folder = fetch_models(models_dir)
        segmentation = folder / "pyannote-segmentation-3.0.onnx"
        embedding = folder / "nemo_en_titanet_small.onnx"
        self._embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=threads)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self._embedding_config)

        def config(threshold: float, clusters: int = -1):
            made = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                    num_threads=threads,
                ),
                embedding=self._embedding_config,
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=clusters, threshold=threshold),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not made.validate():
                raise RuntimeError("invalid diarization config")
            return made

        self._config = config
        self._clusters = -1  # what the whole-recording diarizer is currently set to find

        # Its clustering threshold is a distance: higher merges more. For a whole recording, 0.7-0.9
        # found the right speaker count on the test conversation (0.5 split speakers). For live
        # utterances it is set to over-split instead: the session's book merges the pieces back,
        # while a merge here (a short turn lumped in with the next speaker) cannot be undone.
        self._whole = sherpa_onnx.OfflineSpeakerDiarization(config(0.8))
        self._live = sherpa_onnx.OfflineSpeakerDiarization(config(float(os.environ.get("SCRIBE_DIARIZE_LIVE_THRESHOLD", "0.3"))))

    def turns(self, audio: np.ndarray, live: bool, num_speakers: int | None = None) -> list[tuple[float, float, int]]:
        """`num_speakers` (whole recordings only): cluster to exactly this many speakers."""
        if not live and (clusters := num_speakers or -1) != self._clusters:
            self._whole.set_config(self._config(0.8, clusters))
            self._clusters = clusters
        result = (self._live if live else self._whole).process(audio).sort_by_start_time()
        return [(s.start, s.end, s.speaker) for s in result]

    def voice(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(RATE, audio)
        stream.input_finished()
        return np.array(self._extractor.compute(stream), dtype=np.float32)


# -- child processes -----------------------------------------------------------------------------
# One for live calls and one for whole recordings, as the two used to have their own models and
# locks: a long upload never holds up a call. Each loads its own models once, at start.

_child: object | None = None  # in a child process: the models (a `Sherpa`, or a test's stand-in)


def _child_start(factory, nice: int) -> None:
    global _child
    os.nice(nice)  # the web server and the GPU feed come first when the machine is busy
    _child = factory()


def _child_turns(audio, live: bool, num_speakers: int | None = None):
    return _child.turns(_open(audio), live, num_speakers)  # type: ignore[union-attr]


def _child_voice(audio):
    return _child.voice(_open(audio))  # type: ignore[union-attr]


def _child_ping() -> bool:
    return _child is not None


# Audio of at least this many samples goes to a child through shared memory instead of a pickle:
# a recording of hours is hundreds of MB, and pickling would copy it twice over.
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
    """One worker process that runs `fn` calls one at a time, with its models loaded at start.

    If it dies (killed for memory, say) the call in flight fails with `RuntimeError`, and the next
    call starts a new one rather than leaving every later call broken.
    """

    def __init__(self, factory, name: str, nice: int = 10):
        self._factory, self._name, self._nice = factory, name, nice
        self._lock = threading.Lock()
        self._pool: ProcessPoolExecutor | None = None

    def _ensure(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._pool is None:
                # spawn, not fork: this process has threads, CUDA and an event loop.
                self._pool = ProcessPoolExecutor(
                    1,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_child_start,
                    initargs=(self._factory, self._nice),
                )
            return self._pool

    def start(self) -> None:
        """Start the process and wait until its models have loaded."""
        self.call(_child_ping)

    def call(self, fn, *args):
        pool = self._ensure()
        try:
            return pool.submit(fn, *args).result()
        except BrokenProcessPool as error:
            with self._lock:
                if self._pool is pool:
                    self._pool = None
            pool.shutdown(wait=False, cancel_futures=True)
            log.error("the %s process died: %s", self._name, error)
            raise RuntimeError(f"the {self._name} process died") from error

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


class Diarizer:
    """sherpa-onnx diarization and speaker embeddings. Thread-safe: live calls (`turns`) and whole
    recordings (`whole`) use separate models and locks, so a long upload never holds up a call.

    `isolated` (the default) runs the models in child processes. sherpa-onnx holds Python's global
    lock for a whole call, so in this process one recording would freeze every request, the
    dashboard's included, until it was done.
    """

    def __init__(self, models_dir: Path, threads: int = 8, isolated: bool = True, factory=None):
        if factory is None:
            fetch_models(models_dir)  # once, here, not in both children at the same time
            factory = partial(Sherpa, models_dir, threads)
        if isolated:
            self._sherpa = None
            self._whole_process = ChildProcess(factory, "speaker separation (recordings)")
            self._live_process = ChildProcess(factory, "speaker separation (calls)")
            for process in (self._whole_process, self._live_process):
                process.start()
        else:
            self._sherpa = factory()
        self._live_lock = threading.Lock()
        self._whole_lock = threading.Lock()

    def close(self) -> None:
        if self._sherpa is None:
            self._whole_process.close()
            self._live_process.close()

    def _call(self, live: bool, local, remote, audio: np.ndarray):
        if self._sherpa is not None:
            return local(audio)
        shared, block = _share(audio)
        try:
            return (self._live_process if live else self._whole_process).call(remote, shared)
        finally:
            if block is not None:
                block.close()
                block.unlink()

    def _turns(self, audio: np.ndarray, live: bool = False, num_speakers: int | None = None) -> list[Turn]:
        raw = self._call(
            live,
            lambda a: self._sherpa.turns(a, live, num_speakers),
            partial(_child_turns, live=live, num_speakers=num_speakers),
            audio,
        )
        return tidy([Turn(start, end, label) for start, end, label in raw])

    def voice(self, audio: np.ndarray, live: bool = True) -> np.ndarray:
        return self._call(live, self._sherpa.voice if self._sherpa is not None else None, _child_voice, audio)

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

    def whole(self, audio: np.ndarray, num_speakers: int | None = None) -> list[tuple[Turn, int]]:
        """Turns of a whole recording, speakers numbered 1.. in order of first appearance.

        `num_speakers`, if the caller knows it, makes the clusterer find exactly that many.
        Otherwise it finds as many as it likes, and the scraps among them join the real speakers
        (`absorb_scraps`).
        """
        with self._whole_lock:
            turns = self._turns(audio, num_speakers=num_speakers)
            if num_speakers is None:
                turns = absorb_scraps(turns, audio, lambda clip: self.voice(clip, live=False))
        numbers: dict[int, int] = {}
        return [(t, numbers.setdefault(t.label, len(numbers) + 1)) for t in turns]


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
    keep = {label for label, sec in seconds.items() if sec >= SCRAP_S and sec / total >= SCRAP_SHARE}
    keep.add(max(seconds, key=lambda label: seconds[label]))  # somebody has to be the speaker
    scraps = [label for label in seconds if label not in keep]
    if not scraps:
        return turns

    def centroid(label: int) -> np.ndarray | None:
        clips, got = [], 0.0
        for turn in sorted((t for t in turns if t.label == label), key=lambda t: -t.duration):
            if got >= EMBED_CAP_S:
                break
            clips.append(audio[int(turn.start * RATE) : int(turn.end * RATE)])
            got += turn.duration
        clip = np.concatenate(clips)
        if len(clip) < RATE // 2:
            return None
        try:
            vector = np.asarray(voice(clip), dtype=np.float32)
        except Exception:  # a voice that cannot be measured is placed by time instead
            log.exception("measuring a speaker's voice failed")
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else None

    known = {label: c for label in keep if (c := centroid(label)) is not None}
    kept_turns = [t for t in turns if t.label in keep]

    def nearest_in_time(turn: Turn) -> int:
        return min(kept_turns, key=lambda k: max(k.start - turn.end, turn.start - k.end, 0.0)).label

    move: dict[int, int] = {}
    for label in scraps:
        mine = centroid(label)
        if mine is None or not known:
            move[label] = -1  # by time, turn by turn
            continue
        best = max(known, key=lambda k: float(mine @ known[k]))
        if seconds[label] < SCRAP_S or float(mine @ known[best]) >= MATCH:
            move[label] = best

    def relabelled(turn: Turn) -> int:
        target = move.get(turn.label, turn.label)
        return nearest_in_time(turn) if target == -1 else target

    out = [Turn(t.start, t.end, relabelled(t)) for t in turns]
    log.info("speakers: %d found, %d after folding in scraps", len(seconds), len({t.label for t in out}))
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
