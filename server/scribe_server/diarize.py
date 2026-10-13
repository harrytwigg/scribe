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
import os
import tarfile
import threading
import urllib.request
from dataclasses import dataclass, field
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


class Diarizer:
    """sherpa-onnx diarization and speaker embeddings. Thread-safe: live calls (`turns`) and whole
    recordings (`whole`) use separate models and locks, so a long upload never holds up a call."""

    def __init__(self, models_dir: Path, threads: int = 8):
        import sherpa_onnx

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

        self._sherpa = sherpa_onnx
        self._embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=threads)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self._embedding_config)

        def diarizer(threshold: float):
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                    num_threads=threads,
                ),
                embedding=self._embedding_config,
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not config.validate():
                raise RuntimeError("invalid diarization config")
            return sherpa_onnx.OfflineSpeakerDiarization(config)

        # Its clustering threshold is a distance: higher merges more. For a whole recording, 0.7-0.9
        # found the right speaker count on the test conversation (0.5 split speakers). For live
        # utterances it is set to over-split instead: the session's book merges the pieces back,
        # while a merge here (a short turn lumped in with the next speaker) cannot be undone.
        self._whole = diarizer(0.8)
        self._live = diarizer(float(os.environ.get("SCRIBE_DIARIZE_LIVE_THRESHOLD", "0.3")))
        self._live_lock = threading.Lock()
        self._whole_lock = threading.Lock()

    def _turns(self, audio: np.ndarray, live: bool = False) -> list[Turn]:
        result = (self._live if live else self._whole).process(audio).sort_by_start_time()
        return tidy([Turn(s.start, s.end, s.speaker) for s in result])

    def voice(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(RATE, audio)
        stream.input_finished()
        return np.array(self._extractor.compute(stream), dtype=np.float32)

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

    def whole(self, audio: np.ndarray) -> list[tuple[Turn, int]]:
        """Turns of a whole recording, speakers numbered 1.. in order of first appearance."""
        with self._whole_lock:
            turns = self._turns(audio)
        numbers: dict[int, int] = {}
        return [(t, numbers.setdefault(t.label, len(numbers) + 1)) for t in turns]


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
