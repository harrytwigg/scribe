"""Decode uploaded audio to 16 kHz mono float32, and split it into speech chunks for ASR."""

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile
import soxr

RATE = 16_000
# Parakeet in onnx-asr handles ~20-30 s per call (docs/spec.md, 3.3); every input is cut to this.
MAX_CHUNK_S = 25.0
# Less speech than this is noise (keyboard, breath): no model call at all.
MIN_SPEECH_S = 0.25


class AudioError(ValueError):
    pass


def decode(source: bytes | Path) -> np.ndarray:
    """Any format libsndfile reads (WAV, FLAC, OGG/Opus, MP3) to 16 kHz mono float32, and
    otherwise (M4A/AAC, video containers: many podcast feeds) anything FFmpeg reads, via PyAV.
    A path is read in blocks, so a long recording at a high rate and several channels never sits
    in memory at its source size."""
    try:
        return _decode_sndfile(source)
    except Exception as error:  # soundfile raises a bare LibsndfileError/RuntimeError
        return _decode_av(source, error)


def _decode_sndfile(source: bytes | Path) -> np.ndarray:
    opened = soundfile.SoundFile(io.BytesIO(source) if isinstance(source, bytes) else str(source))
    with opened as audio:
        rate = audio.samplerate
        resampler = soxr.ResampleStream(rate, RATE, 1, dtype="float32") if rate != RATE else None
        parts = [
            (resampler.resample_chunk(mono) if resampler else mono)
            for mono in (block.mean(axis=1) for block in audio.blocks(blocksize=rate * 30, dtype="float32", always_2d=True))
        ]
        if resampler:
            parts.append(resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
    if not parts:
        raise AudioError("cannot decode audio: it has no samples")
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def _decode_av(source: bytes | Path, first_error: Exception) -> np.ndarray:
    import av  # faster-whisper's dependency; imported here, it is slow to load

    try:
        with av.open(io.BytesIO(source) if isinstance(source, bytes) else str(source)) as container:
            if not container.streams.audio:
                raise AudioError("cannot decode audio: no audio stream")
            # Planar, the source's own channels: they are averaged below as in the main path
            # (letting FFmpeg downmix would change the level).
            resampler = av.AudioResampler(format="fltp", rate=RATE)
            parts = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    parts.append(out.to_ndarray().mean(axis=0))
            for out in resampler.resample(None):
                parts.append(out.to_ndarray().mean(axis=0))
    except AudioError:
        raise
    except Exception as error:
        raise AudioError(f"cannot decode audio: {first_error}") from error
    if not parts:
        raise AudioError("cannot decode audio: it has no samples")
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


# A VAD maps 16 kHz audio to speech spans in samples, each at most MAX_CHUNK_S long.
Vad = Callable[[np.ndarray], list[tuple[int, int]]]


def speech_spans(audio: np.ndarray, vad: Vad) -> list[tuple[int, int]]:
    """Where the speech in `audio` is, in samples, grouped into chunks of at most MAX_CHUNK_S.

    Neighbouring spans are merged while the merged chunk still fits, so a short utterance stays
    one call with its pauses (context helps the model) and a long recording becomes a few.
    """
    spans = [span for span in vad(audio) if span[1] > span[0]]
    if sum(end - start for start, end in spans) < MIN_SPEECH_S * RATE:
        return []
    limit = int(MAX_CHUNK_S * RATE)
    chunks: list[tuple[int, int]] = []
    for start, end in spans:
        if chunks and end - chunks[-1][0] <= limit:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


def speech_chunks(audio: np.ndarray, vad: Vad) -> list[np.ndarray]:
    return [audio[start:end] for start, end in speech_spans(audio, vad)]


def wav_bytes(audio: np.ndarray, rate: int = RATE) -> bytes:
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
