"""Decode uploaded audio to 16 kHz mono float32, and split it into speech chunks for ASR."""

import io
from collections.abc import Callable

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


def decode(data: bytes) -> np.ndarray:
    """Any format libsndfile reads (WAV, FLAC, OGG/Opus, MP3) to 16 kHz mono float32."""
    try:
        audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as error:  # soundfile raises a bare LibsndfileError/RuntimeError
        raise AudioError(f"cannot decode audio: {error}") from error
    mono = audio.mean(axis=1)
    if rate != RATE:
        mono = soxr.resample(mono, rate, RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


# A VAD maps 16 kHz audio to speech spans in samples, each at most MAX_CHUNK_S long.
Vad = Callable[[np.ndarray], list[tuple[int, int]]]


def speech_chunks(audio: np.ndarray, vad: Vad) -> list[np.ndarray]:
    """Speech in `audio`, grouped into chunks of at most MAX_CHUNK_S.

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
    return [audio[start:end] for start, end in chunks]


def wav_bytes(audio: np.ndarray, rate: int = RATE) -> bytes:
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
