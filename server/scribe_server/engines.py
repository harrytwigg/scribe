"""ASR engine adapters: load a model directory, transcribe 16 kHz mono float32, unload."""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .catalogue import CatalogueEntry


@dataclass
class Word:
    text: str  # as it joins into the transcript: a word starts with a space, punctuation does not
    start: float  # seconds into the audio given
    end: float


class Engine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...

    # Optional: the same transcript with word timings, so a diarized utterance is transcribed once
    # and its words given to speakers (docs/spec.md, 3.8). None when the model has no timings.
    # def transcribe_words(self, audio: np.ndarray) -> list[Word] | None: ...

    def close(self) -> None: ...


class FasterWhisperEngine:
    def __init__(self, entry: CatalogueEntry, path: Path, device: str):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(str(path), device=device, compute_type=compute_type)
        self._language = entry.language

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
            # Each utterance stands alone; carrying text over invites repetition loops.
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_words(self, audio: np.ndarray) -> list[Word] | None:
        segments, _ = self._model.transcribe(
            audio, language=self._language, beam_size=5, vad_filter=True,
            condition_on_previous_text=False, word_timestamps=True,
        )
        return [Word(w.word, w.start, w.end) for segment in segments for w in (segment.words or [])]

    def close(self) -> None:
        del self._model
        gc.collect()


def words_from_tokens(tokens: list[str], starts: list[float], end: float) -> list[Word]:
    """Subword tokens with start times (onnx-asr) to words: a token beginning with a space starts
    a word; anything else (the rest of a word, punctuation) joins the word before."""
    words: list[Word] = []
    for token, start in zip(tokens, starts):
        if words and not token.startswith(" "):
            words[-1].text += token
        else:
            if words:
                words[-1].end = start
            words.append(Word(token, start, end))
    return words


class OnnxAsrEngine:
    def __init__(self, entry: CatalogueEntry, path: Path, device: str):
        import onnx_asr

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self._model = onnx_asr.load_model(entry.onnx_type, str(path), providers=providers)
        self._timed = self._model.with_timestamps()

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._model.recognize(audio, sample_rate=16_000)
        return str(result).strip()

    def transcribe_words(self, audio: np.ndarray) -> list[Word] | None:
        result = self._timed.recognize(audio, sample_rate=16_000)
        tokens, starts = result.tokens or [], result.timestamps or []
        if tokens and len(starts) != len(tokens):
            return None  # this model gives no timings (Canary)
        return words_from_tokens(tokens, starts, len(audio) / 16_000)

    def close(self) -> None:
        del self._model, self._timed
        gc.collect()


def load_engine(entry: CatalogueEntry, path: Path, device: str) -> Engine:
    if entry.engine == "faster-whisper":
        return FasterWhisperEngine(entry, path, device)
    if entry.engine == "onnx-asr":
        return OnnxAsrEngine(entry, path, device)
    raise ValueError(f"unknown engine {entry.engine}")
