"""ASR engine adapters: load a model directory, transcribe 16 kHz mono float32, unload."""

import gc
from pathlib import Path
from typing import Protocol

import numpy as np

from .catalogue import CatalogueEntry


class Engine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...

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

    def close(self) -> None:
        del self._model
        gc.collect()


class OnnxAsrEngine:
    def __init__(self, entry: CatalogueEntry, path: Path, device: str):
        import onnx_asr

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self._model = onnx_asr.load_model(entry.onnx_type, str(path), providers=providers)

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._model.recognize(audio, sample_rate=16_000)
        return str(result).strip()

    def close(self) -> None:
        del self._model
        gc.collect()


def load_engine(entry: CatalogueEntry, path: Path, device: str) -> Engine:
    if entry.engine == "faster-whisper":
        return FasterWhisperEngine(entry, path, device)
    if entry.engine == "onnx-asr":
        return OnnxAsrEngine(entry, path, device)
    raise ValueError(f"unknown engine {entry.engine}")
