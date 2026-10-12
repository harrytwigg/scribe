"""Silero voice activity detection, via onnx-asr, on the CPU."""

from pathlib import Path

import numpy as np

from .audio import MAX_CHUNK_S, RATE

REPO = "istupakov/silero-vad-onnx"


class SileroVad:
    def __init__(self, models_dir: Path):
        import onnx_asr
        from huggingface_hub import snapshot_download

        path = models_dir / "_silero-vad"
        if not (path / "silero_vad.onnx").exists():
            # 2 MB; a plain directory, not the HF cache (docs/spec.md, section 2).
            snapshot_download(REPO, local_dir=path, allow_patterns=["silero_vad.onnx", "config.json"])
        self._vad = onnx_asr.load_vad("silero", str(path), providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray) -> list[tuple[int, int]]:
        if len(audio) == 0:
            return []
        (segments,) = self._vad.segment_batch(
            audio[None, :],
            np.array([len(audio)], dtype=np.int64),
            RATE,
            max_speech_duration_s=MAX_CHUNK_S,
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        )
        return [(int(start), int(end)) for start, end in segments]
