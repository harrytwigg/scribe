"""The models scribe offers: what the GUI's model picker lists (docs/spec.md, 3.3 and 5).

An entry ships here only once the GPU test (`tests/test_gpu.py`) has transcribed the reference
clip with it.
"""

from dataclasses import asdict, dataclass, field
from typing import Literal

Engine = Literal["faster-whisper", "onnx-asr"]

# Parakeet and Canary repos also carry int8 copies of every graph; on a GPU we load fp32 only.
_ONNX_FP32 = ["*.json", "*.txt", "*.onnx", "*.onnx.data", "!*int8*"]


@dataclass(frozen=True)
class CatalogueEntry:
    id: str
    name: str
    engine: Engine
    repo: str
    description: str
    languages: str
    licence: str
    size_mb: int
    # 1 (worst) to 5 (best); relative hints for the picker, from the Open ASR leaderboard.
    accuracy: int
    speed: int
    recommended: bool = False
    # onnx-asr's model type name; unused by faster-whisper.
    onnx_type: str | None = None
    # Whisper `.en` models are English-only; None lets Whisper detect the language.
    language: str | None = None
    # Glob patterns (fnmatch); a leading `!` excludes.
    files: list[str] = field(default_factory=list)

    def public(self) -> dict:
        data = asdict(self)
        for key in ("files", "onnx_type", "repo"):
            data.pop(key)
        data["source"] = f"https://huggingface.co/{self.repo}"
        return data


CATALOGUE: list[CatalogueEntry] = [
    CatalogueEntry(
        id="parakeet-tdt-0.6b-v2",
        name="Parakeet TDT 0.6B v2",
        engine="onnx-asr",
        repo="istupakov/parakeet-tdt-0.6b-v2-onnx",
        onnx_type="nemo-parakeet-tdt-0.6b-v2",
        description="NVIDIA's English model. Most accurate and by far the fastest here.",
        languages="English",
        licence="CC-BY-4.0",
        size_mb=2490,
        accuracy=5,
        speed=5,
        recommended=True,
        files=_ONNX_FP32,
    ),
    CatalogueEntry(
        id="parakeet-tdt-0.6b-v3",
        name="Parakeet TDT 0.6B v3",
        engine="onnx-asr",
        repo="istupakov/parakeet-tdt-0.6b-v3-onnx",
        onnx_type="nemo-parakeet-tdt-0.6b-v3",
        description="Multilingual Parakeet: 25 European languages, detected automatically.",
        languages="25 European languages",
        licence="CC-BY-4.0",
        size_mb=2550,
        accuracy=5,
        speed=5,
        files=_ONNX_FP32,
    ),
    CatalogueEntry(
        id="canary-1b-v2",
        name="Canary 1B v2",
        engine="onnx-asr",
        repo="istupakov/canary-1b-v2-onnx",
        onnx_type="nemo-canary-1b-v2",
        description="NVIDIA's larger multilingual model. Slower than Parakeet.",
        languages="25 European languages",
        licence="CC-BY-4.0",
        size_mb=3970,
        accuracy=4,
        speed=3,
        files=_ONNX_FP32,
    ),
    CatalogueEntry(
        id="whisper-large-v3-turbo",
        name="Whisper large-v3 turbo",
        engine="faster-whisper",
        repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        description="OpenAI Whisper, pruned decoder. A robust all-rounder in 99 languages.",
        languages="99 languages",
        licence="MIT",
        size_mb=1620,
        accuracy=4,
        speed=3,
        files=["*"],
    ),
    CatalogueEntry(
        id="whisper-large-v3",
        name="Whisper large-v3",
        engine="faster-whisper",
        repo="Systran/faster-whisper-large-v3",
        description="The full Whisper model. Slowest here; best Whisper on hard audio.",
        languages="99 languages",
        licence="MIT",
        size_mb=3090,
        accuracy=4,
        speed=2,
        files=["*"],
    ),
    CatalogueEntry(
        id="distil-large-v3.5",
        name="Distil-Whisper large-v3.5",
        engine="faster-whisper",
        repo="distil-whisper/distil-large-v3.5-ct2",
        description="Distilled Whisper: Whisper's behaviour, faster. English.",
        languages="English",
        licence="MIT",
        size_mb=1520,
        accuracy=4,
        speed=4,
        language="en",
        files=["*.json", "*.bin", "!.ipynb_checkpoints/*"],
    ),
    CatalogueEntry(
        id="whisper-small.en",
        name="Whisper small.en",
        engine="faster-whisper",
        repo="Systran/faster-whisper-small.en",
        description="Small and light. Use when the GPU is busy or for a quick check.",
        languages="English",
        licence="MIT",
        size_mb=480,
        accuracy=2,
        speed=4,
        language="en",
        files=["*"],
    ),
    CatalogueEntry(
        id="whisper-tiny.en",
        name="Whisper tiny.en",
        engine="faster-whisper",
        repo="Systran/faster-whisper-tiny.en",
        description="Tiny: least accurate, runs anywhere, even on a CPU.",
        languages="English",
        licence="MIT",
        size_mb=75,
        accuracy=1,
        speed=5,
        language="en",
        files=["*"],
    ),
]


def by_id(catalogue: list[CatalogueEntry]) -> dict[str, CatalogueEntry]:
    return {entry.id: entry for entry in catalogue}
