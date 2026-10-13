"""Stress-test live diarization: four male LibriSpeech speakers taking 16 turns, cut into
utterances every 12 s and 25 s as the extension would, scored per turn (docs/spec.md, 3.8).

    curl -L -o /tmp/test-clean.parquet \
      https://huggingface.co/api/datasets/openslr/librispeech_asr/parquet/clean/test/0.parquet  # 350 MB
    uv run --with pyarrow python scripts/diarize_stress.py /tmp/test-clean.parquet
"""

import io
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pyarrow.parquet as pq
import soundfile

from scribe_server.diarize import RATE, Diarizer, SpeakerBook

SPEAKERS = [7176, 7729, 8230, 260]  # all male: the harder case


def conversation(parquet: str) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    rows = pq.read_table(parquet, columns=["speaker_id", "id", "audio"]).to_pylist()
    rng = np.random.default_rng(1)
    pool = {k: sorted((r for r in rows if r["speaker_id"] == k), key=lambda r: r["id"]) for k in SPEAKERS}
    used = dict.fromkeys(SPEAKERS, 0)
    audio, truth, t = [], [], 0.0
    for speaker in rng.choice(SPEAKERS, 16):
        while True:
            row = pool[speaker][used[speaker]]
            used[speaker] += 1
            x, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if 2 <= len(x) / rate <= 7:
                break
        gap = rng.uniform(0.2, 0.6)
        truth.append((t, t + len(x) / rate, int(speaker)))
        audio += [x, np.zeros(int(gap * rate), np.float32)]
        t += len(x) / rate + gap
    return np.concatenate(audio), truth


def main(parquet: str) -> None:
    onnxruntime.preload_dlls()
    audio, truth = conversation(parquet)
    diarizer = Diarizer(Path("~/.local/share/scribe/models").expanduser())
    for cut_s in (12, 25):
        book, lines = SpeakerBook(), []
        for start in range(0, len(audio), cut_s * RATE):
            for turn, speaker in diarizer.turns(audio[start : start + cut_s * RATE], book):
                lines.append((turn.start + start / RATE, turn.end + start / RATE, speaker.number))
        labels: dict[int, list] = {}
        for begin, end, who in truth:
            overlap: dict[int, float] = {}
            for a, b, number in lines:
                if (o := min(end, b) - max(begin, a)) > 0:
                    overlap[number] = overlap.get(number, 0) + o
            labels.setdefault(who, []).append(max(overlap, key=overlap.get) if overlap else None)
        # Each true speaker scored against the label they got most often.
        correct = sum(max(found.count(x) for x in set(found)) for found in labels.values())
        print(f"cut {cut_s}s: {len(truth)} turns, {len(book.speakers)} speakers found, turn accuracy {correct / len(truth):.2f}", labels)


if __name__ == "__main__":
    main(sys.argv[1])
