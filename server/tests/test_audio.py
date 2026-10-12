import numpy as np

from scribe_server.audio import MAX_CHUNK_S, RATE, decode, speech_chunks

from .conftest import energy_vad, silence, tone, wav


def test_decode_downmixes_and_resamples():
    audio = decode(wav(tone(1.0, rate=48_000, channels=2), rate=48_000))
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - RATE) <= 2


def test_silence_has_no_chunks():
    assert speech_chunks(silence(3), energy_vad) == []


def test_tiny_noise_is_dropped():
    assert speech_chunks(np.concatenate([silence(1), tone(0.1), silence(1)]), energy_vad) == []


def test_short_utterance_with_pause_is_one_chunk():
    audio = np.concatenate([tone(2), silence(0.5), tone(2)])
    chunks = speech_chunks(audio, energy_vad)
    assert len(chunks) == 1
    assert abs(len(chunks[0]) / RATE - 4.5) < 0.05


def test_long_audio_is_split_at_most_25s():
    audio = np.concatenate([np.concatenate([tone(9), silence(0.5)]) for _ in range(8)])  # ~76 s
    chunks = speech_chunks(audio, energy_vad)
    assert len(chunks) >= 3
    assert all(len(c) <= MAX_CHUNK_S * RATE for c in chunks)
    # 72 s of speech; pauses inside a chunk are kept, the ones between chunks are not
    assert 72 <= sum(len(c) for c in chunks) / RATE <= 75.5


def test_unbroken_speech_is_capped():
    chunks = speech_chunks(tone(60), energy_vad)
    assert all(len(c) <= MAX_CHUNK_S * RATE for c in chunks)
    assert abs(sum(len(c) for c in chunks) / RATE - 60) < 0.1
