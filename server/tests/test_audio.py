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


def test_decode_reads_a_path_in_blocks_like_bytes(tmp_path):
    path = tmp_path / "a.wav"
    data = wav(tone(95.0, rate=44_100, channels=2), rate=44_100)  # several 30 s blocks, resampled
    path.write_bytes(data)
    from_path, from_bytes = decode(path), decode(data)
    assert abs(len(from_path) - 95 * RATE) <= 40 and abs(len(from_path) - len(from_bytes)) <= 40
    n = min(len(from_path), len(from_bytes))
    assert np.abs(from_path[1000:n - 1000] - from_bytes[1000:n - 1000]).max() < 0.02
    assert abs(float(np.abs(from_path).max()) - 0.3) < 0.02


def test_decode_aac_keeps_the_level_of_the_other_formats(tmp_path):
    import io

    import av

    left_right = tone(2.0, channels=2)  # identical channels, peak 0.3
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("aac", rate=RATE)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray((left_right.T * 32767).astype("int16").copy(order="C"), format="s16p", layout="stereo")
        frame.sample_rate = RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    path = tmp_path / "a.m4a"
    path.write_bytes(buffer.getvalue())
    for source in (path, buffer.getvalue()):
        decoded = decode(source)
        assert abs(len(decoded) / RATE - 2.0) < 0.1
        assert abs(float(np.abs(decoded).max()) - 0.3) < 0.05, "AAC must not be louder than the same audio as WAV"
