// 48 kHz stereo s16le PCM (what Discord's Opus decodes to) -> mono, and WAV wrapping.

export const RATE = 48_000
export const FRAME_MS = 20
export const BYTES_PER_MS_MONO = (RATE / 1000) * 2

/** Average the two channels of interleaved s16le stereo into s16le mono. */
export function downmix(stereo: Buffer): Buffer {
  const frames = Math.floor(stereo.length / 4)
  const mono = Buffer.alloc(frames * 2)
  for (let i = 0; i < frames; i++) {
    mono.writeInt16LE((stereo.readInt16LE(i * 4) + stereo.readInt16LE(i * 4 + 2)) >> 1, i * 2)
  }
  return mono
}

/** Wrap s16le mono PCM as a WAV file. The server resamples to 16 kHz itself. */
export function wav(pcm: Buffer, rate = RATE): Buffer {
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + pcm.length, 4)
  header.write('WAVEfmt ', 8)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20) // PCM
  header.writeUInt16LE(1, 22) // mono
  header.writeUInt32LE(rate, 24)
  header.writeUInt32LE(rate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(pcm.length, 40)
  return Buffer.concat([header, pcm])
}

export const durationMs = (monoPcm: Buffer) => monoPcm.length / BYTES_PER_MS_MONO
