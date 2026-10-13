// A VoiceSource that replays WAV clips as Discord would deliver them: per-user streams of 20 ms
// Opus packets, each preceded by a speaking event. It stands in for Discord *after*
// `receiver.subscribe`; it does not exercise transport decryption, DAVE, or SSRC mapping
// (docs/spec.md, 3.2).

import { readFileSync } from 'node:fs'
import { PassThrough, type Readable } from 'node:stream'

import OpusScript from 'opusscript'

import type { VoiceSource } from './voice-source.js'

const RATE = 48_000
const FRAME = 960 // samples per channel per 20 ms
const FRAME_BYTES = FRAME * 2 * 2

export interface ReplayClip {
  userId: string
  /** 48 kHz stereo s16le */
  pcm: Buffer
  /** When the user starts speaking, ms after `play()` (scaled by `speed`). */
  atMs: number
}

/** Read a 16-bit PCM WAV and return it as 48 kHz stereo s16le (linear-interpolated). */
export function loadWav(path: string): Buffer {
  const data = readFileSync(path)
  let offset = 12
  let rate = 0
  let channels = 0
  let pcm: Buffer | null = null
  while (offset + 8 <= data.length) {
    const id = data.toString('ascii', offset, offset + 4)
    const size = data.readUInt32LE(offset + 4)
    if (id === 'fmt ') {
      channels = data.readUInt16LE(offset + 10)
      rate = data.readUInt32LE(offset + 12)
      if (data.readUInt16LE(offset + 22) !== 16) throw new Error(`${path}: only 16-bit PCM WAV`)
    } else if (id === 'data') pcm = data.subarray(offset + 8, offset + 8 + size)
    offset += 8 + size + (size % 2)
  }
  if (!pcm || !rate) throw new Error(`${path}: not a PCM WAV`)
  const inFrames = Math.floor(pcm.length / (2 * channels))
  const mono = new Float64Array(inFrames)
  for (let i = 0; i < inFrames; i++) {
    let sum = 0
    for (let c = 0; c < channels; c++) sum += pcm.readInt16LE((i * channels + c) * 2)
    mono[i] = sum / channels
  }
  const outFrames = Math.floor((inFrames * RATE) / rate)
  const out = Buffer.alloc(outFrames * 4)
  for (let i = 0; i < outFrames; i++) {
    const position = (i * rate) / RATE
    const left = Math.floor(position)
    const right = Math.min(left + 1, inFrames - 1)
    const value = Math.round(mono[left] + (mono[right] - mono[left]) * (position - left))
    out.writeInt16LE(value, i * 4)
    out.writeInt16LE(value, i * 4 + 2)
  }
  return out
}

export function encodeOpus(pcm: Buffer): Buffer[] {
  const encoder = new OpusScript(RATE, 2, OpusScript.Application.AUDIO)
  const packets: Buffer[] = []
  for (let offset = 0; offset < pcm.length; offset += FRAME_BYTES) {
    let frame = pcm.subarray(offset, offset + FRAME_BYTES)
    if (frame.length < FRAME_BYTES) frame = Buffer.concat([frame, Buffer.alloc(FRAME_BYTES - frame.length)])
    packets.push(Buffer.from(encoder.encode(frame, FRAME)))
  }
  encoder.delete()
  return packets
}

export class ReplaySource implements VoiceSource {
  private listeners: ((userId: string) => void)[] = []
  private streams = new Map<string, PassThrough>()

  /** `speed` 1 is real time; higher plays faster (packets still arrive in order). */
  constructor(
    private readonly clips: ReplayClip[],
    private readonly speed = 1,
  ) {}

  onSpeaking(listener: (userId: string) => void): void {
    this.listeners.push(listener)
  }

  subscribe(userId: string, _silenceMs: number): Readable {
    const stream = new PassThrough({ objectMode: true })
    this.streams.set(userId, stream)
    return stream
  }

  private sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms / this.speed))
  }

  private async speak(clip: ReplayClip, packets: Buffer[]): Promise<void> {
    await this.sleep(clip.atMs)
    for (const listener of this.listeners) listener(clip.userId)
    const stream = this.streams.get(clip.userId)
    const started = Date.now()
    for (const [index, packet] of packets.entries()) {
      if (stream?.destroyed) return
      stream?.write(packet)
      const due = ((index + 1) * 20) / this.speed - (Date.now() - started)
      if (due > 0) await new Promise((resolve) => setTimeout(resolve, due))
      else if (index % 50 === 0) await new Promise((resolve) => setImmediate(resolve))
    }
    // Discord ends the subscription after the configured silence; so do we.
    await this.sleep(800)
    this.streams.delete(clip.userId)
    stream?.end()
  }

  /** Play every clip; resolves when all have finished. Clips of one user must not overlap. */
  async play(): Promise<void> {
    // Encode first, so encoding time does not shift the timeline.
    const encoded = this.clips.map((clip) => encodeOpus(clip.pcm))
    await Promise.all(this.clips.map((clip, i) => this.speak(clip, encoded[i])))
  }
}
