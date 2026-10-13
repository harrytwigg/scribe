import { describe, expect, it } from 'vitest'

import type { ScribeClient, Speaker } from '../src/client.js'
import { durationMs } from '../src/pcm.js'
import { Pipeline, isEncryptedFrame } from '../src/pipeline.js'
import { ReplaySource, encodeOpus, encryptedFrames } from '../src/replay.js'

// A tone, so Opus keeps real signal through the round trip.
function tone(ms: number, hz: number) {
  const frames = 48 * ms
  const pcm = Buffer.alloc(frames * 4)
  for (let i = 0; i < frames; i++) {
    const v = Math.round(8000 * Math.sin((2 * Math.PI * hz * i) / 48_000))
    pcm.writeInt16LE(v, i * 4)
    pcm.writeInt16LE(v, i * 4 + 2)
  }
  return pcm
}

class RecordingClient {
  posts: { speaker: Speaker; ms: number; startedAt: number }[] = []
  /** Holds every post until opened, like a retry waiting out its delay. */
  gate: Promise<void> = Promise.resolve()
  async postUtterance(_session: string, speaker: Speaker, pcm: Buffer, startedAt: number, allowed = () => true) {
    await this.gate
    if (!allowed()) return { ok: true, skipped: true }
    this.posts.push({ speaker, ms: durationMs(pcm), startedAt })
    return { ok: true, text: `${speaker.name} said something` }
  }
}

describe('Pipeline with a replayed call', () => {
  it('attributes concurrent speakers, caps long speech, skips opted-out users', async () => {
    const source = new ReplaySource(
      [
        { userId: 'a', pcm: tone(3000, 300), atMs: 0 },
        { userId: 'b', pcm: tone(30_000, 500), atMs: 500 }, // overlaps a, longer than the cap
        { userId: 'c', pcm: tone(2000, 700), atMs: 100 }, // opted out
        { userId: 'a', pcm: tone(1500, 300), atMs: 5000 },
      ],
      50,
    )
    const client = new RecordingClient()
    const lines: string[] = []
    const pipeline = new Pipeline({
      source,
      client: client as unknown as ScribeClient,
      sessionId: 's',
      speaker: (id) => ({ id, name: id.toUpperCase() }),
      isOptedOut: (id) => id === 'c',
      onLine: (speaker, text) => lines.push(`${speaker.id}:${text}`),
    })
    await source.play()
    await pipeline.stop()
    const byUser = (id: string) => client.posts.filter((p) => p.speaker.id === id).map((p) => Math.round(p.ms / 100) * 100)
    expect(byUser('a')).toEqual([3000, 1500])
    expect(byUser('b')).toEqual([25_000, 5000])
    expect(byUser('c')).toEqual([])
    expect(lines).toContain('a:A said something')
  })

  it('flushes a speaker who is mid-sentence when the call stops', async () => {
    const source = new ReplaySource([{ userId: 'a', pcm: tone(10_000, 300), atMs: 0 }], 1)
    const client = new RecordingClient()
    const pipeline = new Pipeline({
      source,
      client: client as unknown as ScribeClient,
      sessionId: 's',
      speaker: (id) => ({ id, name: id }),
      isOptedOut: () => false,
    })
    void source.play()
    await new Promise((resolve) => setTimeout(resolve, 1500))
    await pipeline.stop()
    expect(client.posts).toHaveLength(1)
    expect(client.posts[0].ms).toBeGreaterThan(1000)
    expect(client.posts[0].ms).toBeLessThan(2000)
  })
})

describe('Pipeline with undecodable packets', () => {
  it('keeps hearing a speaker whose first packet cannot be decoded', async () => {
    // One bad packet as the bot joined used to destroy that user's decoder: they were never heard
    // again for the rest of the call, while everyone else was.
    const source = new ReplaySource(
      [
        { userId: 'harry', pcm: tone(2000, 300), atMs: 0, corruptFirst: 1 },
        { userId: 'badger', pcm: tone(2000, 500), atMs: 200 },
        { userId: 'harry', pcm: tone(1500, 300), atMs: 4000 },
      ],
      20,
    )
    const client = new RecordingClient()
    const errors: string[] = []
    const pipeline = new Pipeline({
      source,
      client: client as unknown as ScribeClient,
      sessionId: 's',
      speaker: (id) => ({ id, name: id }),
      isOptedOut: () => false,
      onError: (error) => errors.push(error instanceof Error ? error.message : String(error)),
    })
    await source.play()
    await pipeline.stop()
    const byUser = (id: string) => client.posts.filter((p) => p.speaker.id === id).map((p) => Math.round(p.ms / 100) * 100)
    expect(byUser('harry')).toEqual([2000, 1500])
    expect(byUser('badger')).toEqual([2000])
    expect(pipeline.heard.get('harry')).toMatchObject({ corrupt: 1, encrypted: 0, lines: 2 })
    expect(errors).toEqual(['skipped 1 undecodable packet(s) from harry'])
  })

  it('loses only the bad packets, not the speech after them', async () => {
    const source = new ReplaySource([{ userId: 'a', pcm: tone(1000, 300), atMs: 0, corruptFirst: 5 }], 20)
    const client = new RecordingClient()
    const pipeline = new Pipeline({
      source,
      client: client as unknown as ScribeClient,
      sessionId: 's',
      speaker: (id) => ({ id, name: id }),
      isOptedOut: () => false,
    })
    await source.play()
    await pipeline.stop()
    expect(client.posts.map((p) => Math.round(p.ms / 100) * 100)).toEqual([1000])
  })
})

describe('Pipeline with frames it cannot decrypt', () => {
  it('never decodes DAVE-encrypted frames, and keeps hearing everyone', async () => {
    // 10 s of frames still encrypted (the bot joined before it could decrypt this speaker). Half
    // of them decode as noise, which would be posted as speech, and decoding enough of them
    // corrupts the heap every decoder shares, silencing everyone else too.
    const source = new ReplaySource(
      [
        { userId: 'harry', pcm: tone(1000, 300), atMs: 0, encryptedFirst: 500 },
        { userId: 'badger', pcm: tone(2000, 500), atMs: 11_000 },
        { userId: 'harry', pcm: tone(1500, 300), atMs: 14_000 },
      ],
      20,
    )
    const client = new RecordingClient()
    const pipeline = new Pipeline({
      source,
      client: client as unknown as ScribeClient,
      sessionId: 's',
      speaker: (id) => ({ id, name: id }),
      isOptedOut: () => false,
    })
    await source.play()
    await pipeline.stop()
    const byUser = (id: string) => client.posts.filter((p) => p.speaker.id === id).map((p) => Math.round(p.ms / 100) * 100)
    // Only the real speech: none of the encrypted frames became audio.
    expect(byUser('harry')).toEqual([1000, 1500])
    expect(byUser('badger')).toEqual([2000])
    expect(pipeline.heard.get('harry')).toMatchObject({ encrypted: 500, corrupt: 0 })
  })
})

describe('isEncryptedFrame', () => {
  it('matches DAVE frames and not real Opus that happens to end in 0xFAFA', () => {
    expect(encryptedFrames(200, 7).every(isEncryptedFrame)).toBe(true)
    // Opus ends in 0xFAFA far more often than chance; the trailer size byte tells them apart.
    const real = encodeOpus(tone(60_000, 300))
    expect(real.filter(isEncryptedFrame).length).toBeLessThanOrEqual(1)
    const lookalike = Buffer.concat([Buffer.alloc(40, 0x11), Buffer.from([0x03, 0xfa, 0xfa])])
    expect(isEncryptedFrame(lookalike)).toBe(false)
  })
})

describe('Pipeline opt-out', () => {
  it('never posts speech already cut when the speaker opts out', async () => {
    // 30 s turn: the first 25 s is cut and queued while the speaker is still talking.
    const source = new ReplaySource([{ userId: 'a', pcm: tone(30_000, 300), atMs: 0 }], 50)
    const client = new RecordingClient()
    let open!: () => void
    client.gate = new Promise((resolve) => (open = resolve))
    const optedOut = new Set<string>()
    const pipeline = new Pipeline({
      source,
      client: client as unknown as ScribeClient,
      sessionId: 's',
      speaker: (id) => ({ id, name: id }),
      isOptedOut: (id) => optedOut.has(id),
    })
    const playing = source.play()
    await new Promise((resolve) => setTimeout(resolve, 650)) // past the 25 s cut at 50x
    optedOut.add('a')
    pipeline.drop('a')
    open()
    await playing
    await pipeline.stop()
    expect(client.posts).toEqual([])
  })
})
