import { describe, expect, it } from 'vitest'

import type { ScribeClient, Speaker } from '../src/client.js'
import { durationMs } from '../src/pcm.js'
import { Pipeline } from '../src/pipeline.js'
import { ReplaySource } from '../src/replay.js'

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
  async postUtterance(_session: string, speaker: Speaker, pcm: Buffer, startedAt: number) {
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
