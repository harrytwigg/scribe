import { describe, expect, it } from 'vitest'

import { durationMs } from '../src/pcm.js'
import { Segmenter, type Utterance } from '../src/segmenter.js'

const stereoMs = (ms: number) => Buffer.alloc((48_000 / 1000) * ms * 4, 1)

function collect() {
  const out: Utterance[] = []
  let clock = 1000
  const segmenter = new Segmenter((u) => out.push(u), () => clock++)
  return { out, segmenter }
}

describe('Segmenter', () => {
  it('emits on flush', () => {
    const { out, segmenter } = collect()
    for (let i = 0; i < 100; i++) segmenter.push(stereoMs(20))
    segmenter.flush()
    expect(out).toHaveLength(1)
    expect(durationMs(out[0].pcm)).toBe(2000)
  })

  it('drops utterances under 400 ms', () => {
    const { out, segmenter } = collect()
    for (let i = 0; i < 19; i++) segmenter.push(stereoMs(20))
    segmenter.flush()
    expect(out).toHaveLength(0)
    for (let i = 0; i < 20; i++) segmenter.push(stereoMs(20))
    segmenter.flush()
    expect(out).toHaveLength(1)
  })

  it('cuts at 25 s and keeps going', () => {
    const { out, segmenter } = collect()
    for (let i = 0; i < 3000; i++) segmenter.push(stereoMs(20)) // 60 s
    segmenter.flush()
    expect(out.map((u) => durationMs(u.pcm))).toEqual([25_000, 25_000, 10_000])
    expect(new Set(out.map((u) => u.startedAt)).size).toBe(3)
  })

  it('splits a chunk that straddles the cap', () => {
    const { out, segmenter } = collect()
    segmenter.push(stereoMs(24_990))
    segmenter.push(stereoMs(500))
    segmenter.flush()
    expect(out.map((u) => durationMs(u.pcm))).toEqual([25_000, 490])
  })
})
