import assert from 'node:assert/strict'
import { test } from 'node:test'

import { FRAME, RATE, Segmenter } from '../src/segmenter.js'

const ms = (n) => Math.round((n / 1000) * RATE)

function tone(durationMs, amplitude = 0.3) {
  const samples = new Float32Array(ms(durationMs))
  for (let i = 0; i < samples.length; i++) samples[i] = amplitude * Math.sin((2 * Math.PI * 220 * i) / RATE)
  return samples
}

const quiet = (durationMs) => tone(durationMs, 0.0005)

/** Speech-shaped: syllables of tone with the short dips between them that real speech has. */
function talk(durationMs) {
  const samples = tone(durationMs)
  for (let start = ms(360); start < samples.length; start += ms(400)) samples.fill(0.0005, start, start + ms(40))
  return samples
}

function run(chunks, options) {
  const out = []
  const segmenter = new Segmenter((u) => out.push(u), options)
  for (const chunk of chunks) {
    // Feed in uneven pieces, like the worklet's 128-sample quanta.
    for (let i = 0; i < chunk.length; i += 128) segmenter.push(chunk.slice(i, i + 128))
  }
  segmenter.flush()
  return out
}

test('an utterance ends after the silence gap, keeping its onset', () => {
  const out = run([quiet(1000), tone(1000), quiet(1000), tone(600), quiet(1000)])
  assert.equal(out.length, 2)
  // Starts within the 300 ms pre-roll before the tone at 1 s.
  assert.ok(out[0].startSample >= ms(700) && out[0].startSample <= ms(1000))
  const seconds = out[0].samples.length / RATE
  assert.ok(seconds >= 1 && seconds < 1.6, `first utterance ${seconds} s`)
  assert.ok(out[1].startSample >= ms(2700) && out[1].startSample <= ms(3000))
})

test('a gap shorter than the silence threshold does not split', () => {
  const out = run([quiet(500), tone(800), quiet(300), tone(800), quiet(1000)])
  assert.equal(out.length, 1)
})

test('long speech is cut at the cap and carries on', () => {
  const out = run([quiet(500), talk(31_000), quiet(1000)])
  assert.equal(out.length, 2)
  assert.equal(out[0].samples.length, ms(25_000))
  assert.ok(out[1].startSample >= out[0].startSample + ms(25_000) - FRAME)
})

test('a sound that never dips becomes the noise floor and stops counting as speech', () => {
  const out = run([quiet(500), tone(20_000), quiet(1000)])
  assert.equal(out.length, 1)
  assert.ok(out[0].samples.length < ms(12_000))
})

test('blips shorter than the speech floor are dropped', () => {
  assert.equal(run([quiet(1000), tone(200), quiet(1000)]).length, 0)
})

test('a steady noise floor is not speech, but speech over it is', () => {
  const noise = (durationMs) => tone(durationMs, 0.02)
  const loud = run([noise(6000), tone(1000, 0.3), noise(2000)])
  assert.equal(loud.length, 1)
})

test('flush closes an open utterance', () => {
  const out = []
  const segmenter = new Segmenter((u) => out.push(u))
  segmenter.push(quiet(500))
  segmenter.push(tone(1000))
  assert.equal(out.length, 0)
  segmenter.flush()
  assert.equal(out.length, 1)
})
