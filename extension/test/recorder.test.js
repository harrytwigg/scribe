import assert from 'node:assert/strict'
import { beforeEach, test } from 'node:test'

import { ChannelRecorder } from '../src/recorder.js'
import { FakeRecorder, fakeClock } from './fakes.js'

let clock
beforeEach(() => {
  FakeRecorder.instances = []
  clock = fakeClock()
})

function recorder(onSegment, options = {}) {
  return new ChannelRecorder({ id: 'stream' }, 'tab', onSegment, { Recorder: FakeRecorder, now: clock.now, schedule: clock.schedule, segmentMs: 30_000, overlapMs: 400, ...options })
}

test('a channel is recorded as Opus in WebM at a small bitrate', () => {
  const r = recorder(() => {})
  r.start()
  assert.equal(FakeRecorder.instances.length, 1)
  assert.deepEqual(FakeRecorder.instances[0].options, { audioBitsPerSecond: 24_000, mimeType: 'audio/webm;codecs=opus' })
})

test('every 30 s a new recorder takes over, a moment before the old one stops, so no audio falls between', async () => {
  const segments = []
  const r = recorder((segment) => segments.push(segment))
  r.start()
  await Promise.resolve() // the recorder reports that it has started
  await clock.advance(30_000)
  assert.equal(FakeRecorder.instances.length, 2)
  assert.equal(FakeRecorder.instances[0].state, 'recording') // still running through the overlap
  assert.equal(segments.length, 0)
  await clock.advance(400)
  assert.equal(FakeRecorder.instances[0].state, 'inactive')
  assert.equal(segments.length, 1)
  assert.equal(segments[0].index, 0)
  assert.equal(segments[0].channel, 'tab')
  assert.equal(segments[0].startedAt, 1_800_000_000)
  assert.equal(segments[0].durationMs, 30_400) // the overlap is part of it
  assert.equal(segments[0].mime, 'audio/webm;codecs=opus')
  assert.equal(await segments[0].blob.text(), 'audio 0')

  await clock.advance(30_000)
  await clock.advance(400)
  assert.deepEqual(segments.map((s) => s.index), [0, 1])
  assert.equal(segments[1].startedAt, 1_800_000_030) // the second began where the first's 30 s ended
})

test('stopping hands over the segment being recorded, and the old one if it is still overlapping', async () => {
  const segments = []
  const r = recorder((segment) => segments.push(segment))
  r.start()
  await Promise.resolve()
  await clock.advance(30_100) // inside an overlap
  await r.stop()
  assert.deepEqual(segments.map((s) => s.index).sort(), [0, 1])
  assert.ok(FakeRecorder.instances.every((i) => i.state === 'inactive'))
  assert.equal(clock.timers.length, 0, 'no timer is left to start another')
})

test('a recorder that fails to start next time is retried at the one after, and the old one keeps going', async () => {
  const segments = []
  const r = recorder((segment) => segments.push(segment))
  r.start()
  await Promise.resolve()
  const first = FakeRecorder.instances[0]
  const original = FakeRecorder.prototype.start
  FakeRecorder.prototype.start = () => {
    throw new Error('the stream ended')
  }
  await clock.advance(30_000)
  FakeRecorder.prototype.start = original
  assert.equal(r.error, 'the stream ended')
  assert.equal(first.state, 'recording')
  await clock.advance(30_000)
  assert.equal(r.error, null)
  assert.equal(first.state, 'recording') // until the new one has had its overlap
  await clock.advance(400)
  assert.equal(first.state, 'inactive')
  assert.deepEqual(segments.map((s) => s.index), [0]) // the one that never started took no index
})

test('a browser without MediaRecorder says so', () => {
  const r = new ChannelRecorder({}, 'mic', () => {}, { Recorder: undefined })
  assert.throws(() => r.start(), /no MediaRecorder/)
})
