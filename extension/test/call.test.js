import assert from 'node:assert/strict'
import { test } from 'node:test'

import { Call } from '../src/call.js'
import { ServerError } from '../src/client.js'

function call({ heartbeat = async () => ({ ok: true }), heardAgoMs = 0 } = {}) {
  const c = new Call({ server: 'http://box:8787', myName: 'Me', othersName: 'Others' }, () => {}, { idleStopMs: 600_000 })
  c.sessionId = 's1'
  c.heard = Date.now() - heardAgoMs
  c.client.heartbeat = heartbeat
  c.stopped = 0
  c.stop = () => {
    c.stopped++
    c.stopping = Promise.resolve()
  }
  return c
}

test('a live call sends a heartbeat and carries on', async () => {
  let beats = 0
  const c = call({ heartbeat: async () => beats++ })
  await c.tick()
  assert.equal(beats, 1)
  assert.equal(c.stopped, 0)
})

test('ten minutes without a transcribed line ends the call', async () => {
  const c = call({ heardAgoMs: 600_001 })
  await c.tick()
  assert.equal(c.stopped, 1)
  assert.equal(c.reason, 'no speech for 10 min')
})

test('a session the server ended (reaped while asleep) ends the call', async () => {
  const c = call({ heartbeat: async () => { throw new ServerError(409, 'session has ended') } })
  await c.tick()
  assert.equal(c.stopped, 1)
  assert.equal(c.reason, 'the server ended the session')
})

test('an unreachable server does not end the call', async () => {
  const c = call({ heartbeat: async () => { throw new TypeError('fetch failed') } })
  await c.tick()
  assert.equal(c.stopped, 0)
})

test('a failed session create leaves the call done, so the next start is allowed', async () => {
  const c = new Call({ server: 'http://box:8787' }, () => {})
  c.client.createSession = async () => { throw new ServerError(401, 'sign in') }
  await assert.rejects(c.start('x', { tab: null, mic: null }), /sign in/)
  assert.equal(c.done, true)
})

// -- recording ---------------------------------------------------------------------------------

import { ScribeClient } from '../src/client.js'
import { ChunkUploader, memoryChunkStore } from '../src/uploader.js'
import { FakeRecorder, scriptedFetch } from './fakes.js'

/** A call on a server we script, recording through fake recorders. */
function recordingCall({ upload = [], utterance = [503], ...options } = {}) {
  FakeRecorder.instances = []
  const puts = scriptedFetch(upload, [201, { created: true }])
  const live = scriptedFetch(utterance, [503, { detail: 'transcription queue is full' }])
  const c = new Call({ server: 'http://box:8787', myName: 'Alice', othersName: 'Others' }, () => {}, {
    idleStopMs: 600_000, liveGraceMs: 20, uploadGraceMs: 30, recorderOptions: { Recorder: FakeRecorder },
    uploader: new ChunkUploader({ clientFor: () => new ScribeClient('http://box:8787', { fetch: puts.fetch }), store: memoryChunkStore(), sleep: () => new Promise((resolve) => setTimeout(resolve, 5)), backoffMs: [10] }),
    ...options,
  })
  c.client = new ScribeClient('http://box:8787', { retryDelaysMs: [60_000], fetch: live.fetch })
  c.sessionId = 's1'
  c.heard = Date.now()
  return { c, puts, live }
}

test('the audio is recorded and uploaded whether or not the server can transcribe', async () => {
  const { c, puts, live } = recordingCall({ utterance: [] }) // every utterance is refused with 503
  c.record({ id: 'tab stream' }, 'tab')
  c.record({ id: 'mic stream' }, 'mic')
  assert.deepEqual(c.recorders.map((r) => r.channel), ['tab', 'mic'])
  // A phrase is heard and cannot be posted...
  c.post({ id: 'tab', name: 'Others' }, new Float32Array(16_000), 1_800_000_001)
  // ...and the call ends, with the server refusing transcription all along.
  c.client.endSession = async () => ({})
  await c.stop()
  const channels = puts.calls.map((call) => call.url.match(/recording\/(\w+)\/(\d+)/).slice(1).join('/')).sort()
  assert.deepEqual(channels, ['mic/0', 'tab/0']) // one segment each, delivered before the session ended
  assert.ok(live.calls.some((call) => call.url.endsWith('/markers')), 'the lost phrase is marked, for the server to fill')
  assert.equal(c.uploader.pending, 0)
})

test('a recording the server cannot take yet stays in local storage and does not hold up the end of the call', async () => {
  const { c } = recordingCall({ upload: Array(100).fill(new TypeError('fetch failed')) })
  c.record({}, 'tab')
  let ended = false
  c.client.endSession = async () => void (ended = true)
  const started = Date.now()
  await c.stop()
  assert.ok(ended, 'the session is ended')
  assert.ok(Date.now() - started < 2000)
  assert.equal(c.uploader.pending, 1)
  assert.equal((await c.uploader.store.all()).length, 1)
  c.uploader.close()
})

test('utterances still being retried when the call ends become markers rather than holding it up', async () => {
  const { c, live } = recordingCall() // 503 for every attempt, with a minute between them
  c.post({ id: 'mic', name: 'Alice' }, new Float32Array(16_000 * 4), 1_800_000_010)
  await new Promise((resolve) => setTimeout(resolve, 10)) // the first attempt has failed; it waits a minute
  c.client.endSession = async () => ({})
  const started = Date.now()
  await c.stop()
  assert.ok(Date.now() - started < 2000)
  const marker = live.calls.find((call) => call.url.endsWith('/markers'))
  assert.equal(JSON.parse(marker.init.body).duration_ms, 4000)
})

test('speech the server could not transcribe still counts as the call being alive', () => {
  const { c } = recordingCall()
  c.heard = Date.now() - 599_000
  c.post({ id: 'tab', name: 'Others' }, new Float32Array(16_000), 1)
  assert.ok(Date.now() - c.heard < 1000)
})

test('recording can be turned off, and a browser that cannot record does not stop the call', () => {
  const { c } = recordingCall()
  c.config.record = false
  assert.equal(c.config.record, false)
  const noRecorder = recordingCall({ recorderOptions: { Recorder: undefined } }).c
  noRecorder.record({}, 'tab')
  assert.deepEqual(noRecorder.recorders, [])
  assert.match(noRecorder.recordingError, /not recording the tab/)
})

test('ending the call waits for the last segment to be stored before it asks whether the queue is empty', { timeout: 3000 }, async () => {
  const { c, puts } = recordingCall()
  c.record({}, 'tab')
  let release
  const store = c.uploader.store
  const put = store.put
  store.put = (record) => new Promise((resolve) => (release = () => put.call(store, record).then(resolve))) // a slow disk
  c.client.endSession = async () => ({})
  const stopping = c.stop()
  await new Promise((resolve) => setTimeout(resolve, 10))
  release()
  await stopping
  assert.equal(puts.calls.length, 1, 'the segment written last was still delivered before the session ended')
  assert.equal(c.uploader.pending, 0)
})
