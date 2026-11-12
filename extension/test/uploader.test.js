import assert from 'node:assert/strict'
import { test } from 'node:test'

import { ScribeClient, ServerError } from '../src/client.js'
import { ChunkUploader, memoryChunkStore } from '../src/uploader.js'
import { scriptedFetch } from './fakes.js'

const segment = (index, channel = 'tab') => ({ channel, index, blob: new Blob(['x']), mime: 'audio/webm', startedAt: 100 + index * 30, durationMs: 30_400 })

function uploader(script, options = {}) {
  const server = scriptedFetch(script, [201, { created: true }])
  const client = new ScribeClient('http://box:8787', { fetch: server.fetch })
  const slept = []
  const store = memoryChunkStore()
  let t = 0 // a clock that only the waiting moves
  const u = new ChunkUploader({ clientFor: () => client, store, now: () => t, sleep: async (ms) => void (slept.push(ms), (t += ms)), ...options })
  return { u, server, slept, store }
}

test('a segment is kept locally, sent to its place, and forgotten once the server has it', async () => {
  const { u, server, store } = uploader([])
  await u.add('http://box:8787', 's1', segment(0))
  assert.equal(await u.whenIdle(1000), true)
  assert.equal(server.calls.length, 1)
  const call = server.calls[0]
  assert.equal(call.init.method, 'PUT')
  assert.equal(call.url, 'http://box:8787/api/sessions/s1/recording/tab/0?started_at=100&duration_ms=30400&sample_rate=0')
  assert.equal(call.init.headers['Content-Type'], 'audio/webm')
  assert.equal(await call.init.body.text(), 'x')
  assert.deepEqual(await store.all(), [])
})

test('a busy or absent server is retried for as long as it takes, the segment staying on disk meanwhile', async () => {
  const { u, server, slept, store } = uploader([[503, { detail: 'busy' }], new TypeError('fetch failed'), [503, {}], [201, { created: true }]])
  await u.add('http://box:8787', 's1', segment(0))
  assert.equal(await u.whenIdle(1000), true)
  assert.equal(server.calls.length, 4)
  assert.deepEqual(slept, [1000, 2000, 5000]) // growing waits
  assert.deepEqual(await store.all(), [])
  assert.equal(u.sent, 1)
})

test('while one segment waits to be retried, the others still go', async () => {
  const { u, server } = uploader([[503, {}]])
  await u.add('http://box:8787', 's1', segment(0))
  await u.add('http://box:8787', 's1', segment(1))
  assert.equal(await u.whenIdle(1000), true)
  const sent = server.calls.map((call) => call.url.match(/tab\/(\d+)/)[1])
  assert.deepEqual(sent, ['0', '1', '0'])
})

test('a segment the server already has is not an error, and is not counted twice', async () => {
  const { u, server } = uploader([[200, { created: false }]])
  await u.add('http://box:8787', 's1', segment(0))
  await u.whenIdle(1000)
  assert.equal(server.calls.length, 1)
  assert.equal(u.sent, 0)
  assert.equal(u.pending, 0)
})

test('the same segment added twice is queued once', async () => {
  const { u, server } = uploader([])
  await Promise.all([u.add('http://box:8787', 's1', segment(0)), u.add('http://box:8787', 's1', segment(0))])
  await u.whenIdle(1000)
  assert.equal(server.calls.length, 1)
})

test('a server that will take no recording stops the whole session, and later segments are dropped', async () => {
  const { u, server, store } = uploader([[403, { detail: 'call recording is off' }]])
  await u.add('http://box:8787', 's1', segment(0))
  await u.add('http://box:8787', 's1', segment(1))
  await u.add('http://box:8787', 's2', segment(0))
  await u.whenIdle(1000)
  await u.add('http://box:8787', 's1', segment(2))
  assert.deepEqual(server.calls.map((c) => c.url.match(/sessions\/(\w+)/)[1]), ['s1', 's2'])
  assert.equal(u.lastError, null) // s2 went through afterwards
  assert.deepEqual(await store.all(), [])
})

test('a segment the server rejects as malformed is dropped, not retried forever', async () => {
  const { u, server } = uploader([[400, { detail: 'segment index must be between 0 and 99999' }]])
  await u.add('http://box:8787', 's1', segment(0))
  await u.whenIdle(1000)
  assert.equal(server.calls.length, 1)
  assert.equal(u.pending, 0)
})

test('what an earlier document left behind is sent, and what is a week old is not', async () => {
  const { u, server, store } = uploader([], { now: () => 10 * 24 * 3600 * 1000 })
  await store.put({ key: 'a', server: 'http://box:8787', sessionId: 's1', channel: 'mic', index: 4, blob: new Blob(['x']), mime: 'audio/webm', startedAt: 1, durationMs: 1, addedAt: 9 * 24 * 3600 * 1000 })
  await store.put({ key: 'b', server: 'http://box:8787', sessionId: 's0', channel: 'mic', index: 1, blob: new Blob(['x']), mime: 'audio/webm', startedAt: 1, durationMs: 1, addedAt: 1000 })
  await u.resume()
  await u.whenIdle(1000)
  assert.deepEqual(server.calls.map((c) => c.url.match(/sessions\/(\w+)\/recording\/(\w+)\/(\d+)/).slice(1)), [['s1', 'mic', '4']])
  assert.deepEqual(await store.all(), [])
})

test('whenIdle gives up waiting after its timeout and leaves the segment queued', async () => {
  const { u } = uploader([], { sleep: () => new Promise(() => {}) }) // a wait that never ends
  u.clientFor = () => ({ putRecordingChunk: async () => { throw new ServerError(503, 'busy') } })
  await u.add('http://box:8787', 's1', segment(0))
  assert.equal(await u.whenIdle(20), false)
  assert.equal(u.pending, 1)
  u.close()
})

test('a segment is stored before it is sent', async () => {
  const events = []
  const store = { put: async () => void events.push('stored'), delete: async () => void events.push('deleted'), all: async () => [] }
  const client = { putRecordingChunk: async () => void events.push('sent') }
  const u = new ChunkUploader({ clientFor: () => client, store })
  await u.add('http://box:8787', 's1', segment(0))
  await u.whenIdle(1000)
  assert.deepEqual(events, ['stored', 'sent', 'deleted'])
})
