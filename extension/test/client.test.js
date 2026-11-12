import assert from 'node:assert/strict'
import { test } from 'node:test'

import { ScribeClient } from '../src/client.js'
import { wav } from '../src/wav.js'

function fakeServer(responses) {
  const calls = []
  const fetch = async (url, init) => {
    calls.push({ url: String(url), init })
    const [status, body] = responses.shift() ?? [200, {}]
    return new Response(JSON.stringify(body), { status })
  }
  return { calls, fetch }
}

const speaker = { id: 'tab', name: 'Others' }

test('a busy server is retried with the same key, then succeeds', async () => {
  const server = fakeServer([[503, { detail: 'busy' }], [201, { text: 'hello' }]])
  const client = new ScribeClient('http://box:8787', { retryDelaysMs: [1, 1], fetch: server.fetch })
  const result = await client.postUtterance('s1', speaker, new Float32Array(1600), 100.5)
  assert.deepEqual(result, { ok: true, text: 'hello', skipped: undefined, lines: [{ text: 'hello' }] })
  const keys = server.calls.map((call) => call.init.body.get('client_key'))
  assert.deepEqual(keys, ['tab:100.5', 'tab:100.5'])
  assert.equal(server.calls[0].init.headers.Authorization, undefined)
  assert.equal(server.calls[0].url, 'http://box:8787/api/sessions/s1/utterances')
})

test('a mixed channel asks for diarization and gets every line back', async () => {
  const lines = [{ speaker_name: 'Speaker 1', text: 'hi' }, { speaker_name: 'Speaker 2', text: 'hello' }]
  const server = fakeServer([[201, { text: 'hi hello', lines }]])
  const client = new ScribeClient('http://box:8787', { fetch: server.fetch })
  const result = await client.postUtterance('s1', { ...speaker, diarize: true }, new Float32Array(1600), 1)
  assert.equal(server.calls[0].init.body.get('diarize'), 'true')
  assert.deepEqual(result.lines, lines)
  const plain = fakeServer([[201, { text: 'x' }]])
  await new ScribeClient('http://box:8787', { fetch: plain.fetch }).postUtterance('s1', speaker, new Float32Array(1600), 1)
  assert.equal(plain.calls[0].init.body.get('diarize'), null)
})

test('giving up leaves a marker line', async () => {
  const server = fakeServer([[503, {}], [503, {}], [503, {}], [201, {}]])
  const client = new ScribeClient('http://box:8787', { retryDelaysMs: [1, 1], fetch: server.fetch })
  const result = await client.postUtterance('s1', speaker, new Float32Array(32000), 1)
  assert.equal(result.ok, false)
  const marker = server.calls.at(-1)
  assert.match(marker.url, /\/markers$/)
  assert.equal(JSON.parse(marker.init.body).text, '[unavailable: 2 s of audio from Others]')
})

test('a client error is not retried', async () => {
  const server = fakeServer([[409, { detail: 'session has ended' }]])
  const client = new ScribeClient('http://box:8787', { retryDelaysMs: [1, 1], fetch: server.fetch })
  const result = await client.postUtterance('s1', speaker, new Float32Array(1600), 1)
  assert.deepEqual(result, { ok: false, error: 'session has ended' })
  assert.equal(server.calls.length, 1)
})

test('sessions are created with the browser source', async () => {
  const server = fakeServer([[201, { id: 's9' }]])
  const client = new ScribeClient('http://box:8787', { fetch: server.fetch })
  assert.equal((await client.createSession('Meet - abc')).id, 's9')
  assert.deepEqual(JSON.parse(server.calls[0].init.body), { source: 'browser', title: 'Meet - abc' })
})

test('wav header and clipping', () => {
  const bytes = wav(new Float32Array([0, 1, -1, 2]), 16000)
  const view = new DataView(bytes.buffer)
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'RIFF')
  assert.equal(view.getUint32(24, true), 16000)
  assert.equal(view.getUint32(40, true), 8)
  assert.deepEqual([1, 2, 3].map((i) => view.getInt16(44 + i * 2, true)), [32767, -32768, 32767])
})

test('a busy server is waited on for about five minutes before the audio is given up on', async () => {
  const { LIVE_RETRY_DELAYS_MS } = await import('../src/client.js')
  const total = LIVE_RETRY_DELAYS_MS.reduce((a, b) => a + b, 0)
  assert.ok(total >= 270_000 && total <= 330_000, `waits ${total / 1000} s`)
  assert.equal(new ScribeClient('http://box:8787').retryDelaysMs, LIVE_RETRY_DELAYS_MS)
})

test('a request timeout and a throttled server are retried', async () => {
  const server = fakeServer([[408, {}], [429, {}], [201, { text: 'hi' }]])
  const client = new ScribeClient('http://box:8787', { retryDelaysMs: [1, 1], fetch: server.fetch })
  assert.equal((await client.postUtterance('s1', speaker, new Float32Array(1600), 1)).ok, true)
  assert.equal(server.calls.length, 3)
})

test('the marker for a lost utterance says how long it was and which utterance it stands for', async () => {
  const server = fakeServer([[503, {}], [201, {}]])
  const client = new ScribeClient('http://box:8787', { retryDelaysMs: [], fetch: server.fetch })
  await client.postUtterance('s1', speaker, new Float32Array(40_000), 100.5)
  const body = JSON.parse(server.calls.at(-1).init.body)
  assert.deepEqual(body, {
    speaker_id: 'tab', speaker_name: 'Others', started_at: 100.5, text: '[unavailable: 3 s of audio from Others]',
    duration_ms: 2500, client_key: 'tab:100.5',
  })
})

test('when the call is over the waiting stops and the gap is marked at once', async () => {
  const server = fakeServer([[503, {}], [201, {}]])
  const client = new ScribeClient('http://box:8787', { retryDelaysMs: [60_000, 60_000], fetch: server.fetch })
  const stop = new AbortController()
  const result = client.postUtterance('s1', speaker, new Float32Array(16000), 1, { signal: stop.signal })
  await new Promise((resolve) => setTimeout(resolve, 20)) // it is now waiting a minute to retry
  stop.abort()
  assert.equal((await result).ok, false)
  assert.equal(server.calls.length, 2)
  assert.match(server.calls[1].url, /\/markers$/)
})

test('a recording segment is put at its place with its start time and length', async () => {
  const server = fakeServer([[201, { created: true }]])
  const client = new ScribeClient('http://box:8787', { fetch: server.fetch })
  const blob = new Blob(['abc'])
  assert.deepEqual(await client.putRecordingChunk('s1', 'mic', 7, { blob, mime: 'audio/webm;codecs=opus', startedAt: 1800000000.25, durationMs: 30400.4, sampleRate: 48000 }), { created: true })
  const { url, init } = server.calls[0]
  assert.equal(url, 'http://box:8787/api/sessions/s1/recording/mic/7?started_at=1800000000.25&duration_ms=30400&sample_rate=48000')
  assert.equal(init.method, 'PUT')
  assert.equal(init.headers['Content-Type'], 'audio/webm;codecs=opus')
  assert.equal(init.body, blob)
})
