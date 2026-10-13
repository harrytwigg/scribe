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
  const client = new ScribeClient('http://box:8787', 'tok', { retryDelaysMs: [1, 1], fetch: server.fetch })
  const result = await client.postUtterance('s1', speaker, new Float32Array(1600), 100.5)
  assert.deepEqual(result, { ok: true, text: 'hello', skipped: undefined })
  const keys = server.calls.map((call) => call.init.body.get('client_key'))
  assert.deepEqual(keys, ['tab:100.5', 'tab:100.5'])
  assert.equal(server.calls[0].init.headers.Authorization, 'Bearer tok')
  assert.equal(server.calls[0].url, 'http://box:8787/api/sessions/s1/utterances')
})

test('giving up leaves a marker line', async () => {
  const server = fakeServer([[503, {}], [503, {}], [503, {}], [201, {}]])
  const client = new ScribeClient('http://box:8787', 'tok', { retryDelaysMs: [1, 1], fetch: server.fetch })
  const result = await client.postUtterance('s1', speaker, new Float32Array(32000), 1)
  assert.equal(result.ok, false)
  const marker = server.calls.at(-1)
  assert.match(marker.url, /\/markers$/)
  assert.equal(JSON.parse(marker.init.body).text, '[unavailable: 2 s of audio from Others]')
})

test('a client error is not retried', async () => {
  const server = fakeServer([[409, { detail: 'session has ended' }]])
  const client = new ScribeClient('http://box:8787', 'tok', { retryDelaysMs: [1, 1], fetch: server.fetch })
  const result = await client.postUtterance('s1', speaker, new Float32Array(1600), 1)
  assert.deepEqual(result, { ok: false, error: 'session has ended' })
  assert.equal(server.calls.length, 1)
})

test('sessions are created with the browser source', async () => {
  const server = fakeServer([[201, { id: 's9' }]])
  const client = new ScribeClient('http://box:8787', 'tok', { fetch: server.fetch })
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
