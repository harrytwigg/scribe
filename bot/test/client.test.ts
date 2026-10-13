import { type IncomingMessage, type ServerResponse, createServer } from 'node:http'
import type { AddressInfo } from 'node:net'

import { afterEach, describe, expect, it } from 'vitest'

import { ScribeClient } from '../src/client.js'

type Handler = (req: IncomingMessage, res: ServerResponse, body: string) => void

async function serve(handler: Handler) {
  const server = createServer((req, res) => {
    let body = ''
    req.on('data', (c) => (body += c))
    req.on('end', () => handler(req, res, body))
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  const { port } = server.address() as AddressInfo
  servers.push(server)
  return `http://127.0.0.1:${port}`
}

const servers: ReturnType<typeof createServer>[] = []
afterEach(() => servers.splice(0).forEach((s) => s.close()))

const speaker = { id: 'u1', name: 'Ada' }
const pcm = Buffer.alloc(48 * 2 * 3000) // 3 s mono

describe('ScribeClient.postUtterance', () => {
  it('sends the bearer token and the utterance fields', async () => {
    let seen: { auth?: string; body: string } = { body: '' }
    const url = await serve((req, res, body) => {
      seen = { auth: req.headers.authorization, body }
      res.writeHead(201, { 'Content-Type': 'application/json' }).end('{"text":"hello"}')
    })
    const result = await new ScribeClient(url, 'tok').postUtterance('s1', speaker, pcm, 123.5)
    expect(result).toEqual({ ok: true, text: 'hello', skipped: undefined })
    expect(seen.auth).toBe('Bearer tok')
    expect(seen.body).toContain('name="speaker_name"\r\n\r\nAda')
    expect(seen.body).toContain('name="started_at"\r\n\r\n123.5')
  })

  it('retries a busy server, then succeeds', async () => {
    let calls = 0
    const url = await serve((_req, res) => {
      calls++
      if (calls < 3) res.writeHead(503, { 'Content-Type': 'application/json' }).end('{"detail":"queue full"}')
      else res.writeHead(201, { 'Content-Type': 'application/json' }).end('{"text":"late"}')
    })
    const result = await new ScribeClient(url, undefined, { retryDelaysMs: [10, 10] }).postUtterance('s1', speaker, pcm, 1)
    expect(result.text).toBe('late')
    expect(calls).toBe(3)
  })

  it('records a marker after the retries run out', async () => {
    const paths: string[] = []
    let marker = ''
    const url = await serve((req, res, body) => {
      paths.push(req.url!)
      if (req.url!.endsWith('/markers')) {
        marker = JSON.parse(body).text
        res.writeHead(201, { 'Content-Type': 'application/json' }).end('{}')
      } else res.writeHead(503, { 'Content-Type': 'application/json' }).end('{"detail":"queue full"}')
    })
    const result = await new ScribeClient(url, undefined, { retryDelaysMs: [5, 5] }).postUtterance('s1', speaker, pcm, 1)
    expect(result.ok).toBe(false)
    expect(paths.filter((p) => p.endsWith('/utterances'))).toHaveLength(3)
    expect(marker).toBe('[unavailable: 3 s of audio from Ada]')
  })

  it('does not retry a client error', async () => {
    let calls = 0
    const url = await serve((_req, res) => {
      calls++
      res.writeHead(409, { 'Content-Type': 'application/json' }).end('{"detail":"session has ended"}')
    })
    const result = await new ScribeClient(url, undefined, { retryDelaysMs: [5, 5] }).postUtterance('s1', speaker, pcm, 1)
    expect(result).toEqual({ ok: false, error: 'session has ended' })
    expect(calls).toBe(1)
  })
})

describe('ScribeClient.postUtterance with an opt-out', () => {
  it('posts nothing when the speaker is not allowed', async () => {
    let calls = 0
    const url = await serve((_req, res) => {
      calls++
      res.writeHead(201, { 'Content-Type': 'application/json' }).end('{"text":"x"}')
    })
    const result = await new ScribeClient(url).postUtterance('s1', speaker, pcm, 1, () => false)
    expect(result).toEqual({ ok: true, skipped: true })
    expect(calls).toBe(0)
  })

  it('stops retrying, and records no marker, once the speaker opts out', async () => {
    const paths: string[] = []
    let allowed = true
    const url = await serve((req, res) => {
      paths.push(req.url!)
      allowed = false // the user opts out while the first attempt is failing
      res.writeHead(503, { 'Content-Type': 'application/json' }).end('{"detail":"queue full"}')
    })
    const result = await new ScribeClient(url, undefined, { retryDelaysMs: [5, 5] }).postUtterance('s1', speaker, pcm, 1, () => allowed)
    expect(result.skipped).toBe(true)
    expect(paths).toEqual(['/api/sessions/s1/utterances'])
  })

  it('sends the same idempotency key on every retry', async () => {
    const keys: string[] = []
    let calls = 0
    const url = await serve((_req, res, body) => {
      keys.push(/name="client_key"\r\n\r\n([^\r]*)/.exec(body)![1])
      if (++calls < 3) res.writeHead(503, { 'Content-Type': 'application/json' }).end('{}')
      else res.writeHead(201, { 'Content-Type': 'application/json' }).end('{"text":"ok"}')
    })
    await new ScribeClient(url, undefined, { retryDelaysMs: [5, 5] }).postUtterance('s1', speaker, pcm, 42.5)
    expect(keys).toEqual(['u1:42.5', 'u1:42.5', 'u1:42.5'])
  })

  it('retries a timeout, with the same key', async () => {
    const keys: string[] = []
    const url = await serve((_req, res, body) => {
      keys.push(/name="client_key"\r\n\r\n([^\r]*)/.exec(body)![1])
      if (keys.length === 1) setTimeout(() => res.writeHead(201).end('{}'), 300) // too slow
      else res.writeHead(201, { 'Content-Type': 'application/json' }).end('{"text":"ok"}')
    })
    const result = await new ScribeClient(url, undefined, { retryDelaysMs: [5], timeoutMs: 100 }).postUtterance('s1', speaker, pcm, 7)
    expect(result.text).toBe('ok')
    expect(new Set(keys)).toEqual(new Set(['u1:7']))
  })
})
