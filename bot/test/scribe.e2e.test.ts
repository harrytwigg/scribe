// End to end, everything but Discord's network: two replayed speakers talking over each other,
// through the real bot pipeline (Opus decode, segmentation, HTTP) into a real scribe server on the
// GPU, with the model swapped mid-call. Proves the pipeline downstream of `receiver.subscribe`
// (docs/spec.md, 3.2 and 6), not Discord receive.
//
//   pnpm e2e        (needs the server's venv: cd ../server && uv sync; models download if missing)
//
// SCRIBE_E2E_MODELS=<first>,<second> picks the models (default Parakeet v2, then Whisper turbo).

import { type ChildProcess, spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import { ScribeClient } from '../src/client.js'
import { Pipeline } from '../src/pipeline.js'
import { ReplaySource, loadWav } from '../src/replay.js'
import { wer } from './wer.js'

const FIXTURES = join(import.meta.dirname, 'fixtures')
const SERVER = join(import.meta.dirname, '..', '..', 'server')
const TOKEN = 'e2e-token'
const [FIRST, SECOND] = (process.env.SCRIBE_E2E_MODELS ?? 'parakeet-tdt-0.6b-v2,whisper-large-v3-turbo').split(',')
const JFK = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
const LIBRI: Record<string, string> = JSON.parse(readFileSync(join(FIXTURES, 'librispeech.json'), 'utf8'))

let server: ChildProcess
let dataDir = ''
let base = ''
const log: string[] = []

async function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const s = createServer().listen(0, '127.0.0.1', () => {
      const { port } = s.address() as { port: number }
      s.close(() => resolve(port))
    })
  })
}

async function api<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(base + path, {
    method,
    headers: { Authorization: `Bearer ${TOKEN}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!response.ok) throw new Error(`${method} ${path}: ${response.status} ${await response.text()}`)
  return response.json() as Promise<T>
}

type Model = { id: string; status: string; active: boolean; loading: boolean; error: string | null }

async function until<T>(what: string, fn: () => Promise<T | undefined>, timeoutMs: number): Promise<T> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    const value = await fn()
    if (value !== undefined) return value
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`)
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
}

async function ensureDownloaded(id: string) {
  const model = (await api<Model[]>('GET', '/api/models')).find((m) => m.id === id)!
  if (model.status === 'not_downloaded') await api('POST', `/api/models/${id}/download`)
  await until(`${id} download`, async () => {
    const m = (await api<Model[]>('GET', '/api/models')).find((x) => x.id === id)!
    if (m.error) throw new Error(m.error)
    return m.status === 'downloaded' ? m : undefined
  }, 1_800_000)
}

async function activate(id: string) {
  await api('POST', `/api/models/${id}/activate`)
  await until(`${id} active`, async () => {
    const m = (await api<Model[]>('GET', '/api/models')).find((x) => x.id === id)!
    if (m.error) throw new Error(m.error)
    return m.active ? m : undefined
  }, 600_000)
}

beforeAll(async () => {
  const port = await freePort()
  dataDir = mkdtempSync(join(tmpdir(), 'scribe-e2e-'))
  base = `http://127.0.0.1:${port}`
  server = spawn(join(SERVER, '.venv', 'bin', 'python'), ['-m', 'scribe_server'], {
    cwd: SERVER,
    env: {
      ...process.env,
      SCRIBE_HOST: '127.0.0.1',
      SCRIBE_PORT: String(port),
      SCRIBE_DATA_DIR: dataDir,
      // Share the installed models rather than downloading them again.
      SCRIBE_MODELS_DIR: process.env.SCRIBE_MODELS_DIR ?? join(homedir(), '.local', 'share', 'scribe', 'models'),
      SCRIBE_API_TOKEN: TOKEN,
      SCRIBE_DEFAULT_MODEL: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  server.stdout!.on('data', (d) => log.push(String(d)))
  server.stderr!.on('data', (d) => log.push(String(d)))
  await until('server', () => fetch(`${base}/api/whoami`).then(() => true).catch(() => undefined), 120_000)
  await ensureDownloaded(FIRST)
  await ensureDownloaded(SECOND)
  await activate(FIRST)
})

afterAll(() => {
  server?.kill('SIGTERM')
  if (dataDir) rmSync(dataDir, { recursive: true, force: true })
  if (process.env.SCRIBE_E2E_LOG) writeFileSync(process.env.SCRIBE_E2E_LOG, log.join(''))
})

describe('replayed call -> bot pipeline -> scribe server', () => {
  it('transcribes two overlapping speakers, with a model swap mid-call', async () => {
    const client = new ScribeClient(base, TOKEN)
    const session = await client.createSession('e2e call', 'guild', 'channel')
    const ada = 'ada-id'
    const bob = 'bob-id'
    const source = new ReplaySource(
      [
        { userId: ada, pcm: loadWav(join(FIXTURES, 'jfk.wav')), atMs: 0 },
        { userId: bob, pcm: loadWav(join(FIXTURES, '1272-128104-0000.wav')), atMs: 2_000 }, // overlaps Ada
        { userId: bob, pcm: loadWav(join(FIXTURES, '1272-128104-0002.wav')), atMs: 10_000 },
        { userId: ada, pcm: loadWav(join(FIXTURES, 'jfk.wav')), atMs: 14_000 }, // overlaps Bob
        { userId: bob, pcm: loadWav(join(FIXTURES, '1272-128104-0004.wav')), atMs: 24_000 }, // 29 s: cut at 25 s
      ],
      2, // twice real time: packets still arrive in order, the test takes ~30 s
    )
    const errors: unknown[] = []
    const pipeline = new Pipeline({
      source,
      client,
      sessionId: session.id,
      speaker: (id) => ({ id, name: id === ada ? 'Ada' : 'Bob' }),
      isOptedOut: () => false,
      onError: (error) => errors.push(error),
    })

    const playing = source.play()
    // Swap once the first line has landed, while speech is still arriving.
    await until('first line', async () => {
      const s = await api<{ utterances: unknown[] }>('GET', `/api/sessions/${session.id}`)
      return s.utterances.length ? true : undefined
    }, 120_000)
    const swapping = activate(SECOND)
    await playing
    await swapping
    await pipeline.stop()
    await client.endSession(session.id)

    expect(errors).toEqual([])
    type Line = { speaker_id: string; speaker_name: string; text: string; model_id: string; started_at: number }
    const { utterances } = await api<{ utterances: Line[] }>('GET', `/api/sessions/${session.id}`)
    const said = (id: string) => utterances.filter((u) => u.speaker_id === id)
    console.log(utterances.map((u) => `${u.speaker_name} [${u.model_id}] ${u.text}`).join('\n'))

    expect(said(ada)).toHaveLength(2)
    expect(said(bob)).toHaveLength(4) // three clips, the last cut in two at 25 s
    expect(new Set(utterances.map((u) => u.model_id))).toEqual(new Set([FIRST, SECOND]))
    for (const line of said(ada)) expect(wer(JFK, line.text)).toBeLessThanOrEqual(0.1)
    const bobText = said(bob).map((u) => u.text).join(' ')
    expect(wer(Object.values(LIBRI).join(' '), bobText)).toBeLessThanOrEqual(0.15)
    expect(bobText.toLowerCase()).not.toContain('fellow americans')
    expect(said(ada).map((u) => u.text).join(' ').toLowerCase()).not.toContain('quilter')

    const md = await (await fetch(`${base}/api/sessions/${session.id}/export?format=md`, { headers: { Authorization: `Bearer ${TOKEN}` } })).text()
    expect(md).toMatch(/\*\*\[00:00:0\d\] Ada:\*\* And so,? my fellow Americans/)
  })
})
