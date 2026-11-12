// Delivers a call's recording to the server, however long that takes (docs/spec.md, 4.4).
//
// A segment is written to local storage first, then sent; it is deleted only when the server has
// acknowledged it. A busy or unreachable server is retried with a growing wait (up to a minute)
// for as long as it takes, because the server stores segments without any model or queue and the
// recording exists precisely for the times it cannot transcribe. If the offscreen document is
// closed or the browser restarts first, the next one finds the leftovers and sends them.

import { ServerError } from './client.js'

const MAX_AGE_MS = 7 * 24 * 3600 * 1000
// The server will never take these: do not keep trying.
const REFUSED = new Set([400, 403, 404, 410, 413, 422])

/** Segments kept in memory only: for tests, and where IndexedDB is unavailable. */
export function memoryChunkStore() {
  const records = new Map()
  return {
    async put(record) {
      records.set(record.key, record)
    },
    async delete(key) {
      records.delete(key)
    },
    async all() {
      return [...records.values()]
    },
  }
}

/** Segments kept in the extension's IndexedDB, where a closed document or a restart cannot lose them. */
export function idbChunkStore(name = 'scribe-recordings') {
  let opened
  const open = () =>
    (opened ??= new Promise((resolve, reject) => {
      const request = indexedDB.open(name, 1)
      request.onupgradeneeded = () => request.result.createObjectStore('chunks', { keyPath: 'key' })
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    }))
  const run = async (mode, work) => {
    const db = await open()
    return new Promise((resolve, reject) => {
      const tx = db.transaction('chunks', mode)
      const result = work(tx.objectStore('chunks'))
      tx.oncomplete = () => resolve(result.result)
      tx.onerror = tx.onabort = () => reject(tx.error)
    })
  }
  return {
    put: (record) => run('readwrite', (store) => store.put(record)),
    delete: (key) => run('readwrite', (store) => store.delete(key)),
    all: () => run('readonly', (store) => store.getAll()),
  }
}

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

export class ChunkUploader {
  /**
   * @param {{clientFor: (server: string) => {putRecordingChunk: Function}, store?: object, backoffMs?: number[],
   *          maxAgeMs?: number, now?: () => number, sleep?: (ms: number) => Promise<void>}} options
   */
  constructor({ clientFor, store = memoryChunkStore(), backoffMs = [1000, 2000, 5000, 10_000, 30_000, 60_000], maxAgeMs = MAX_AGE_MS, now = Date.now, sleep = defaultSleep } = {}) {
    this.clientFor = clientFor
    this.store = store
    this.backoffMs = backoffMs
    this.maxAgeMs = maxAgeMs
    this.now = now
    this.sleep = sleep
    this.items = new Map() // key -> {record, attempts, due}
    this.refused = new Map() // session id -> why the server will take no more of it
    this.waiters = []
    this.sent = 0
    this.lastError = null
    this.closed = false
    this.onChange = () => {}
    this.onIdle = () => {}
  }

  get pending() {
    return this.items.size
  }

  /** Keep a segment and start delivering it. Resolves once it is stored locally. */
  async add(server, sessionId, segment) {
    if (this.refused.has(sessionId)) return
    const record = {
      key: `${server}|${sessionId}|${segment.channel}|${segment.index}`,
      server, sessionId, channel: segment.channel, index: segment.index,
      blob: segment.blob, mime: segment.mime, startedAt: segment.startedAt, durationMs: segment.durationMs,
      addedAt: this.now(),
    }
    // Stored before it is sent, so a delete after the send cannot overtake the write. If local
    // storage fails it is still sent: it is kept in memory for this run.
    await this.store.put(record).catch((error) => (this.lastError = `could not keep a recording segment on disk: ${error.message}`))
    this.queue(record)
  }

  /** Pick up segments an earlier document left behind. */
  async resume() {
    for (const record of await this.store.all().catch(() => [])) {
      if (this.now() - record.addedAt > this.maxAgeMs) await this.store.delete(record.key).catch(() => {})
      else if (!this.items.has(record.key)) this.queue(record)
    }
  }

  queue(record) {
    this.items.set(record.key, { record, attempts: 0, due: 0 })
    this.onChange()
    this.kick()
  }

  kick() {
    this.running ??= this.loop().finally(() => {
      this.running = null
      this.onChange()
      if (!this.items.size) {
        for (const resolve of this.waiters.splice(0)) resolve(true)
        this.onIdle()
      }
    })
  }

  async loop() {
    while (!this.closed && this.items.size) {
      const now = this.now()
      const waiting = [...this.items.values()]
      const next = waiting.find((item) => item.due <= now)
      if (!next) {
        await this.sleep(Math.max(Math.min(...waiting.map((item) => item.due)) - now, 50))
        continue
      }
      await this.send(next)
    }
  }

  async send(item) {
    const { record } = item
    if (this.now() - record.addedAt > this.maxAgeMs) return this.drop(record)
    try {
      const reply = await this.clientFor(record.server).putRecordingChunk(record.sessionId, record.channel, record.index, record)
      if (reply?.created !== false) this.sent++
      this.lastError = null
      await this.drop(record)
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : String(error)
      if (error instanceof ServerError && REFUSED.has(error.status)) {
        // Recording is off, or the call was deleted: nothing more of this session will be taken.
        if (error.status === 403 || error.status === 404) {
          this.refused.set(record.sessionId, this.lastError)
          for (const other of [...this.items.values()]) if (other.record.sessionId === record.sessionId) await this.drop(other.record)
        } else await this.drop(record)
      } else {
        item.attempts++
        item.due = this.now() + this.backoffMs[Math.min(item.attempts, this.backoffMs.length) - 1]
      }
    }
    this.onChange()
  }

  async drop(record) {
    this.items.delete(record.key)
    await this.store.delete(record.key).catch(() => {})
  }

  /** Resolves true once nothing is left to deliver, or false after `timeoutMs`. */
  whenIdle(timeoutMs) {
    if (!this.items.size) return Promise.resolve(true)
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), timeoutMs)
      this.waiters.push((value) => {
        clearTimeout(timer)
        resolve(value)
      })
    })
  }

  close() {
    this.closed = true
  }
}
