// The scribe server's API, as the extension uses it. Mirrors the bot's client (bot/src/client.ts):
// the same idempotency key and the same marker when audio cannot be delivered. A busy server is
// waited on far longer than the bot waits (about five minutes): a laptop that is struggling to
// transcribe a call recovers in minutes, and what is still undelivered after that is not lost, it
// is a marker the server fills from the call's recording (recorder.js).

import { RATE } from './segmenter.js'
import { wav } from './wav.js'

export class ServerError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

const RETRYABLE = new Set([408, 425, 429, 502, 503, 504])

// Seconds to wait before each retry of an utterance: about five minutes in all.
export const LIVE_RETRY_DELAYS_MS = [1, 3, 5, 10, 15, 30, 30, 30, 30, 30, 30, 30, 30, 30].map((s) => s * 1000)

/** Waits `ms` (a little randomly, so retries of many utterances do not arrive together), or until aborted. */
function pause(ms, signal) {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve()
    const done = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', done)
      resolve()
    }
    const timer = setTimeout(done, ms * (0.85 + Math.random() * 0.3))
    signal?.addEventListener('abort', done)
  })
}

export class ScribeClient {
  constructor(baseUrl, { retryDelaysMs = LIVE_RETRY_DELAYS_MS, timeoutMs = 120_000, fetch: fetchImpl } = {}) {
    this.baseUrl = baseUrl
    this.retryDelaysMs = retryDelaysMs
    this.timeoutMs = timeoutMs
    // Unbound, a browser's fetch throws "Illegal invocation".
    this.fetch = fetchImpl ?? ((...args) => globalThis.fetch(...args))
  }

  async request(method, path, body, contentType, signal) {
    const headers = {}
    if (contentType) headers['Content-Type'] = contentType
    const timeout = AbortSignal.timeout(this.timeoutMs)
    const response = await this.fetch(new URL(path, this.baseUrl), {
      method,
      headers,
      body,
      signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
    })
    if (!response.ok) {
      let detail = response.statusText
      try {
        detail = (await response.json()).detail ?? detail
      } catch {}
      throw new ServerError(response.status, detail)
    }
    return response.json()
  }

  status() {
    return this.request('GET', '/api/status')
  }

  createSession(title) {
    return this.request('POST', '/api/sessions', JSON.stringify({ source: 'browser', title }), 'application/json')
  }

  /** Keeps a quiet call open: the server ends browser sessions it stops hearing from. */
  heartbeat(id) {
    return this.request('POST', `/api/sessions/${id}/heartbeat`)
  }

  endSession(id) {
    return this.request('POST', `/api/sessions/${id}/end`)
  }

  postMarker(sessionId, speaker, startedAt, text, durationMs = 0, clientKey = null) {
    return this.request(
      'POST',
      `/api/sessions/${sessionId}/markers`,
      JSON.stringify({ speaker_id: speaker.id, speaker_name: speaker.name, started_at: startedAt, text, duration_ms: durationMs, client_key: clientKey }),
      'application/json',
    )
  }

  /**
   * One segment of a channel's recording. Safe to send again until it is acknowledged: the server
   * keeps the first copy of an index and says so (`created: false`) to the rest.
   */
  putRecordingChunk(sessionId, channel, index, { blob, mime, startedAt, durationMs, sampleRate = 0 }) {
    const query = new URLSearchParams({ started_at: String(startedAt), duration_ms: String(Math.round(durationMs)), sample_rate: String(sampleRate) })
    return this.request('PUT', `/api/sessions/${sessionId}/recording/${channel}/${index}?${query}`, blob, mime || 'audio/webm')
  }

  /**
   * Post one utterance. A busy or unreachable server is retried; if it still fails (or `signal`
   * aborts the waiting: the call is over), a marker line records the gap and how long it was, for
   * the server to fill from the recording. Every attempt carries the same key, so a retry cannot
   * store the line twice.
   * @returns {Promise<{ok: boolean, text?: string, skipped?: boolean, error?: string}>}
   */
  async postUtterance(sessionId, speaker, samples, startedAt, { signal } = {}) {
    const body = wav(samples, RATE)
    const seconds = Math.round(samples.length / RATE)
    const durationMs = Math.round((samples.length / RATE) * 1000)
    const key = `${speaker.id}:${startedAt}`
    let lastError = ''
    for (let attempt = 0; attempt <= this.retryDelaysMs.length; attempt++) {
      if (attempt > 0) {
        await pause(this.retryDelaysMs[attempt - 1], signal)
        if (signal?.aborted) break
      }
      const form = new FormData()
      form.append('file', new Blob([body], { type: 'audio/wav' }), 'utterance.wav')
      form.append('speaker_id', speaker.id)
      form.append('speaker_name', speaker.name)
      form.append('started_at', String(startedAt))
      form.append('client_key', key)
      // A channel carrying several voices (the call tab) is split into speakers on the server.
      if (speaker.diarize) form.append('diarize', 'true')
      try {
        const result = await this.request('POST', `/api/sessions/${sessionId}/utterances`, form, undefined, signal)
        return { ok: true, text: result.text, skipped: result.skipped, lines: result.lines ?? (result.text ? [result] : []) }
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error)
        const retryable = !(error instanceof ServerError) || RETRYABLE.has(error.status)
        if (!retryable) return { ok: false, error: lastError }
      }
    }
    await this.postMarker(sessionId, speaker, startedAt, `[unavailable: ${seconds} s of audio from ${speaker.name}]`, durationMs, key).catch(() => {})
    return { ok: false, error: lastError || 'gave up waiting for the server' }
  }
}
