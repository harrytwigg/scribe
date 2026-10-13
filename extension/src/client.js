// The scribe server's API, as the extension uses it. Mirrors the bot's client (bot/src/client.ts):
// same retries, same idempotency key, same marker when audio cannot be delivered.

import { RATE } from './segmenter.js'
import { wav } from './wav.js'

export class ServerError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

const RETRYABLE = new Set([502, 503, 504])

export class ScribeClient {
  constructor(baseUrl, token, { retryDelaysMs = [1000, 3000], timeoutMs = 120_000, fetch: fetchImpl } = {}) {
    this.baseUrl = baseUrl
    this.token = token
    this.retryDelaysMs = retryDelaysMs
    this.timeoutMs = timeoutMs
    // Unbound, a browser's fetch throws "Illegal invocation".
    this.fetch = fetchImpl ?? ((...args) => globalThis.fetch(...args))
  }

  async request(method, path, body, contentType) {
    const headers = this.token ? { Authorization: `Bearer ${this.token}` } : {}
    if (contentType) headers['Content-Type'] = contentType
    const response = await this.fetch(new URL(path, this.baseUrl), {
      method,
      headers,
      body,
      signal: AbortSignal.timeout(this.timeoutMs),
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

  postMarker(sessionId, speaker, startedAt, text) {
    return this.request(
      'POST',
      `/api/sessions/${sessionId}/markers`,
      JSON.stringify({ speaker_id: speaker.id, speaker_name: speaker.name, started_at: startedAt, text }),
      'application/json',
    )
  }

  /**
   * Post one utterance. A busy or unreachable server is retried; if it still fails, a marker line
   * records the gap. Every attempt carries the same key, so a retry cannot store the line twice.
   * @returns {Promise<{ok: boolean, text?: string, skipped?: boolean, error?: string}>}
   */
  async postUtterance(sessionId, speaker, samples, startedAt) {
    const body = wav(samples, RATE)
    const key = `${speaker.id}:${startedAt}`
    let lastError = ''
    for (let attempt = 0; attempt <= this.retryDelaysMs.length; attempt++) {
      if (attempt > 0) await new Promise((resolve) => setTimeout(resolve, this.retryDelaysMs[attempt - 1]))
      const form = new FormData()
      form.append('file', new Blob([body], { type: 'audio/wav' }), 'utterance.wav')
      form.append('speaker_id', speaker.id)
      form.append('speaker_name', speaker.name)
      form.append('started_at', String(startedAt))
      form.append('client_key', key)
      try {
        const result = await this.request('POST', `/api/sessions/${sessionId}/utterances`, form)
        return { ok: true, text: result.text, skipped: result.skipped }
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error)
        const retryable = !(error instanceof ServerError) || RETRYABLE.has(error.status)
        if (!retryable) return { ok: false, error: lastError }
      }
    }
    const seconds = Math.round(samples.length / RATE)
    await this.postMarker(sessionId, speaker, startedAt, `[unavailable: ${seconds} s of audio from ${speaker.name}]`).catch(() => {})
    return { ok: false, error: lastError }
  }
}
