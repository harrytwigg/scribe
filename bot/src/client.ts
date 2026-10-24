// The scribe server's API, as the bot uses it (docs/spec.md, 3.2 and 3.3).

import { durationMs, wav } from './pcm.js'

export interface Speaker {
  id: string
  name: string
}

export interface PostResult {
  ok: boolean
  text?: string
  skipped?: boolean
  error?: string
}

export class ServerError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
  }
}

const RETRYABLE = new Set([502, 503, 504])

export class ScribeClient {
  constructor(
    private readonly baseUrl: string,
    private readonly options: { retryDelaysMs?: number[]; timeoutMs?: number } = {},
  ) {}

  private async request<T>(method: string, path: string, body?: BodyInit, contentType?: string): Promise<T> {
    const response = await fetch(new URL(path, this.baseUrl), {
      method,
      headers: contentType ? { 'Content-Type': contentType } : {},
      body,
      signal: AbortSignal.timeout(this.options.timeoutMs ?? 120_000),
    })
    if (!response.ok) {
      let detail = response.statusText
      try {
        detail = ((await response.json()) as { detail?: string }).detail ?? detail
      } catch {}
      throw new ServerError(response.status, detail)
    }
    return (await response.json()) as T
  }

  status() {
    return this.request<{ active_model: string | null; open_sessions: number }>('GET', '/api/status')
  }

  createSession(title: string, guild: string, channel: string) {
    return this.request<{ id: string }>(
      'POST',
      '/api/sessions',
      JSON.stringify({ source: 'discord', title, guild, channel }),
      'application/json',
    )
  }

  listSessions() {
    return this.request<{ id: string; source: string; guild: string | null; ended_at: number | null }[]>('GET', '/api/sessions')
  }

  endSession(id: string) {
    return this.request('POST', `/api/sessions/${id}/end`)
  }

  async optouts(): Promise<Set<string>> {
    return new Set(await this.request<string[]>('GET', '/api/optouts/discord'))
  }

  setOptout(userId: string, optedOut: boolean) {
    return this.request(
      'POST',
      '/api/optouts',
      JSON.stringify({ source: 'discord', user_id: userId, opted_out: optedOut }),
      'application/json',
    )
  }

  /**
   * Post one utterance. A busy or unreachable server is retried (1 s, 3 s); if it still fails,
   * a marker line records the gap instead, so the call is never blocked (docs/spec.md, 3.2).
   *
   * Every attempt carries the same key, so a retry after a timeout (when the first attempt may
   * still have been stored) cannot store the line twice. `allowed` is asked before every attempt:
   * a speaker who opts out meanwhile is never posted.
   */
  async postUtterance(
    sessionId: string,
    speaker: Speaker,
    pcm: Buffer,
    startedAt: number,
    allowed: () => boolean = () => true,
  ): Promise<PostResult> {
    const delays = this.options.retryDelaysMs ?? [1000, 3000]
    const key = `${speaker.id}:${startedAt}`
    let lastError = ''
    for (let attempt = 0; attempt <= delays.length; attempt++) {
      if (attempt > 0) await new Promise((resolve) => setTimeout(resolve, delays[attempt - 1]))
      if (!allowed()) return { ok: true, skipped: true }
      const form = new FormData()
      form.append('file', new Blob([new Uint8Array(wav(pcm))], { type: 'audio/wav' }), 'utterance.wav')
      form.append('speaker_id', speaker.id)
      form.append('speaker_name', speaker.name)
      form.append('started_at', String(startedAt))
      form.append('client_key', key)
      try {
        const result = await this.request<{ text?: string; skipped?: boolean }>(
          'POST',
          `/api/sessions/${sessionId}/utterances`,
          form,
        )
        return { ok: true, text: result.text, skipped: result.skipped }
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error)
        const retryable = !(error instanceof ServerError) || RETRYABLE.has(error.status)
        if (!retryable) return { ok: false, error: lastError }
      }
    }
    if (!allowed()) return { ok: true, skipped: true }
    const seconds = Math.round(durationMs(pcm) / 1000)
    await this.postMarker(sessionId, speaker, startedAt, `[unavailable: ${seconds} s of audio from ${speaker.name}]`).catch(
      () => {},
    )
    return { ok: false, error: lastError }
  }

  postMarker(sessionId: string, speaker: Speaker, startedAt: number, text: string) {
    return this.request(
      'POST',
      `/api/sessions/${sessionId}/markers`,
      JSON.stringify({ speaker_id: speaker.id, speaker_name: speaker.name, started_at: startedAt, text }),
      'application/json',
    )
  }
}
