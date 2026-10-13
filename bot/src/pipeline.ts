// Per-user Opus -> PCM -> utterances -> scribe server.

import type { Readable } from 'node:stream'

import OpusScript from 'opusscript'

import type { ScribeClient, Speaker } from './client.js'
import { Segmenter } from './segmenter.js'
import type { VoiceSource } from './voice-source.js'

export const SILENCE_MS = 800

export interface PipelineOptions {
  source: VoiceSource
  client: ScribeClient
  sessionId: string
  speaker: (userId: string) => Promise<Speaker> | Speaker
  /** Opted-out users are never subscribed. */
  isOptedOut: (userId: string) => boolean
  onLine?: (speaker: Speaker, text: string) => void
  onError?: (error: unknown) => void
}

/** What has been heard from one user this call, for `/scribe status` and the end-of-call log. */
export interface Heard {
  packets: number
  /** Packets that would not decode: a DAVE frame before the session can decrypt it, or damage. */
  corrupt: number
  /** Lines the server transcribed (utterances it found no speech in are not counted). */
  lines: number
}

export class Pipeline {
  private readonly active = new Map<string, { segmenter: Segmenter; opus: Readable }>()
  readonly heard = new Map<string, Heard>()
  private readonly pending = new Set<Promise<unknown>>()
  private stopped = false

  constructor(private readonly options: PipelineOptions) {
    options.source.onSpeaking((userId) => this.listen(userId))
  }

  private listen(userId: string): void {
    const { source, client, sessionId, isOptedOut, onLine, onError } = this.options
    if (this.stopped || this.active.has(userId) || isOptedOut(userId)) return
    const speakerPromise = Promise.resolve(this.options.speaker(userId))
    const heard = this.heard.get(userId) ?? { packets: 0, corrupt: 0, lines: 0 }
    this.heard.set(userId, heard)
    const segmenter = new Segmenter((utterance) => {
      const job = speakerPromise
        .then((speaker) =>
          client.postUtterance(sessionId, speaker, utterance.pcm, utterance.startedAt, () => !isOptedOut(userId)).then((result) => {
            if (result.ok && result.text) {
              heard.lines++
              onLine?.(speaker, result.text)
            }
            if (!result.ok) onError?.(new Error(`utterance from ${speaker.name} lost: ${result.error}`))
          }),
        )
        .catch((error) => onError?.(error))
        .finally(() => this.pending.delete(job))
      this.pending.add(job)
    })
    const opus = source.subscribe(userId, SILENCE_MS)
    // Each packet is decoded on its own, so a bad one costs that packet only. A stream decoder
    // (prism's Transform) is destroyed by its first bad packet and then silently stops the user
    // being heard for the rest of the call: the bug that lost a whole speaker (docs/spec.md, 3.2).
    const decoder = new OpusScript(48_000, 2, OpusScript.Application.AUDIO)
    this.active.set(userId, { segmenter, opus })
    let corrupt = 0
    let finished = false
    const finish = () => {
      if (finished) return
      finished = true
      decoder.delete()
      if (corrupt) {
        void speakerPromise
          .then((speaker) => onError?.(new Error(`skipped ${corrupt} undecodable packet(s) from ${speaker.name}`)))
          .catch(() => {})
      }
      // Dropped (opt-out) or stopped streams are already out of `active`, and must not flush here.
      if (this.active.get(userId)?.opus !== opus) return
      this.active.delete(userId)
      segmenter.flush()
    }
    opus.on('data', (packet: Buffer) => {
      if (finished) return
      heard.packets++
      let pcm: Buffer
      try {
        pcm = decoder.decode(packet)
      } catch {
        corrupt++
        heard.corrupt++
        return
      }
      segmenter.push(pcm)
    })
    opus.on('error', (error) => onError?.(error))
    opus.on('end', finish)
    // Destroyed without ending (an error, an opt-out, the call stopping): still release the user.
    opus.on('close', finish)
  }

  /** Stop listening to one user at once, discarding what they were saying (opt-out). */
  drop(userId: string): void {
    const listener = this.active.get(userId)
    if (!listener) return
    this.active.delete(userId)
    listener.opus.destroy()
  }

  /** Stop listening; send what each speaker had said so far, and wait for it to reach the server. */
  async stop(): Promise<void> {
    this.stopped = true
    for (const [userId, { segmenter, opus }] of this.active) {
      this.active.delete(userId)
      segmenter.flush()
      opus.destroy()
    }
    while (this.pending.size) await Promise.allSettled([...this.pending])
  }
}
