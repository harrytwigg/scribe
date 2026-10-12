// Per-user Opus -> PCM -> utterances -> scribe server.

import type { Readable } from 'node:stream'

import prism from 'prism-media'

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

export class Pipeline {
  private readonly active = new Map<string, { segmenter: Segmenter; opus: Readable }>()
  private readonly pending = new Set<Promise<unknown>>()
  private stopped = false

  constructor(private readonly options: PipelineOptions) {
    options.source.onSpeaking((userId) => this.listen(userId))
  }

  private listen(userId: string): void {
    const { source, client, sessionId, isOptedOut, onLine, onError } = this.options
    if (this.stopped || this.active.has(userId) || isOptedOut(userId)) return
    const speakerPromise = Promise.resolve(this.options.speaker(userId))
    const segmenter = new Segmenter((utterance) => {
      const job = speakerPromise
        .then((speaker) =>
          client.postUtterance(sessionId, speaker, utterance.pcm, utterance.startedAt).then((result) => {
            if (result.ok && result.text) onLine?.(speaker, result.text)
            if (!result.ok) onError?.(new Error(`utterance from ${speaker.name} lost: ${result.error}`))
          }),
        )
        .catch((error) => onError?.(error))
        .finally(() => this.pending.delete(job))
      this.pending.add(job)
    })
    const opus = source.subscribe(userId, SILENCE_MS)
    const decoder = new prism.opus.Decoder({ rate: 48_000, channels: 2, frameSize: 960 })
    this.active.set(userId, { segmenter, opus })
    const done = () => {
      if (this.active.get(userId)?.opus !== opus) return
      this.active.delete(userId)
      segmenter.flush()
    }
    // A corrupt packet (a DAVE decrypt failure surfaces here) loses that packet, not the stream.
    decoder.on('error', (error) => onError?.(error))
    opus.on('error', (error) => {
      onError?.(error)
      done()
    })
    decoder.on('data', (pcm: Buffer) => segmenter.push(pcm))
    decoder.on('end', done)
    opus.on('end', () => decoder.end())
    opus.pipe(decoder, { end: false })
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

  get speaking(): number {
    return this.active.size
  }
}
