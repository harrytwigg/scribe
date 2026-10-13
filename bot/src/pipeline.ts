// Per-user Opus -> PCM -> utterances -> scribe server.

import type { Readable } from 'node:stream'

import OpusScript from 'opusscript'

import type { ScribeClient, Speaker } from './client.js'
import { Segmenter } from './segmenter.js'
import type { VoiceSource } from './voice-source.js'

export const SILENCE_MS = 800

/** Smallest DAVE trailer: 8-byte tag, 1-byte nonce, the size byte itself, the 2-byte marker. */
const MIN_DAVE_TRAILER = 12

/**
 * True for a DAVE end-to-end encrypted frame. @discordjs/voice hands one through still encrypted
 * when it cannot decrypt it yet (just after joining, or mid key change). It must never reach the
 * decoder: about half of them "decode" as noise, which would be posted as the user's speech, and
 * decoding many corrupts opusscript's heap, which every decoder in the process shares.
 *
 * Every DAVE frame ends in the marker 0xFAFA, preceded by the size of its trailer (tag, nonce,
 * unencrypted ranges, that byte and the marker). Real Opus frames end in 0xFAFA far more often
 * than chance (about 1 in 1500 here), so the size byte must be plausible too. A real frame that
 * still matches costs 20 ms of audio.
 */
export function isEncryptedFrame(packet: Buffer): boolean {
  const n = packet.length
  if (n < MIN_DAVE_TRAILER || packet[n - 2] !== 0xfa || packet[n - 1] !== 0xfa) return false
  const trailer = packet[n - 3]
  return trailer >= MIN_DAVE_TRAILER && trailer <= n
}

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
  /** Packets that were not valid Opus. */
  corrupt: number
  /** Frames still DAVE-encrypted, dropped unheard: expected for a moment after the bot joins. */
  encrypted: number
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
    const heard = this.heard.get(userId) ?? { packets: 0, corrupt: 0, encrypted: 0, lines: 0 }
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
    let decoder: OpusScript
    try {
      decoder = new OpusScript(48_000, 2, OpusScript.Application.AUDIO)
    } catch (error) {
      onError?.(error)
      opus.destroy()
      return
    }
    this.active.set(userId, { segmenter, opus })
    let corrupt = 0
    let finished = false
    const finish = () => {
      if (finished) return
      finished = true
      // Release the user first: nothing below may stop them being heard next time.
      if (this.active.get(userId)?.opus === opus) {
        this.active.delete(userId)
        segmenter.flush()
      }
      try {
        decoder.delete()
      } catch (error) {
        onError?.(error)
      }
      if (corrupt) {
        void speakerPromise
          .then((speaker) => onError?.(new Error(`skipped ${corrupt} undecodable packet(s) from ${speaker.name}`)))
          .catch(() => {})
      }
    }
    opus.on('data', (packet: Buffer) => {
      if (finished) return
      heard.packets++
      if (isEncryptedFrame(packet)) {
        heard.encrypted++
        return
      }
      let pcm: Buffer
      try {
        pcm = decoder.decode(packet)
      } catch (error) {
        if (error instanceof Error && error.message.startsWith('Decode error')) {
          corrupt++
          heard.corrupt++
          return
        }
        // Anything else is the decoder itself failing, not the packet: end this stream (the user
        // is heard again on their next speaking event, with a new decoder) and say so.
        onError?.(error)
        opus.destroy()
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
