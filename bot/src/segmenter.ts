// Cuts one speaker's PCM into utterances (docs/spec.md, 3.2): at the end of their speech (the
// receive stream ends after 800 ms of silence), or at 25 s, whichever comes first. Anything shorter
// than 400 ms is dropped as noise.

import { BYTES_PER_MS_MONO, downmix } from './pcm.js'

export const MAX_UTTERANCE_MS = 25_000
export const MIN_UTTERANCE_MS = 400

export interface Utterance {
  /** s16le mono, 48 kHz */
  pcm: Buffer
  /** wall-clock start, epoch seconds */
  startedAt: number
}

export class Segmenter {
  private chunks: Buffer[] = []
  private bytes = 0
  private startedAt = 0

  constructor(
    private readonly emit: (utterance: Utterance) => void,
    private readonly now: () => number = () => Date.now() / 1000,
    private readonly maxMs = MAX_UTTERANCE_MS,
    private readonly minMs = MIN_UTTERANCE_MS,
  ) {}

  /** Feed decoded 48 kHz stereo s16le PCM. */
  push(stereo: Buffer): void {
    let mono = downmix(stereo)
    while (mono.length > 0) {
      if (this.bytes === 0) this.startedAt = this.now()
      const room = this.maxMs * BYTES_PER_MS_MONO - this.bytes
      const take = mono.subarray(0, room)
      this.chunks.push(take)
      this.bytes += take.length
      mono = mono.subarray(take.length)
      if (this.bytes >= this.maxMs * BYTES_PER_MS_MONO) this.flush()
    }
  }

  /** The speaker went quiet (or left): emit what is buffered. */
  flush(): void {
    if (this.bytes >= this.minMs * BYTES_PER_MS_MONO) {
      this.emit({ pcm: Buffer.concat(this.chunks), startedAt: this.startedAt })
    }
    this.chunks = []
    this.bytes = 0
  }
}
