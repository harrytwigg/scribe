// Cuts one channel of 16 kHz mono audio into utterances (docs/spec.md, 4).
//
// The same rules as the Discord bot's segmenter (3.2): an utterance ends after `silenceMs` of quiet
// or at `maxMs`, and one with less than `minSpeechMs` of speech is dropped. Speech here is plain
// energy at least `marginDb` above the channel's noise floor: the quietest 20 ms of the last
// `floorMs`, which follows a noisier room within seconds but not continuous speech. The server
// runs Silero VAD on every utterance anyway, so this only decides where to cut, not what is speech.

export const RATE = 16000
export const FRAME = 320 // 20 ms

export class Segmenter {
  /**
   * @param {(u: {samples: Float32Array, startSample: number}) => void} emit
   */
  constructor(emit, { silenceMs = 800, maxMs = 25000, minSpeechMs = 400, preRollMs = 300, minDb = -50, marginDb = 12, floorMs = 10000 } = {}) {
    this.emit = emit
    this.silenceFrames = Math.round(silenceMs / 20)
    this.maxFrames = Math.round(maxMs / 20)
    this.minSpeechFrames = Math.round(minSpeechMs / 20)
    this.preRollFrames = Math.round(preRollMs / 20)
    this.minDb = minDb
    this.marginDb = marginDb
    this.levels = [] // dB of the last `floorMs` of frames
    this.floorFrames = Math.round(floorMs / 20)
    this.pending = new Float32Array(0) // samples not yet making a whole frame
    this.position = 0 // index of the next whole frame's first sample
    this.preRoll = [] // the last few quiet frames, so a word's onset is kept
    this.frames = null // frames of the open utterance
    this.startSample = 0
    this.speech = 0
    this.quiet = 0
  }

  /** Feed any number of samples. */
  push(samples) {
    let buffer = samples
    if (this.pending.length) {
      buffer = new Float32Array(this.pending.length + samples.length)
      buffer.set(this.pending)
      buffer.set(samples, this.pending.length)
    }
    let offset = 0
    for (; offset + FRAME <= buffer.length; offset += FRAME) this.frame(buffer.slice(offset, offset + FRAME))
    this.pending = buffer.slice(offset)
  }

  /** Close any open utterance (the call is ending). */
  flush() {
    this.close()
  }

  frame(frame) {
    const start = this.position
    this.position += FRAME
    const db = level(frame)
    this.levels.push(db)
    if (this.levels.length > this.floorFrames) this.levels.shift()
    const loud = db > Math.max(this.minDb, Math.min(...this.levels) + this.marginDb)

    if (this.frames === null) {
      if (!loud) {
        this.preRoll.push(frame)
        if (this.preRoll.length > this.preRollFrames) this.preRoll.shift()
        return
      }
      this.frames = [...this.preRoll]
      this.startSample = start - this.preRoll.length * FRAME
      this.preRoll = []
      this.speech = 0
      this.quiet = 0
    }
    this.frames.push(frame)
    if (loud) {
      this.speech++
      this.quiet = 0
    } else {
      this.quiet++
    }
    if (this.quiet >= this.silenceFrames || this.frames.length >= this.maxFrames) this.close()
  }

  close() {
    if (this.frames === null) return
    const { frames, speech, startSample } = this
    this.frames = null
    if (speech < this.minSpeechFrames) return
    // Trim the trailing quiet, keeping a little of it.
    const keep = frames.length - Math.max(0, this.quiet - 10)
    const samples = new Float32Array(keep * FRAME)
    for (let i = 0; i < keep; i++) samples.set(frames[i], i * FRAME)
    this.emit({ samples, startSample })
  }
}

function level(frame) {
  let sum = 0
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i]
  return 10 * Math.log10(sum / frame.length + 1e-12)
}
