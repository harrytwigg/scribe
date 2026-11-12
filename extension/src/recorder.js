// Records one channel of a call to audio files, off the path that transcribes it (docs/spec.md, 4.4).
//
// A MediaRecorder encodes the channel's stream (Opus in WebM, which is small: about 24 kbit/s) in
// its own thread, so it adds nothing to the time a line takes to appear. Every `segmentMs` it is
// replaced by a new one, so each segment is a complete file that decodes on its own: losing or
// damaging one costs its 30 s, not the rest of the call. The new recorder starts `overlapMs`
// before the old one stops, so no audio falls between two segments; the server places each by its
// wall-clock start time and, where two overlap, takes the later.

export const SEGMENT_MS = 30_000
export const OVERLAP_MS = 400
export const BITS_PER_SECOND = 24_000

const MIME_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']

/** The first format this browser can record, or undefined to let it choose. */
function pickMime(Recorder) {
  return MIME_TYPES.find((type) => Recorder.isTypeSupported?.(type))
}

/** A timer that can be replaced in tests: returns a function that cancels it. */
const realSchedule = (fn, ms) => {
  const timer = setTimeout(fn, ms)
  return () => clearTimeout(timer)
}

export class ChannelRecorder {
  /**
   * @param {MediaStream} stream
   * @param {string} channel  "tab" or "mic"
   * @param {(segment: {channel: string, index: number, blob: Blob, mime: string, startedAt: number, durationMs: number}) => void} onSegment
   */
  constructor(stream, channel, onSegment, options = {}) {
    const { segmentMs = SEGMENT_MS, overlapMs = OVERLAP_MS, bitsPerSecond = BITS_PER_SECOND, Recorder = globalThis.MediaRecorder, now = Date.now, schedule = realSchedule } = options
    this.stream = stream
    this.channel = channel
    this.onSegment = onSegment
    this.segmentMs = segmentMs
    this.overlapMs = overlapMs
    this.bitsPerSecond = bitsPerSecond
    this.Recorder = Recorder
    this.now = now
    this.schedule = schedule
    this.index = 0
    this.live = new Set() // recorders still running: the current one, and the previous during the overlap
    this.cancels = new Set() // timers waiting to stop an old recorder
    this.stopped = false
    this.error = null
  }

  /** Throws if this browser cannot record the stream. */
  start() {
    if (!this.Recorder) throw new Error('this browser has no MediaRecorder')
    this.mime = pickMime(this.Recorder)
    this.current = this.begin()
    this.cancelTimer = this.schedule(() => this.rotate(), this.segmentMs)
  }

  begin() {
    const options = { audioBitsPerSecond: this.bitsPerSecond }
    if (this.mime) options.mimeType = this.mime
    const recorder = new this.Recorder(this.stream, options)
    const part = { index: this.index, startedAt: this.now() / 1000, blobs: [], recorder }
    recorder.ondataavailable = (event) => {
      if (event.data?.size) part.blobs.push(event.data)
    }
    // The clock starts when the recorder says it has: its first sample is no earlier than that.
    recorder.onstart = () => {
      part.startedAt = this.now() / 1000
    }
    recorder.onerror = (event) => {
      this.error = event?.error?.message ?? 'the recorder failed'
    }
    part.done = new Promise((resolve) => {
      recorder.onstop = () => {
        this.live.delete(part)
        this.deliver(part)
        resolve()
      }
    })
    recorder.start() // throws if the stream has ended; then the index is not used
    this.index++
    this.live.add(part)
    return part
  }

  deliver(part) {
    if (!part.blobs.length) return
    const mime = part.recorder.mimeType || this.mime || 'audio/webm'
    const durationMs = Math.max(0, this.now() - part.startedAt * 1000)
    this.onSegment({ channel: this.channel, index: part.index, blob: new Blob(part.blobs, { type: mime }), mime, startedAt: part.startedAt, durationMs })
  }

  rotate() {
    if (this.stopped) return
    const old = this.current
    try {
      this.current = this.begin()
      this.error = null
    } catch (error) {
      // The stream has ended, or the browser refused: keep the old segment going and try again.
      this.error = error.message
      this.cancelTimer = this.schedule(() => this.rotate(), this.segmentMs)
      return
    }
    this.cancelTimer = this.schedule(() => this.rotate(), this.segmentMs)
    const cancel = this.schedule(() => {
      this.cancels.delete(cancel)
      this.finish(old)
    }, this.overlapMs)
    this.cancels.add(cancel)
  }

  finish(part) {
    if (part.recorder.state !== 'inactive') part.recorder.stop()
  }

  /** Stop recording; resolves once the last segment has been handed to `onSegment`. */
  async stop() {
    this.stopped = true
    this.cancelTimer?.()
    for (const cancel of this.cancels) cancel()
    this.cancels.clear()
    const parts = [...this.live]
    for (const part of parts) {
      try {
        this.finish(part)
      } catch {} // already stopped by the browser: its stop event has delivered it
    }
    await Promise.all(parts.map((part) => part.done))
  }
}
