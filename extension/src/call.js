// One transcribed call: the tab's audio ("them") and the mic ("me") as two channels, each cut into
// utterances and posted to the scribe server (docs/spec.md, 4). Runs in the offscreen document.
//
// Each channel is also recorded whole (recorder.js) and uploaded as it goes (uploader.js), on a
// path of its own: a server too busy to transcribe still keeps the audio, and the gaps in the
// transcript are filled from it afterwards.

import { ScribeClient, ServerError } from './client.js'
import { ChannelRecorder } from './recorder.js'
import { RATE, Segmenter } from './segmenter.js'
import { ChunkUploader } from './uploader.js'

const HEARTBEAT_MS = 30_000
// When the call ends, utterances still being retried get this long before they become markers.
const LIVE_GRACE_MS = 30_000
// And the recording this long to reach the server; what is left is kept and sent later.
const UPLOAD_GRACE_MS = 15_000

export class Call {
  /**
   * @param {{server: string, myName: string, othersName: string, separateSpeakers?: boolean}} config
   * @param {(status: object) => void} report
   */
  constructor(config, report, { idleStopMs = 10 * 60_000, uploader, recorderOptions, clientOptions, liveGraceMs = LIVE_GRACE_MS, uploadGraceMs = UPLOAD_GRACE_MS } = {}) {
    this.idleStopMs = idleStopMs
    this.liveGraceMs = liveGraceMs
    this.uploadGraceMs = uploadGraceMs
    this.config = config
    this.report = report
    this.client = new ScribeClient(config.server, clientOptions)
    // Shared across calls when the document passes one in, so a recording still being delivered
    // outlives the call that made it.
    this.uploader = uploader ?? new ChunkUploader({ clientFor: () => this.client })
    this.recorderOptions = recorderOptions
    this.recorders = []
    this.adding = new Set() // segments being written to local storage
    this.giveUp = new AbortController() // ends the waiting on a busy server when the call is over
    this.contexts = []
    this.inflight = new Set()
    this.lines = 0
    this.failed = 0
    this.sessionId = null
    this.stopping = null
    this.done = false // fully stopped: session ended, nothing left in flight
  }

  /**
   * @param {string} title
   * @param {{tab: MediaStream, mic: MediaStream | null}} streams
   */
  async start(title, streams) {
    this.streams = streams
    let session
    try {
      session = await this.client.createSession(title)
    } catch (error) {
      this.done = true
      throw error
    }
    this.sessionId = session.id
    try {
      await this.capture(streams)
    } catch (error) {
      this.error = error.message
      await this.stop()
      throw error
    }
    this.heard = Date.now()
    this.timer = setInterval(() => this.tick(), HEARTBEAT_MS)
    this.publish('recording')
    return session
  }

  /**
   * Every 30 s: tell the server the call is alive, and give up on one that is over. Leaving a Meet
   * call can keep the tab (and its capture) open, so a long silence on every channel ends it.
   */
  async tick() {
    if (this.stopping) return
    if (Date.now() - this.heard > this.idleStopMs) {
      this.reason = `no speech for ${Math.round(this.idleStopMs / 60_000)} min`
      return this.stop()
    }
    try {
      await this.client.heartbeat(this.sessionId)
    } catch (error) {
      if (error instanceof ServerError && (error.status === 404 || error.status === 409)) {
        this.reason = 'the server ended the session'
        this.stop()
      }
    }
  }

  /** Start recording a channel. The call carries on without a recording if the browser cannot. */
  record(stream, channel) {
    try {
      const recorder = new ChannelRecorder(stream, channel, (segment) => {
        const adding = this.uploader.add(this.config.server, this.sessionId, segment).catch(() => {})
        this.adding.add(adding)
        adding.finally(() => this.adding.delete(adding))
      }, this.recorderOptions)
      recorder.start()
      this.recorders.push(recorder)
    } catch (error) {
      this.recordingError = `not recording the ${channel}: ${error.message}`
    }
  }

  async capture(streams) {
    // First, so the recording starts with the call and not after the audio worklets have loaded.
    if (this.config.record !== false) {
      this.record(streams.tab, 'tab')
      if (streams.mic) this.record(streams.mic, 'mic')
    }
    // Capturing a tab silences it for the user, so play it back.
    const playback = new AudioContext()
    playback.createMediaStreamSource(streams.tab).connect(playback.destination)
    this.contexts.push(playback)
    await this.channel(streams.tab, { id: 'tab', name: this.config.othersName, diarize: this.config.separateSpeakers !== false })
    if (streams.mic) await this.channel(streams.mic, { id: 'mic', name: this.config.myName })
    // The tab closing ends its track, and with it the call. (Navigating does not: the background
    // stops the call when the captured tab loads a new page.)
    for (const track of streams.tab.getAudioTracks()) track.addEventListener('ended', () => this.stop())
  }

  async channel(stream, speaker) {
    // A 16 kHz context: the browser resamples the stream into it.
    const context = new AudioContext({ sampleRate: RATE })
    this.contexts.push(context)
    await context.audioWorklet.addModule(new URL('./worklet.js', import.meta.url))
    const tap = new AudioWorkletNode(context, 'scribe-tap')
    const t0 = Date.now() / 1000
    const segmenter = new Segmenter(({ samples, startSample }) => this.post(speaker, samples, t0 + startSample / RATE))
    tap.port.onmessage = (event) => segmenter.push(event.data)
    ;(this.taps ??= []).push(tap)
    context.createMediaStreamSource(stream).connect(tap)
    // A worklet only runs while connected to the output; a muted gain keeps it running silently.
    const mute = context.createGain()
    mute.gain.value = 0
    tap.connect(mute).connect(context.destination)
    ;(this.segmenters ??= []).push(segmenter)
  }

  post(speaker, samples, startedAt) {
    // Speech was heard, whether or not the server manages to transcribe it: a call on a struggling
    // server is not a silent one (and must not be ended as if it were).
    this.heard = Date.now()
    const job = this.client.postUtterance(this.sessionId, speaker, samples, startedAt, { signal: this.giveUp.signal }).then((result) => {
      if (result.ok && result.text) {
        this.lines += result.lines.length || 1
        const line = result.lines.at(-1)
        this.last = `${line?.speaker_name ?? speaker.name}: ${line?.text ?? result.text}`
      } else if (!result.ok) {
        this.failed++
        this.error = result.error
      }
      this.publish('recording')
    })
    this.inflight.add(job)
    job.finally(() => this.inflight.delete(job))
  }

  publish(state = this.state) {
    this.state = state
    this.report({
      state,
      sessionId: this.sessionId,
      lines: this.lines,
      failed: this.failed,
      last: this.last,
      error: this.error,
      reason: this.reason,
      recording: this.recorders.length > 0,
      recordingError: this.recordingError ?? this.uploader.lastError ?? undefined,
      pendingChunks: this.uploader.pending,
    })
  }

  /** Stop capturing, wait for every utterance to land, then end the session. */
  stop() {
    this.stopping ??= (async () => {
      clearInterval(this.timer)
      this.publish('stopping')
      // Recorders first: they hand over their last segment when stopped, before the tracks end.
      await Promise.all(this.recorders.map((recorder) => recorder.stop().catch(() => {})))
      for (const stream of Object.values(this.streams ?? {})) stream?.getTracks().forEach((track) => track.stop())
      await Promise.all(this.contexts.map((context) => context.close().catch(() => {})))
      // Let audio already posted from the worklet arrive, then cut what is left.
      await new Promise((resolve) => setTimeout(resolve, 100))
      for (const tap of this.taps ?? []) tap.port.onmessage = null
      for (const segmenter of this.segmenters ?? []) segmenter.flush()
      // Utterances a struggling server is still working on get a little longer; after that they
      // become markers, which the server fills from the recording.
      const settled = Promise.allSettled([...this.inflight])
      await Promise.race([settled, new Promise((resolve) => setTimeout(resolve, this.liveGraceMs))])
      this.giveUp.abort()
      await settled
      // The recording gets a moment too; what is left stays in local storage and is sent later.
      await Promise.allSettled([...this.adding]) // the last segment must be stored (or sent) before anything waits on the queue
      await this.uploader.whenIdle(this.uploadGraceMs)
      if (this.sessionId) await this.client.endSession(this.sessionId).catch((error) => (this.error = error.message))
      this.done = true
      this.publish('stopped')
    })()
    return this.stopping
  }
}
