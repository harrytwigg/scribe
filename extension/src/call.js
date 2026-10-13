// One transcribed call: the tab's audio ("them") and the mic ("me") as two channels, each cut into
// utterances and posted to the scribe server (docs/spec.md, 4). Runs in the offscreen document.

import { ScribeClient } from './client.js'
import { RATE, Segmenter } from './segmenter.js'

export class Call {
  /**
   * @param {{server: string, token: string, myName: string, othersName: string}} config
   * @param {(status: object) => void} report
   */
  constructor(config, report) {
    this.config = config
    this.report = report
    this.client = new ScribeClient(config.server, config.token)
    this.contexts = []
    this.inflight = new Set()
    this.lines = 0
    this.failed = 0
    this.sessionId = null
    this.stopping = null
  }

  /**
   * @param {string} title
   * @param {{tab: MediaStream, mic: MediaStream | null}} streams
   */
  async start(title, streams) {
    this.streams = streams
    const session = await this.client.createSession(title)
    this.sessionId = session.id
    try {
      await this.capture(streams)
    } catch (error) {
      this.error = error.message
      await this.stop()
      throw error
    }
    this.publish('recording')
    return session
  }

  async capture(streams) {
    // Capturing a tab silences it for the user, so play it back.
    const playback = new AudioContext()
    playback.createMediaStreamSource(streams.tab).connect(playback.destination)
    this.contexts.push(playback)
    await this.channel(streams.tab, { id: 'tab', name: this.config.othersName })
    if (streams.mic) await this.channel(streams.mic, { id: 'mic', name: this.config.myName })
    // The tab closing or navigating away ends its track, and with it the call.
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
    const job = this.client.postUtterance(this.sessionId, speaker, samples, startedAt).then((result) => {
      if (result.ok && result.text) {
        this.lines++
        this.last = `${speaker.name}: ${result.text}`
      } else if (!result.ok) {
        this.failed++
        this.error = result.error
      }
      this.publish('recording')
    })
    this.inflight.add(job)
    job.finally(() => this.inflight.delete(job))
  }

  publish(state) {
    this.report({ state, sessionId: this.sessionId, lines: this.lines, failed: this.failed, last: this.last, error: this.error })
  }

  /** Stop capturing, wait for every utterance to land, then end the session. */
  stop() {
    this.stopping ??= (async () => {
      this.publish('stopping')
      for (const stream of Object.values(this.streams ?? {})) stream?.getTracks().forEach((track) => track.stop())
      await Promise.all(this.contexts.map((context) => context.close().catch(() => {})))
      // Let audio already posted from the worklet arrive, then cut what is left.
      await new Promise((resolve) => setTimeout(resolve, 100))
      for (const tap of this.taps ?? []) tap.port.onmessage = null
      for (const segmenter of this.segmenters ?? []) segmenter.flush()
      await Promise.allSettled([...this.inflight])
      if (this.sessionId) await this.client.endSession(this.sessionId).catch((error) => (this.error = error.message))
      this.publish('stopped')
    })()
    return this.stopping
  }
}
