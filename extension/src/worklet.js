// Runs on the audio thread: hands each render quantum, downmixed to mono, to the page.
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const channels = inputs[0]
    if (channels && channels.length) {
      const mono = new Float32Array(channels[0].length)
      for (const channel of channels) for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / channels.length
      this.port.postMessage(mono, [mono.buffer])
    }
    return true
  }
}

registerProcessor('scribe-tap', Tap)
