// Stand-ins for the browser's recorder and clock, shared by the recording tests.

/** Emits its data and then `stop`, as a real MediaRecorder does. */
export class FakeRecorder {
  static instances = []
  static isTypeSupported = (type) => type.startsWith('audio/webm')

  constructor(stream, options) {
    this.stream = stream
    this.options = options
    this.mimeType = options.mimeType
    this.state = 'inactive'
    this.number = FakeRecorder.instances.length
    FakeRecorder.instances.push(this)
  }

  start() {
    this.state = 'recording'
    queueMicrotask(() => this.onstart?.())
  }

  stop() {
    if (this.state === 'inactive') throw new Error('InvalidStateError')
    this.state = 'inactive'
    queueMicrotask(() => {
      this.ondataavailable?.({ data: new Blob([`audio ${this.number}`]) })
      this.onstop?.()
    })
  }
}

/** A clock and timers that only move when the test says. */
export function fakeClock(start = 1_800_000_000_000) {
  const clock = { time: start, timers: [] }
  clock.now = () => clock.time
  clock.schedule = (fn, ms) => {
    const timer = { at: clock.time + ms, fn }
    clock.timers.push(timer)
    return () => clock.timers.splice(0, clock.timers.length, ...clock.timers.filter((t) => t !== timer))
  }
  clock.advance = async (ms) => {
    const until = clock.time + ms
    for (;;) {
      clock.timers.sort((a, b) => a.at - b.at)
      const next = clock.timers[0]
      if (!next || next.at > until) break
      clock.timers.shift()
      clock.time = next.at
      next.fn()
      await Promise.resolve()
      await Promise.resolve()
    }
    clock.time = until
    await Promise.resolve()
    await Promise.resolve()
  }
  return clock
}

/** A fetch that answers from a script, one entry per call: [status, body] or an Error to throw. */
export function scriptedFetch(script, fallback = [200, {}]) {
  const calls = []
  const fetch = async (url, init) => {
    calls.push({ url: String(url), init })
    const next = script.length ? script.shift() : fallback
    if (next instanceof Error) throw next
    return new Response(JSON.stringify(next[1]), { status: next[0] })
  }
  return { calls, fetch }
}
