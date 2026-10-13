// The offscreen document holds the call: an MV3 service worker cannot keep a MediaStream, and is
// stopped when idle. Only chrome.runtime is available here, so the background passes everything in.

import { Call } from './call.js'

let call = null

function report(status) {
  chrome.runtime.sendMessage({ type: 'status', status }).catch(() => {})
}

/** Opens the captured tab and, if allowed, the mic. */
async function streams(tabStreamId, useMic) {
  const tab = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: tabStreamId } },
    video: false,
  })
  let mic = null
  let warning
  if (useMic) {
    try {
      // Echo cancellation keeps the call's own playback out of "me" when not on headphones.
      mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
    } catch (error) {
      warning = `mic unavailable (${error.name}): allow it on the options page; transcribing the tab only`
    }
  }
  return { tab, mic, warning }
}

/** Starts a call from the given streams; the e2e test calls this directly with its own. */
export async function start(config, title, media) {
  if (call && !call.stopping) throw new Error('already transcribing')
  call = new Call(config, (status) => report({ ...status, warning: media.warning, server: config.server, title }))
  try {
    await call.start(title, { tab: media.tab, mic: media.mic })
  } catch (error) {
    report({ state: 'stopped', error: error.message })
    throw error
  }
}

export function stop() {
  return call?.stop()
}

globalThis.scribe = { start, stop }

chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message.target !== 'offscreen') return
  if (message.type === 'start') {
    streams(message.tabStreamId, message.config.useMic)
      .then((media) =>
        start(message.config, message.title, media).catch((error) => {
          for (const stream of [media.tab, media.mic]) stream?.getTracks().forEach((track) => track.stop())
          throw error
        }),
      )
      .then(() => respond({ ok: true }), (error) => respond({ ok: false, error: error.message }))
    return true
  }
  if (message.type === 'stop') {
    Promise.resolve(stop()).then(() => respond({ ok: true }))
    return true
  }
})
