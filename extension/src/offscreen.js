// The offscreen document holds the call: an MV3 service worker cannot keep a MediaStream, and is
// stopped when idle. Only chrome.runtime is available here, so the background passes everything in.

import { Call } from './call.js'
import { ScribeClient } from './client.js'
import { ChunkUploader, idbChunkStore, memoryChunkStore } from './uploader.js'

let call = null
let lastStatus = null

// The recording's segments outlive any one call: they are delivered from here until the server has
// them, and this document is kept open (background.js) while any are left.
const uploader = new ChunkUploader({
  clientFor: (server) => new ScribeClient(server),
  store: typeof indexedDB === 'undefined' ? memoryChunkStore() : idbChunkStore(),
})

function report(status) {
  lastStatus = status
  chrome.runtime.sendMessage({ type: 'status', status: { ...status, pendingChunks: uploader.pending } }).catch(() => {})
}

// Progress of a recording still being delivered shows in the popup, and the end of it lets the
// background close this document.
uploader.onChange = () => {
  if (call && !call.done) call.publish()
  else if (lastStatus?.state === 'stopped') report(lastStatus)
}
uploader.onIdle = () => {
  if (lastStatus?.state === 'stopped') report(lastStatus)
}
uploader.resume().catch(() => {}) // from a call whose document closed, or a browser that quit, before delivering

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
export async function start(config, title, media, options) {
  if (call && !call.done) throw new Error(call.stopping ? 'still finishing the last call: try again in a moment' : 'already transcribing')
  call = new Call(config, (status) => report({ ...status, warning: media.warning, server: config.server, title }), { uploader, ...options })
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

globalThis.scribe = { start, stop, uploader }

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
  if (message.type === 'idle') {
    respond({ idle: !call || call.done, uploading: uploader.pending })
    return
  }
  if (message.type === 'stop') {
    Promise.resolve(stop()).then(() => respond({ ok: true }))
    return true
  }
})
