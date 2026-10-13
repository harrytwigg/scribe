// Starts and stops calls, and keeps the latest status where the popup can read it.

import { load, originPattern } from './settings.js'

const OFFSCREEN = 'src/offscreen.html'

// Start, stop and closing the document run one at a time, so a start never lands in a document
// that the previous call's end is about to close.
let queue = Promise.resolve()
const serial = (task) => (queue = queue.then(task, task))

async function offscreenOpen() {
  return (await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })).length > 0
}

async function ensureOffscreen() {
  if (await offscreenOpen()) return
  await chrome.offscreen.createDocument({
    url: OFFSCREEN,
    reasons: ['USER_MEDIA'],
    justification: 'Capture the call tab and microphone for transcription',
  })
}

async function setStatus(status) {
  await chrome.storage.session.set({ status })
  const recording = status.state === 'recording' || status.state === 'stopping'
  await chrome.action.setBadgeText({ text: recording ? 'REC' : '' })
  await chrome.action.setBadgeBackgroundColor({ color: '#d93025' })
}

async function start({ tabStreamId, title, tabId }) {
  const config = await load()
  if (!config.token) throw new Error('set the server and token on the options page first')
  if (!(await chrome.permissions.contains({ origins: [originPattern(config.server)] }))) {
    throw new Error(`no permission to reach ${config.server}: save it on the options page`)
  }
  await ensureOffscreen()
  await setStatus({ state: 'starting', title })
  const result = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'start', tabStreamId, title, config })
  if (!result?.ok) throw new Error(result?.error ?? 'capture failed to start')
  await chrome.storage.session.set({ capturedTab: tabId })
}

async function stop() {
  if (!(await offscreenOpen())) {
    // No document, so no call: a status still saying otherwise is left from one that died.
    await setStatus({ state: 'stopped' })
    return
  }
  // The document closes itself once the call has ended, which can cut this reply off: that is fine.
  await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' }).catch(() => {})
}

async function onStatus(status) {
  if (!(await offscreenOpen())) return setStatus(status)
  // A start queued behind this may already have begun a new call in the document: then this
  // status is stale, and the document stays.
  const reply = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'idle' }).catch(() => ({ idle: true }))
  if (!reply?.idle) return
  await setStatus(status)
  await chrome.storage.session.remove('capturedTab')
  await chrome.offscreen.closeDocument().catch(() => {})
}

chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message.target === 'offscreen') return
  if (message.type === 'status') {
    // Only the end of a call touches the document, so only that waits its turn.
    if (message.status.state === 'stopped') serial(() => onStatus(message.status))
    else setStatus(message.status)
    return
  }
  const action = { start, stop }[message.type]
  if (!action) return
  serial(() => action(message)).then(
    () => respond({ ok: true }),
    async (error) => {
      // A failed start leaves no call: record why, and close the document if nothing else uses it.
      if (message.type === 'start') await serial(() => onStatus({ state: 'stopped', error: error.message }))
      respond({ ok: false, error: error.message })
    },
  )
  return true
})

// Capture survives navigation within the tab, so a new page loading there (leaving the call,
// reloading, going elsewhere) ends the call. Same-page URL changes do not.
chrome.tabs.onUpdated.addListener(async (tabId, change) => {
  if (change.status !== 'loading') return
  const { capturedTab } = await chrome.storage.session.get('capturedTab')
  if (tabId === capturedTab) serial(stop)
})
