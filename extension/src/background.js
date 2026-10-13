// Starts and stops calls, and keeps the latest status where the popup can read it.

import { load, originPattern } from './settings.js'

const OFFSCREEN = 'src/offscreen.html'

async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })
  if (contexts.length) return
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

async function start({ tabStreamId, title }) {
  const config = await load()
  if (!config.token) throw new Error('set the server and token on the options page first')
  if (!(await chrome.permissions.contains({ origins: [originPattern(config.server)] }))) {
    throw new Error(`no permission to reach ${config.server}: save it on the options page`)
  }
  await ensureOffscreen()
  await setStatus({ state: 'starting', title })
  const result = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'start', tabStreamId, title, config })
  if (!result?.ok) throw new Error(result?.error ?? 'capture failed to start')
}

async function stop() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })
  // The document closes itself once the call has ended, which can cut this reply off: that is fine.
  if (contexts.length) await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' }).catch(() => {})
}

chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (message.target === 'offscreen') return
  if (message.type === 'status') {
    setStatus(message.status).then(async () => {
      if (message.status.state === 'stopped') await chrome.offscreen.closeDocument().catch(() => {})
    })
    return
  }
  const action = { start, stop }[message.type]
  if (!action) return
  action(message).then(
    () => respond({ ok: true }),
    async (error) => {
      const { status } = await chrome.storage.session.get('status')
      if (message.type === 'start' && status?.state === 'starting') await setStatus({ state: 'stopped', error: error.message })
      respond({ ok: false, error: error.message })
    },
  )
  return true
})
