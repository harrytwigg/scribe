import { load } from './settings.js'

const $ = (selector) => document.querySelector(selector)

function render(status = { state: 'idle' }) {
  const busy = ['starting', 'recording', 'stopping'].includes(status.state)
  $('#state').textContent = status.state === 'recording' ? '● recording' : status.state
  $('#state').dataset.state = status.state
  $('#title').textContent = busy ? status.title ?? '' : ''
  const uploading = status.pendingChunks ? `${status.recording === false ? '' : 'recording: '}${status.pendingChunks} part${status.pendingChunks === 1 ? '' : 's'} still uploading` : ''
  $('#detail').textContent = busy
    ? `${status.lines ?? 0} lines${status.failed ? `, ${status.failed} undelivered` : ''}${uploading ? ` · ${uploading}` : ''}${status.last ? ` · ${status.last}` : ''}`
    : [status.reason ? `Stopped: ${status.reason}.` : '', uploading ? `Still uploading the call's ${uploading}.` : ''].filter(Boolean).join(' ')
  const problem = status.error ?? status.recordingError ?? (busy ? status.warning : null)
  $('#error').hidden = !problem
  $('#error').textContent = problem ?? ''
  $('#start').hidden = busy
  $('#stop').hidden = !busy
  $('#stop').disabled = status.state === 'stopping'
}

async function send(message) {
  const result = await chrome.runtime.sendMessage(message)
  if (!result?.ok) render({ state: 'idle', error: result?.error ?? 'no response' })
}

$('#start').addEventListener('click', async () => {
  $('#start').disabled = true
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    // Asked for here, in the click: tab capture needs the user to have invoked the extension.
    const tabStreamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id })
    await send({ type: 'start', tabStreamId, tabId: tab.id, title: tab.title || new URL(tab.url).hostname })
  } catch (error) {
    render({ state: 'idle', error: error.message })
  } finally {
    $('#start').disabled = false
  }
})
$('#stop').addEventListener('click', () => send({ type: 'stop' }))
$('#options').addEventListener('click', () => chrome.runtime.openOptionsPage())
$('#open').addEventListener('click', async () => chrome.tabs.create({ url: (await load()).server }))

chrome.storage.session.get('status').then(({ status }) => render(status))
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'session' && changes.status) render(changes.status.newValue)
})
