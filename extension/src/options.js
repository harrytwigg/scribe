import { ScribeClient } from './client.js'
import { DEFAULTS, load, originPattern } from './settings.js'

const form = document.querySelector('#form')
const result = document.querySelector('#result')

async function fill() {
  const settings = await load()
  for (const [key, value] of Object.entries(settings)) {
    const input = form.elements[key]
    if (input.type === 'checkbox') input.checked = value
    else input.value = value
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault()
  const settings = {}
  for (const key of Object.keys(DEFAULTS)) {
    const input = form.elements[key]
    settings[key] = input.type === 'checkbox' ? input.checked : input.value.trim()
  }
  settings.server = settings.server.replace(/\/+$/, '')
  // Asked in the click: a server outside the built-in tailnet patterns needs the user's grant.
  const origins = [originPattern(settings.server)]
  if (!(await chrome.permissions.contains({ origins })) && !(await chrome.permissions.request({ origins }))) {
    result.textContent = `Not saved: scribe needs permission to reach ${settings.server}.`
    return
  }
  await chrome.storage.local.set(settings)
  await chrome.storage.local.remove('token') // from before the server needed none
  try {
    const status = await new ScribeClient(settings.server).status()
    result.textContent = `Saved. Connected: active model ${status.active_model ?? 'none (pick one in the GUI)'}.`
  } catch (error) {
    result.textContent = `Saved, but the server did not answer: ${error.message}`
  }
})

async function micState() {
  const permission = await navigator.permissions.query({ name: 'microphone' })
  document.querySelector('#mic-state').textContent = permission.state
}

document.querySelector('#mic').addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    stream.getTracks().forEach((track) => track.stop())
  } catch {}
  await micState()
})

fill()
micState()
