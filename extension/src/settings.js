// Settings live in chrome.storage.local (the token too: it never leaves this browser except to the
// configured server).

export const DEFAULTS = {
  server: 'http://harry-box:8787',
  token: '',
  myName: 'Me',
  othersName: 'Others',
  useMic: true,
  separateSpeakers: true,
}

export async function load() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) }
}

/** The host permission pattern for a server URL (any port matches). */
export function originPattern(server) {
  const url = new URL(server)
  return `${url.protocol}//${url.hostname}/*`
}
