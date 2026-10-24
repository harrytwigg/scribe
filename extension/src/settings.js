// Settings live in chrome.storage.local.

export const DEFAULTS = {
  server: 'http://harry-box:8787',
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
