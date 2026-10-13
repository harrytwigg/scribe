import assert from 'node:assert/strict'
import { test } from 'node:test'

import { Call } from '../src/call.js'
import { ServerError } from '../src/client.js'

function call({ heartbeat = async () => ({ ok: true }), heardAgoMs = 0 } = {}) {
  const c = new Call({ server: 'http://box:8787', token: 't', myName: 'Me', othersName: 'Others' }, () => {}, { idleStopMs: 600_000 })
  c.sessionId = 's1'
  c.heard = Date.now() - heardAgoMs
  c.client.heartbeat = heartbeat
  c.stopped = 0
  c.stop = () => {
    c.stopped++
    c.stopping = Promise.resolve()
  }
  return c
}

test('a live call sends a heartbeat and carries on', async () => {
  let beats = 0
  const c = call({ heartbeat: async () => beats++ })
  await c.tick()
  assert.equal(beats, 1)
  assert.equal(c.stopped, 0)
})

test('ten minutes without a transcribed line ends the call', async () => {
  const c = call({ heardAgoMs: 600_001 })
  await c.tick()
  assert.equal(c.stopped, 1)
  assert.equal(c.reason, 'no speech for 10 min')
})

test('a session the server ended (reaped while asleep) ends the call', async () => {
  const c = call({ heartbeat: async () => { throw new ServerError(409, 'session has ended') } })
  await c.tick()
  assert.equal(c.stopped, 1)
  assert.equal(c.reason, 'the server ended the session')
})

test('an unreachable server does not end the call', async () => {
  const c = call({ heartbeat: async () => { throw new TypeError('fetch failed') } })
  await c.tick()
  assert.equal(c.stopped, 0)
})

test('a failed session create leaves the call done, so the next start is allowed', async () => {
  const c = new Call({ server: 'http://box:8787', token: 't' }, () => {})
  c.client.createSession = async () => { throw new ServerError(401, 'sign in') }
  await assert.rejects(c.start('x', { tab: null, mic: null }), /sign in/)
  assert.equal(c.done, true)
})
