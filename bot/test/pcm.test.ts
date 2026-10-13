import { describe, expect, it } from 'vitest'

import { downmix, durationMs, wav } from '../src/pcm.js'

describe('pcm', () => {
  it('downmixes stereo by averaging', () => {
    const stereo = Buffer.alloc(8)
    stereo.writeInt16LE(1000, 0)
    stereo.writeInt16LE(3000, 2)
    stereo.writeInt16LE(-2000, 4)
    stereo.writeInt16LE(-4000, 6)
    const mono = downmix(stereo)
    expect([mono.readInt16LE(0), mono.readInt16LE(2)]).toEqual([2000, -3000])
  })

  it('writes a valid mono 48 kHz WAV header', () => {
    const file = wav(Buffer.alloc(9600))
    expect(file.toString('ascii', 0, 4)).toBe('RIFF')
    expect(file.readUInt16LE(22)).toBe(1)
    expect(file.readUInt32LE(24)).toBe(48_000)
    expect(file.readUInt32LE(40)).toBe(9600)
    expect(durationMs(Buffer.alloc(9600))).toBe(100)
  })
})
