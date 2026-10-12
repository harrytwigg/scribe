import { describe, expect, it, vi } from 'vitest'

import { ScribeBot } from '../src/discord.js'

const config = { discordToken: 'x', serverUrl: 'http://127.0.0.1:1', echo: false, debugVoice: false }

function interaction(sub: string, userId = 'u1') {
  const replies: string[] = []
  return {
    replies,
    options: { getSubcommand: () => sub },
    inCachedGuild: () => true,
    guildId: 'g1',
    user: { id: userId },
    member: { voice: { channel: { id: 'vc' }, channelId: 'vc' }, displayName: 'Ada' },
    reply: async ({ content }: { content: string }) => void replies.push(content),
  }
}

describe('ScribeBot', () => {
  it('starts one call when two people run /scribe join at once', async () => {
    const bot = new ScribeBot(config as never, {} as never)
    let release!: () => void
    const startCall = vi.fn(() => new Promise<void>((resolve) => (release = resolve)))
    ;(bot as unknown as { startCall: typeof startCall }).startCall = startCall
    const onCommand = (bot as unknown as { onCommand: (i: unknown) => Promise<void> }).onCommand.bind(bot)
    const a = interaction('join', 'u1')
    const b = interaction('join', 'u2')
    const first = onCommand(a)
    await onCommand(b)
    expect(b.replies).toEqual(['Already starting.'])
    release()
    await first
    expect(startCall).toHaveBeenCalledTimes(1)
    // The slot is released afterwards, so a later join is not refused.
    const c = interaction('join', 'u3')
    const third = onCommand(c)
    release()
    await third
    expect(startCall).toHaveBeenCalledTimes(2)
    await bot.client.destroy()
  })

  it('applies an opt-out locally even when the server cannot save it, and keeps it on refresh', async () => {
    const server = {
      setOptout: vi.fn().mockRejectedValue(new Error('down')),
      optouts: vi.fn().mockResolvedValue(new Set<string>()),
    }
    const bot = new ScribeBot(config as never, server as never)
    const onCommand = (bot as unknown as { onCommand: (i: unknown) => Promise<void> }).onCommand.bind(bot)
    const i = interaction('optout', 'u9')
    await onCommand(i)
    expect(i.replies[0]).toMatch(/^You will not be transcribed\. \(Only until the bot restarts/)
    const optedOut = () => (bot as unknown as { optedOut: Set<string> }).optedOut
    expect(optedOut().has('u9')).toBe(true)
    await bot.refreshOptouts() // server still lists nobody and still cannot save
    expect(optedOut().has('u9')).toBe(true)
    server.setOptout.mockResolvedValue({})
    server.optouts.mockResolvedValue(new Set(['u9']))
    await bot.refreshOptouts()
    expect(server.setOptout).toHaveBeenLastCalledWith('u9', true)
    expect(optedOut().has('u9')).toBe(true)
    await bot.client.destroy()
  })
})
