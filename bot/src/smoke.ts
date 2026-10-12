// Live check of the Discord voice layer, without recording anyone: log in, join an EMPTY voice
// channel undeafened (so DAVE is negotiated as for a real call), wait for Ready, leave.
//
//   node dist/src/smoke.js            (same environment as the bot)
//
// It proves login, gateway intents and the DAVE voice handshake on this stack. It cannot prove
// receive: that needs a person speaking (docs/spec.md, 6).

import { VoiceConnectionStatus, entersState, joinVoiceChannel } from '@discordjs/voice'
import { ChannelType, Client, Events, GatewayIntentBits } from 'discord.js'

import { loadConfig } from './config.js'

const config = loadConfig()
const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] })
const log = (...args: unknown[]) => console.log(new Date().toISOString(), ...args)
// Voice debug lines carry the session's keys and tokens; keep them out of logs.
const redact = (m: string) =>
  m
    .replace(/"(secret_key|secretKey)":(\[[^\]]*\]|\{[^}]*\})/g, '"$1":"[redacted]"')
    .replace(/"(token|session_id|sessionId)":"[^"]*"/g, '"$1":"[redacted]"')

client.once(Events.ClientReady, async (ready) => {
  log(`logged in as ${ready.user.tag}`)
  try {
    const guild = config.guildId ? await client.guilds.fetch(config.guildId) : client.guilds.cache.first()
    if (!guild) throw new Error('the bot is in no guild')
    const channels = await guild.channels.fetch()
    const me = await guild.members.fetchMe()
    const empty = channels.find(
      (c) =>
        c?.type === ChannelType.GuildVoice &&
        c.members.size === 0 &&
        c.permissionsFor(me).has(['ViewChannel', 'Connect']),
    )
    if (!empty || empty.type !== ChannelType.GuildVoice) throw new Error('no empty voice channel the bot may join')
    log(`joining empty voice channel #${empty.name} in ${guild.name}`)
    const connection = joinVoiceChannel({
      guildId: guild.id,
      channelId: empty.id,
      adapterCreator: guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: true,
      debug: true,
    })
    connection.on('debug', (m) => {
      if (/dave|mls|protocol|close|ready|transition/i.test(m) && !m.includes('state change')) log('[voice]', redact(m))
    })
    connection.on('stateChange', (a, b) => log(`[state] ${a.status} -> ${b.status}`))
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000)
    log('voice connection READY')
    await new Promise((resolve) => setTimeout(resolve, 8000))
    connection.destroy()
    log('left; smoke test passed')
    await client.destroy()
    process.exit(0)
  } catch (error) {
    log('smoke test FAILED', error)
    await client.destroy()
    process.exit(1)
  }
})

await client.login(config.discordToken)
