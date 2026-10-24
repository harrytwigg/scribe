// The Discord side: slash commands, joining voice, consent notices (docs/spec.md, 3.2).

import {
  EndBehaviorType,
  type VoiceConnection,
  VoiceConnectionStatus,
  entersState,
  joinVoiceChannel,
} from '@discordjs/voice'
import {
  ChannelType,
  type ChatInputCommandInteraction,
  Client,
  Events,
  GatewayIntentBits,
  type Guild,
  MessageFlags,
  SlashCommandBuilder,
  type VoiceBasedChannel,
  type VoiceState,
} from 'discord.js'

import { ScribeClient, type Speaker } from './client.js'
import type { Config } from './config.js'
import { Pipeline } from './pipeline.js'
import type { VoiceSource } from './voice-source.js'

const NICKNAME = '🔴 transcribing'

export class DiscordVoiceSource implements VoiceSource {
  constructor(private readonly connection: VoiceConnection) {}

  onSpeaking(listener: (userId: string) => void): void {
    this.connection.receiver.speaking.on('start', listener)
  }

  subscribe(userId: string, silenceMs: number) {
    return this.connection.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: silenceMs },
    })
  }
}

interface Call {
  guild: Guild
  channel: VoiceBasedChannel
  connection: VoiceConnection
  pipeline: Pipeline
  sessionId: string
  previousNickname: string | null
  stopping?: Promise<void>
}

export const commands = new SlashCommandBuilder()
  .setName('scribe')
  .setDescription('Transcribe this voice channel')
  .addSubcommand((s) => s.setName('join').setDescription('Start transcribing the voice channel you are in'))
  .addSubcommand((s) => s.setName('stop').setDescription('Stop transcribing (anyone in the call can)'))
  .addSubcommand((s) => s.setName('optout').setDescription('Never transcribe me'))
  .addSubcommand((s) => s.setName('optin').setDescription('Allow transcribing me again'))
  .addSubcommand((s) => s.setName('status').setDescription('Show the active model and whether this call is being transcribed'))

export class ScribeBot {
  readonly client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] })
  private readonly calls = new Map<string, Call>()
  /** Guilds with a join in progress: claimed before the first await, so two joins cannot race. */
  private readonly joining = new Set<string>()
  private optedOut = new Set<string>()
  /** Opt-outs/ins the server has not saved yet; re-sent and re-applied on every refresh. */
  private readonly unsaved = new Map<string, boolean>()

  constructor(
    private readonly config: Config,
    private readonly server = new ScribeClient(config.serverUrl),
  ) {
    this.client.once(Events.ClientReady, (ready) => this.onReady(ready.user.tag))
    this.client.on(Events.InteractionCreate, (interaction) => {
      if (interaction.isChatInputCommand() && interaction.commandName === 'scribe') {
        this.onCommand(interaction).catch((error) => this.log('command failed', error))
      }
    })
    this.client.on(Events.VoiceStateUpdate, (before, after) => this.onVoiceState(before, after))
  }

  private log(...args: unknown[]) {
    console.log(new Date().toISOString(), ...args)
  }

  async start(): Promise<void> {
    await this.client.login(this.config.discordToken)
  }

  private async onReady(tag: string) {
    this.log(`logged in as ${tag}`)
    await this.refreshOptouts().catch((error) => this.log('could not load opt-outs from the server', error))
    await this.endStaleSessions().catch((error) => this.log('could not close stale sessions', error))
    const body = [commands.toJSON()]
    if (this.config.guildId) await this.client.application!.commands.set(body, this.config.guildId)
    else await this.client.application!.commands.set(body)
    this.log(`slash commands registered ${this.config.guildId ? `in guild ${this.config.guildId}` : 'globally'}`)
  }

  private async onCommand(interaction: ChatInputCommandInteraction) {
    const sub = interaction.options.getSubcommand()
    if (!interaction.inCachedGuild()) {
      await interaction.reply({ content: 'Use this in a server.', flags: MessageFlags.Ephemeral })
      return
    }
    const userId = interaction.user.id
    if (sub === 'optout' || sub === 'optin') {
      const out = sub === 'optout'
      // Locally first, so nothing more is posted even if the server cannot be reached.
      if (out) {
        this.optedOut.add(userId)
        for (const call of this.calls.values()) call.pipeline.drop(userId)
      } else this.optedOut.delete(userId)
      let saved = true
      this.unsaved.set(userId, out)
      await this.server
        .setOptout(userId, out)
        .then(() => this.unsaved.delete(userId))
        .catch((error) => {
          saved = false
          this.log('could not save opt-out', error)
        })
      const done = out ? 'You will not be transcribed.' : 'You can be transcribed again.'
      await interaction.reply({
        content: saved ? done : `${done} (Only until the bot restarts: the scribe server could not save it. Try again later.)`,
        flags: MessageFlags.Ephemeral,
      })
      return
    }
    if (sub === 'status') {
      const status = await this.server.status().catch(() => null)
      const call = this.calls.get(interaction.guildId)
      await interaction.reply({
        content: [
          status ? `Model: ${status.active_model ?? 'none active'}` : 'The scribe server is unreachable.',
          call ? `Transcribing ${call.channel}.` : 'Not transcribing in this server.',
          ...(call ? this.whoIsHeard(call) : []),
        ].join('\n'),
        flags: MessageFlags.Ephemeral,
      })
      return
    }
    if (sub === 'stop') {
      const call = this.calls.get(interaction.guildId)
      if (!call) {
        await interaction.reply({ content: 'Not transcribing.', flags: MessageFlags.Ephemeral })
        return
      }
      const inCall = interaction.member.voice.channelId === call.channel.id
      if (!inCall && !interaction.memberPermissions.has('ManageGuild')) {
        await interaction.reply({ content: 'Only people in the call can stop it.', flags: MessageFlags.Ephemeral })
        return
      }
      await interaction.deferReply()
      await this.stop(interaction.guildId, `stopped by ${interaction.member.displayName}`)
      await interaction.editReply('Transcription stopped.')
      return
    }
    // join
    const channel = interaction.member.voice.channel
    if (!channel) {
      await interaction.reply({ content: 'Join a voice channel first.', flags: MessageFlags.Ephemeral })
      return
    }
    const guildId = interaction.guildId
    if (this.calls.has(guildId) || this.joining.has(guildId)) {
      const where = this.calls.get(guildId)?.channel
      await interaction.reply({ content: where ? `Already transcribing ${where}.` : 'Already starting.', flags: MessageFlags.Ephemeral })
      return
    }
    this.joining.add(guildId)
    try {
      await this.startCall(interaction, channel)
    } finally {
      this.joining.delete(guildId)
    }
  }

  private async startCall(interaction: ChatInputCommandInteraction<'cached'>, channel: VoiceBasedChannel) {
    await interaction.deferReply()
    const status = await this.server.status().catch(() => null)
    if (!status) {
      await interaction.editReply('The scribe server is unreachable, so nothing would be transcribed.')
      return
    }
    if (!status.active_model) {
      await interaction.editReply('No model is active on the scribe server. Pick one in the GUI first.')
      return
    }
    await this.endStaleSessions().catch((error) => this.log('could not close stale sessions', error))
    // Opt-outs live on the server; never start a call without the current list.
    try {
      await this.refreshOptouts()
    } catch {
      await interaction.editReply('Could not load opt-outs from the scribe server, so not starting.')
      return
    }
    try {
      await this.join(channel)
    } catch (error) {
      this.log('join failed', error)
      await interaction.editReply(`Could not start: ${error instanceof Error ? error.message : error}`)
      return
    }
    await interaction.editReply(`Transcribing ${channel} with ${status.active_model}.`)
  }

  /**
   * End Discord sessions the server still has open but this bot is not running: left behind by a
   * crash, a power cut, or a server that was down when a call stopped. Otherwise they count as
   * open forever, and the host's idle check would never let it power off.
   */
  async endStaleSessions(): Promise<void> {
    const sessions = await this.server.listSessions()
    // Read after the await: a guild that finished joining meanwhile is running, and one still
    // joining may have just created its session, so neither is stale.
    const running = new Set([...this.calls.values()].map((call) => call.sessionId))
    for (const session of sessions) {
      const joining = session.guild != null && this.joining.has(session.guild)
      if (session.source === 'discord' && session.ended_at == null && !running.has(session.id) && !joining) {
        await this.server.endSession(session.id)
        this.log(`ended stale session ${session.id}`)
      }
    }
  }

  /** Load opt-outs from the server, keeping (and re-sending) any it has not saved yet. */
  async refreshOptouts(): Promise<void> {
    for (const [userId, out] of this.unsaved) {
      await this.server.setOptout(userId, out).then(() => this.unsaved.delete(userId)).catch(() => {})
    }
    const fresh = await this.server.optouts()
    for (const [userId, out] of this.unsaved) {
      if (out) fresh.add(userId)
      else fresh.delete(userId)
    }
    this.optedOut = fresh
  }

  private async join(channel: VoiceBasedChannel) {
    const guild = channel.guild
    const connection = joinVoiceChannel({
      guildId: guild.id,
      channelId: channel.id,
      adapterCreator: guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: true,
      debug: this.config.debugVoice,
    })
    if (this.config.debugVoice) connection.on('debug', (message) => this.log('[voice]', message))
    connection.on('error', (error) => this.log('voice connection error', error))
    let session: { id: string }
    try {
      await entersState(connection, VoiceConnectionStatus.Ready, 20_000)
      session = await this.server.createSession(`${guild.name} #${channel.name}`, guild.id, channel.id)
    } catch (error) {
      connection.destroy()
      throw error
    }
    const names = new Map<string, string>()
    const pipeline = new Pipeline({
      source: new DiscordVoiceSource(connection),
      client: this.server,
      sessionId: session.id,
      isOptedOut: (userId) => this.optedOut.has(userId) || userId === this.client.user?.id,
      speaker: async (userId): Promise<Speaker> => {
        if (!names.has(userId)) {
          const member = await guild.members.fetch(userId).catch(() => null)
          names.set(userId, member?.displayName ?? userId)
        }
        return { id: userId, name: names.get(userId)! }
      },
      onLine: (speaker, text) => {
        if (this.config.echo && channel.type === ChannelType.GuildVoice) channel.send(`**${speaker.name}:** ${text}`).catch(() => {})
      },
      onError: (error) => this.log('pipeline', error instanceof Error ? error.message : error),
    })
    const me = guild.members.me
    const call: Call = { guild, channel, connection, pipeline, sessionId: session.id, previousNickname: me?.nickname ?? null }
    this.calls.set(guild.id, call)
    await me?.setNickname(NICKNAME).catch(() => this.log('cannot change nickname (needs Change Nickname)'))
    await this.notice(channel, '🔴 This call is being transcribed by scribe. `/scribe optout` to exclude yourself, `/scribe stop` to end it.')
    connection.on(VoiceConnectionStatus.Disconnected, async () => {
      try {
        await Promise.race([
          entersState(connection, VoiceConnectionStatus.Signalling, 5_000),
          entersState(connection, VoiceConnectionStatus.Connecting, 5_000),
        ])
      } catch {
        await this.stop(guild.id, 'disconnected')
      }
    })
    this.log(`transcribing ${guild.name} #${channel.name} as session ${session.id}`)
  }

  private async notice(channel: VoiceBasedChannel, text: string) {
    if ('send' in channel) await channel.send(text).catch(() => this.log('cannot post in the voice channel chat'))
  }

  private onVoiceState(before: VoiceState, after: VoiceState) {
    const call = this.calls.get(after.guild.id)
    if (!call || after.member?.user.bot) return
    if (after.channelId === call.channel.id && before.channelId !== call.channel.id) {
      const text = this.optedOut.has(after.id)
        ? null
        : `🔴 <@${after.id}>, this call is being transcribed. \`/scribe optout\` to exclude yourself.`
      if (text) this.notice(call.channel, text)
    }
    if (before.channelId === call.channel.id && after.channelId !== call.channel.id) {
      const humans = call.channel.members.filter((m) => !m.user.bot).size
      if (humans === 0) this.stop(call.guild.id, 'everyone left').catch((e) => this.log(e))
    }
  }

  /** One line per person in the call: whether their speech is reaching the transcript. */
  private whoIsHeard(call: Call): string[] {
    return call.channel.members
      .filter((member) => !member.user.bot)
      .map((member) => {
        const heard = call.pipeline.heard.get(member.id)
        if (this.optedOut.has(member.id)) return `- ${member.displayName}: opted out`
        if (!heard?.packets) return `- ${member.displayName}: not heard yet`
        const unusable = heard.corrupt + heard.encrypted
        if (unusable === heard.packets) return `- ${member.displayName}: audio arriving but not decodable`
        const skipped = unusable ? ` (${unusable} packets not decodable)` : ''
        return `- ${member.displayName}: ${heard.lines} line${heard.lines === 1 ? '' : 's'}${skipped}`
      })
  }

  async stop(guildId: string, reason: string): Promise<void> {
    const call = this.calls.get(guildId)
    if (!call) return
    call.stopping ??= (async () => {
      await call.pipeline.stop()
      await this.server.endSession(call.sessionId).catch((e) => this.log('end session', e))
      if (call.connection.state.status !== VoiceConnectionStatus.Destroyed) call.connection.destroy()
      await call.guild.members.me?.setNickname(call.previousNickname).catch(() => {})
      const where = this.config.publicUrl ? ` Transcript: ${this.config.publicUrl}` : ''
      await this.notice(call.channel, `⏹️ Transcription stopped (${reason}).${where}`)
      this.calls.delete(guildId)
      this.log(`stopped session ${call.sessionId}: ${reason}`)
      for (const [userId, heard] of call.pipeline.heard) {
        const name = call.guild.members.cache.get(userId)?.displayName ?? userId
        this.log(
          `  heard ${name}: ${heard.lines} lines, ${heard.packets} packets, ${heard.encrypted} still encrypted, ${heard.corrupt} not Opus`,
        )
      }
    })()
    await call.stopping
  }

  async shutdown(): Promise<void> {
    await Promise.allSettled([...this.calls.keys()].map((id) => this.stop(id, 'bot shutting down')))
    await this.client.destroy()
  }
}
