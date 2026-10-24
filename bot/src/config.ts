// Settings, from the environment. Secrets may come from a JSON secrets file read at start
// (docs/spec.md, 3.6), never from the repository.

import { readFileSync } from 'node:fs'

export interface Config {
  discordToken: string
  serverUrl: string
  /** Register slash commands in this guild only (instant), instead of globally. */
  guildId?: string
  /** Echo transcript lines into the voice channel's chat. Off by default (docs/spec.md, 3.2). */
  echo: boolean
  /** Where people can read the transcript, mentioned in the stop message. */
  publicUrl?: string
  debugVoice: boolean
}

function secret(name: string, secrets: Record<string, string>): string | undefined {
  return process.env[name] || secrets[name]
}

export function loadConfig(env = process.env): Config {
  const secrets: Record<string, string> = env.SCRIBE_SECRETS_FILE
    ? JSON.parse(readFileSync(env.SCRIBE_SECRETS_FILE, 'utf8'))
    : {}
  const tokenKey = env.SCRIBE_DISCORD_TOKEN_KEY || 'SCRIBE_DISCORD_BOT_TOKEN'
  const discordToken = env.DISCORD_TOKEN || secret(tokenKey, secrets)
  if (!discordToken) throw new Error(`no Discord token: set DISCORD_TOKEN or ${tokenKey} in SCRIBE_SECRETS_FILE`)
  const flag = (value?: string) => ['1', 'true', 'yes', 'on'].includes((value ?? '').toLowerCase())
  return {
    discordToken,
    serverUrl: env.SCRIBE_SERVER_URL || 'http://127.0.0.1:8787',
    guildId: env.SCRIBE_GUILD_ID || undefined,
    echo: flag(env.SCRIBE_ECHO),
    publicUrl: env.SCRIBE_PUBLIC_URL || undefined,
    debugVoice: flag(env.SCRIBE_DEBUG_VOICE),
  }
}
