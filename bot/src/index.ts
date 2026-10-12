import { loadConfig } from './config.js'
import { ScribeBot } from './discord.js'

const bot = new ScribeBot(loadConfig())
for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.once(signal, async () => {
    await bot.shutdown()
    process.exit(0)
  })
}
await bot.start()
