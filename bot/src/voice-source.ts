// Where per-user Opus packets come from. Discord in production; a replay of WAV files in tests.
// Everything downstream of `subscribe` is shared (docs/spec.md, 3.2).

import type { Readable } from 'node:stream'

export interface VoiceSource {
  /** Called whenever a user starts speaking. */
  onSpeaking(listener: (userId: string) => void): void
  /**
   * The user's Opus packets (one 20 ms 48 kHz stereo frame per chunk) until they have been silent
   * for `silenceMs`; then the stream ends.
   */
  subscribe(userId: string, silenceMs: number): Readable
}
