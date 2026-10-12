# Discord voice-receive + DAVE: stack research (as of 2026-10-12)

Method: primary sources (registries, GitHub repos/issues/changelogs, Discord docs). Unauthenticated web search was weak, so most facts below come from package registries and repo trackers. Items marked [UNCERTAIN] are single-source or inferred.

## 1. DAVE enforcement
- Discord docs (docs.discord.com/developers/topics/voice-connections): "we will only support E2EE calls starting on March 1st, 2026" (DMs, group DMs, voice channels, Go Live). Clients declare `max_dave_protocol_version` in Opcode 0 Identify; omitting it or sending 0 means "no DAVE support".
- Discord blog (discord.com/blog/every-voice-and-video-call-on-discord-is-now-end-to-end-encrypted): migration completed early March 2026. Every voice/video call is E2EE except Stage channels. Discord required all clients, bots/apps included, to support DAVE before joining, and is removing the unencrypted-fallback code.
- A bot library without DAVE cannot join regular voice channels. py-cord issue #3135 reports close code 4017 ("E2EE/DAVE protocol required"); JDA issue #3034 reports "All connections fail with Reason: E2EE/DAVE protocol required". I could not fetch the official 4017 table, so the code's meaning is confirmed only through those two reports.
- Stage channels are exempt (no DAVE), so a non-DAVE library may still work there.
- Receiving is not exempt. Each received RTP payload has a second MLS encryption layer on top of the transport layer (xchacha/aes-gcm). Discord does not downgrade a call because a non-DAVE receiver is present (discord-ext-voice-recv PR #58). The library must decrypt per sender with the MLS session. Skipping that gives `OpusError: corrupted stream` or static.
- DAVE/MLS is implemented by libdave (MIT). Bindings: `davey` (Rust; npm `@snazzah/davey`, PyPI `davey`, crate `davey`) and JDAVE (Java).

## 2. Library matrix (receive + DAVE)

| Library | Versions | Receive under DAVE | Per-user streams | Known problems |
|---|---|---|---|---|
| discord.js voice | `@discordjs/voice` 0.19.2 (npm latest, 2026-03-17); `discord.js` 14.27.0; dep `@snazzah/davey ^0.1.9`, latest 0.1.12 (2026-06-22). `1.0.0-dev` builds exist on the `dev` tag (v15 line). | Yes, designed in. `VoiceReceiver` calls `daveSession.decrypt(payload, userId)`, and 0.19.1 made davey a hard dependency. | Yes. `receiver.subscribe(userId, {end})` returns an Opus stream per user; `receiver.speaking` events; the SSRC-to-user mapping is internal. | See below. |
| discord.py + `discord-ext-voice-recv` | discord.py 2.7.1 (`[voice]` extra pulls `davey>=0.1.0`); voice-recv PyPI 0.5.2a179 (2025-06-18, last release); master reports itself as 0.5.3a180 | No in the released package. Open PRs/issues #54, #56, #58, #62, #65 (latest 2026-10-10) add DAVE decrypt. Issue #61 says upstream does not decrypt, so every frame raises `corrupted stream`. The maintainer has not merged any PR since 2025-06. | Yes (`AudioSink.write(user, VoiceData)`, with SSRC-to-user mapping). | Reports of `corrupted stream` crashing the router thread (#43, #49, #53), secret_key desync after reconnect (#59), listener bug (#63). Needs a patched fork or monkeypatch. |
| py-cord | 2.8.1 (2026-07-25); extra `voice` = `PyNaCl, davey>=0.1.4` | Send-side DAVE only in release (changelog: "DAVE for voice-sending related features", PR #3143). Receive-decrypt PRs #3185, #3201/#3202 and #3386 were closed unmerged. The remaining PR, #3159 ("Voice internals & DAVE support (rec)"), is still open as of 2026-09-05. Sinks and `start_recording` are therefore not DAVE-safe in releases. | Sinks give per-user audio. | A py-cord maintainer wrote in discord.js #11445 that their implementation "is not in a stable working state". Davey issue #15 shows ~95% decrypt failure with an experimental py-cord branch; the davey maintainer says the problem is in the library integration, not davey. |
| nextcord | 3.2.0 (2026-05-21); extra `voice` = `PyNaCl, dave-py<0.2` | No receive feature (I found no sink or recording API). Open issue #1294: opcode 24 (`DAVE_PREPARE_EPOCH`) is never dispatched. Not suitable. | n/a | n/a |
| Rust songbird (+ serenity 0.12.5 / twilight 0.17) | songbird 0.6.1 (2026-10-10, docs-only patch); 0.6.0 (2026-04-05) added DAVE (davey 0.1.2+, `opus2`) | Yes (`receive` feature, `DecodeMode::Decode`). Frames are decrypted in `udp_rx`. | Yes. `CoreEvent::VoiceTick` gives per-SSRC decoded i16 PCM, `SpeakingStateUpdate` maps SSRC to user, and `RtpPacket` gives raw Opus. | Open issue #310 (2026-07-16): per-join silent receive stall after MLS transition, send stall, per-member crypto error, and a "dice roll" success rate. Issue #314 reports an RTP-extension-offset bug in DAVE receive. The PR was closed on 2026-10-04 and does not appear in the `current` log, so it is unmerged [UNCERTAIN how much it matters]. Rust only. |
| JDA (Java) | JDA 6.7.0 (2026-09-20); JDAVE `club.minnced:jdave-api` 0.1.8 plus a native artifact (linux-x86-64/aarch64); requires Java 25 | Yes. `AudioReceiveHandler.handleCombinedAudio` / `handleUserAudio` decrypt via JDAVE. | Yes (`UserAudio`, 20 ms PCM per user). | JDAVE README: "still very new... tested at small scale". JDAVE issue #32 reports a `Decrypt failed with error code FAILURE` WARN on every received packet (open since 2026-05, reproduced by others). Early "breaks with >2 users" bug (JDA #2998) fixed in JDAVE rc.3. |

discord.js specifics:
- Fixed since March: 0.19.2 (2026-03-13/17) strips padding from packets and adds guards (PR #11449). Issue #11419 (reconnect loops, zero audio) and #11445 are closed. A workaround posted in #11419 returns the undecrypted packet on `UnencryptedWhenPassthroughDisabled`.
- Still open: issue #11653 (2026-10-08). If the bot joins after a user is already connected (MLS welcome at transition 0), `lastTransitionId` is `0` and so falsy. Decrypt failures then never recover and the stream is destroyed. The reporter suggests `!== undefined`. Workaround: patch it, or have the bot join first or rejoin.
- `consecutiveFailures` is one counter per session, not per user, so one speaker's failures can affect another's stream (from #11653).
- Davey 0.1.6 PyPI wheel failed receive decrypt (davey issue #20, closed; fixed in 0.1.7, 2026-10-12). npm 0.1.12 is current. Pin and test.

Conflict to flag: `meanwebuser/discord-voice-transcript` docs/DAVE-RESEARCH.md says that on 2026-06-20 discord.js + @discordjs/voice + davey got "zero audio chunks", JDA + libdave "garbage", DPP "zero audio", and voice-recv "partial". Only Craig + Dysnomia worked. This is a single, low-authority repo. The discord.js tracker shows fixes and green reports, and no other independent confirmation of a total failure. I could not run a live test. Treat reliability as unproven until we test on our own server (long sessions, 4+ speakers, joins mid-call, camera on).

## 3. Existing open-source bots
- Scripty (`scripty-bot/scripty`): Rust, serenity+poise+songbird `next` branches, EUPL-1.2 (copyleft; network-use triggers source obligations). Repo ARCHIVED 2025-06-17. Self-hostable in principle, with the STT split into `scripty-bot/stt-service` (also archived, EUPL-1.2). Its Cargo features (`cuda`, `opencl`, `coreml`, `metal`, `openblas`) look like a whisper.cpp / whisper-rs backend [UNCERTAIN, inferred]; it originally used Mozilla DeepSpeech/Coqui STT. Pre-DAVE songbird, so it cannot join voice today. Not a reuse candidate except as design reference. The model is swappable only by editing Rust code.
- Craig (`CraigChat/craig`): ISC, TypeScript, very active (commits 2026-10-12). Uses `@projectdysnomia/dysnomia` (Eris fork, pinned commit) + `@snazzah/davey ^0.1.12` + `@discordjs/opus`. It is a multi-track recorder, not an STT bot, and has no ASR. Self-hostable (Dockerfile, docker-compose, Postgres/Redis, kitchen/cook pipeline) but heavy. Per-user tracks and DAVE work in production on craig.chat (inferred from the repo state; I have not run it).
- Craig-based self-host: `meanwebuser/discord-voice-transcript` (MIT, 2026-07, 1 star): Craig + OpenAI-compatible `/v1/audio/transcriptions` (Whisper server). Too new and unvetted to trust.
- Parley (`SakethKanchi/parley`, ISC, 32 stars, active 2026-10-07): Node + Python, local per-speaker transcription. I did not confirm which voice library it uses or whether it handles DAVE.
- Others (Eavesdrop, discord-transcriber, stenos, DungeonWhispers, ...): all tiny, 0-2 stars, unvetted. No evidence of "SeaVoice" or "discord-speech-recognition" being maintained or DAVE-compatible.
- Verdict: a thin bot is realistic and better than reuse. No maintained, DAVE-compatible, whisper-based bot with a swappable backend and real adoption. Our needs (join, per-user frames, HTTP to ASR) are about 150 lines. Craig is the only strong reuse option if we want a complete recorder.

## 4. Policy / ToS
- Privileged intents are only `GUILD_PRESENCES`, `GUILD_MEMBERS` and `MESSAGE_CONTENT`. `GUILD_VOICE_STATES` is not privileged and needs no portal toggle; the `Guilds` intent is also enough. Voice receive needs no extra intent, only the `Connect` and `View Channel` channel permissions. `Speak` is not needed if the bot never talks. Join with `selfDeaf: false`: a deafened bot receives nothing.
- Developer Policy (support-dev.discord.com article 8563934450327), via a cached copy. I found no voice-specific recording rule. The relevant general rules:
  - "Do not initiate processes on a user or server's behalf without first obtaining their permission", with clearly labeled opt-in.
  - "Do not enable your Application to bypass or circumvent Discord's privacy, safety, and/or security features", and respect users' ability to remove or opt out of the app.
  - Rule 21: no using "message content obtained through the APIs" to train ML/LLM models without Discord's permission. Voice audio is not "message content" textually, but avoid using transcripts as training data [UNCERTAIN how Discord would read this].
  - No selling or disclosing API Data to brokers or ad networks, and no re-identification.
- Consent: Discord has no technical consent gate for bot receive. Legal consent is on us (GDPR, two-party-consent US states). Practical baseline: announce recording or transcription when the bot joins (post to a text channel), allow opt-out by user, and keep to a private, self-hosted deployment. Craig does the same: it notifies, and any member can leave the recording. I found no authoritative source mandating notification; this is best practice, not law advice.

## 5. Recommendation
Use Node/TypeScript with `discord.js` 14.27 + `@discordjs/voice` 0.19.2 + `@snazzah/davey` (pin to a version you tested, currently 0.1.12) + `prism-media` or `@discordjs/opus`. Reasons: the most active and mainstream stack, DAVE decrypt is built into `VoiceReceiver`, the maintainer of davey also contributes to the voice package, the API gives a clean per-user Opus stream, and the known bugs are small and patchable (#11653 via `patch-package`). Second choice: Rust songbird 0.6.x (best raw performance and `VoiceTick` per-user PCM, but open #310 receive-stall bugs). Avoid discord.py voice-recv and py-cord for now (no released DAVE receive); JDA is workable but needs Java 25 and has noisy decrypt WARNs. Fallback if the Node spike fails (a possible explanation: the unverified June report): self-host Craig (Dysnomia) and feed its tracks to the ASR server.

Spike first: a day-long soak test on a real server with 4+ speakers, joining mid-call, camera on, and a bot restart.

Minimal sketch (discord.js, TypeScript):
```ts
import { Client, GatewayIntentBits } from 'discord.js';
import { joinVoiceChannel, EndBehaviorType, VoiceConnectionStatus, entersState } from '@discordjs/voice';
import prism from 'prism-media';

const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] });

async function join(guildId: string, channelId: string) {
  const guild = await client.guilds.fetch(guildId);
  const conn = joinVoiceChannel({
    guildId, channelId, adapterCreator: guild.voiceAdapterCreator,
    selfDeaf: false,   // required to receive
    selfMute: true,
  });
  await entersState(conn, VoiceConnectionStatus.Ready, 20_000);

  const active = new Set<string>();
  conn.receiver.speaking.on('start', (userId) => {
    if (active.has(userId)) return;
    active.add(userId);
    const opus = conn.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 800 },
    });
    const pcm = opus.pipe(new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 }));
    const chunks: Buffer[] = [];
    pcm.on('data', (b: Buffer) => chunks.push(b));        // 48 kHz s16le stereo, 20 ms frames
    pcm.on('end', () => { active.delete(userId); sendToAsr(userId, Buffer.concat(chunks)); });
    opus.on('error', () => active.delete(userId));        // DAVE decrypt errors surface here
  });
}
```
Notes: downmix and resample to 16 kHz mono before the ASR call. Alternatively forward raw Opus and decode on the GPU box. Segment per utterance (end on silence) or stream windows for live latency. Add a retry for stream errors and a reconnect on `Disconnected` (see #11419).

Songbird equivalent: `Config::default().decode_mode(DecodeMode::Decode(DecodeConfig::default()))`, register `CoreEvent::SpeakingStateUpdate` (SSRC to UserId map) and `CoreEvent::VoiceTick`. Per tick, `tick.speaking: HashMap<Ssrc, VoiceData>`, and `data.decoded_voice: Option<Vec<i16>>` is 48 kHz stereo, 20 ms. Example: `examples/serenity/voice_receive` in `serenity-rs/songbird`.

## Sources
- https://discord.com/blog/every-voice-and-video-call-on-discord-is-now-end-to-end-encrypted
- https://docs.discord.com/developers/topics/voice-connections
- https://docs.discord.com/developers/events/gateway
- https://support-dev.discord.com/hc/en-us/articles/8563934450327-Discord-Developer-Policy (cached copy via web.archive.org)
- https://github.com/discordjs/discord.js (issues #11419, #11445, #11653; packages/voice CHANGELOG)
- https://registry.npmjs.org/@discordjs%2Fvoice and https://registry.npmjs.org/@snazzah%2Fdavey
- https://github.com/imayhaveborkedit/discord-ext-voice-recv (issues/PRs #43-#65) and https://pypi.org/project/discord-ext-voice-recv/
- https://pypi.org/project/discord.py/ , py-cord, nextcord, davey
- https://github.com/Pycord-Development/pycord (CHANGELOG, PRs #3143, #3159, #3185, #3386; issue #3135)
- https://github.com/serenity-rs/songbird (CHANGELOG v0.6.0/v0.6.1, issues #310, #314, examples/serenity/voice_receive)
- https://github.com/discord-jda/JDA (README, issues #2998, #3034) and https://github.com/MinnDevelopment/jdave (issue #32)
- https://github.com/Snazzah/davey (issues #15, #20)
- https://github.com/CraigChat/craig , https://github.com/scripty-bot/scripty , https://github.com/scripty-bot/stt-service
- https://github.com/meanwebuser/discord-voice-transcript (docs/DAVE-RESEARCH.md), https://github.com/SakethKanchi/parley
