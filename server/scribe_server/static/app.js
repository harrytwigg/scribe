// scribe GUI: model picker, live transcripts, history, try-a-clip. No build step.

const $ = (selector) => document.querySelector(selector);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
};

class Unauthorised extends Error {}

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  if (response.status === 401) { showLogin(); throw new Unauthorised(); }
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  const type = response.headers.get('content-type') ?? '';
  return type.includes('json') ? response.json() : response.text();
}

const clock = (seconds) => {
  seconds = Math.max(0, Math.floor(seconds));
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60;
  return (h ? `${h}:` : '') + `${String(m).padStart(h ? 2 : 1, '0')}:${String(s).padStart(2, '0')}`;
};
const when = (epoch) => new Date(epoch * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const formatSize = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

// ---- tabs ------------------------------------------------------------------------------------

let currentTab = 'models';
function showTab(name, id = null) {
  currentTab = name;
  for (const button of document.querySelectorAll('nav button')) button.setAttribute('aria-selected', button.dataset.tab === name);
  for (const section of document.querySelectorAll('.tab')) section.hidden = section.id !== `tab-${name}`;
  // Only the visible tab follows a call.
  if (name !== 'live') liveViewer.close();
  if (name !== 'history') historyViewer.close();
  setRoute(name, id);
  if (name === 'live') refreshLive(id).catch(() => {});
  if (name === 'history') refreshHistory(id).catch(() => {});
}
for (const button of document.querySelectorAll('nav button')) button.addEventListener('click', () => showTab(button.dataset.tab));
window.addEventListener('hashchange', () => {
  const { tab, id } = route();
  if (tab !== currentTab || (id && id !== (tab === 'live' ? liveViewer.id : historyViewer.id))) showTab(tab, id);
});

// ---- sign-in ---------------------------------------------------------------------------------

function showLogin() {
  $('#login').hidden = false;
  for (const section of document.querySelectorAll('.tab')) section.hidden = true;
  $('nav').hidden = true;
}
$('#login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await api('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: $('#token').value }) });
    $('#login').hidden = true;
    $('nav').hidden = false;
    $('#token').value = '';
    start();
  } catch (error) {
    if (!(error instanceof Unauthorised)) $('#login-error').textContent = error.message;
    else $('#login-error').textContent = 'Wrong token.';
  }
});

// ---- status ----------------------------------------------------------------------------------

let models = [];
async function refreshStatus() {
  const status = await api('/api/status');
  const active = models.find((m) => m.id === status.active_model);
  const parts = [active ? `Active: ${active.name}` : 'No model active'];
  if (status.loading_model) parts.push(`loading ${status.loading_model}…`);
  if (status.open_sessions) parts.push(`${status.open_sessions} live`);
  if (status.queue_depth) parts.push(`${status.queue_depth} queued`);
  if (status.diarization && status.diarization !== 'ready' && status.diarization !== 'off') parts.push(`speaker separation ${status.diarization}`);
  $('#status').textContent = parts.join(' · ');
}

// ---- models ----------------------------------------------------------------------------------

const bar = (value) => el('span', { class: 'bar', 'aria-label': `${value} of 5` }, [1, 2, 3, 4, 5].map((i) => el('i', { class: i <= value ? 'on' : '' })));

async function modelAction(method, path, label) {
  try { await api(path, { method }); }
  catch (error) { if (!(error instanceof Unauthorised)) alertInline(`${label}: ${error.message}`); }
  refreshModels();
}

function alertInline(message) {
  const node = el('p', { class: 'error' }, message);
  $('#tab-models').prepend(node);
  setTimeout(() => node.remove(), 6000);
}

function modelCard(m) {
  const actions = [];
  if (m.status === 'not_downloaded') actions.push(el('button', { class: 'primary', onclick: () => modelAction('POST', `/api/models/${m.id}/download`, 'Download') }, `Download (${formatSize(m.size_mb)})`));
  if (m.status === 'downloading') actions.push(el('button', { onclick: () => modelAction('DELETE', `/api/models/${m.id}`, 'Cancel') }, 'Cancel'));
  if (m.status === 'downloaded' && !m.active) {
    actions.push(el('button', { class: 'primary', disabled: m.loading, onclick: () => modelAction('POST', `/api/models/${m.id}/activate`, 'Use') }, m.loading ? 'Loading…' : 'Use'));
    actions.push(el('button', { disabled: m.loading, onclick: () => confirm(`Delete ${m.name}?`) && modelAction('DELETE', `/api/models/${m.id}`, 'Delete') }, 'Delete'));
  }
  return el('article', { class: `card${m.active ? ' active' : ''}`, 'data-model': m.id },
    el('h3', {}, m.name, m.active ? el('span', { class: 'badge on' }, 'active') : null, m.recommended ? el('span', { class: 'badge' }, 'recommended') : null),
    el('p', { class: 'meta' }, m.description),
    el('div', { class: 'bars' }, 'Accuracy', bar(m.accuracy), 'Speed', bar(m.speed)),
    el('p', { class: 'meta' }, `${m.languages} · ${formatSize(m.size_mb)} · ${m.licence} · `, el('a', { href: m.source, target: '_blank', rel: 'noopener' }, 'source')),
    m.status === 'downloading' ? el('progress', { max: 1, value: m.progress }) : null,
    m.error ? el('p', { class: 'error' }, m.error) : null,
    el('div', { class: 'actions' }, actions),
  );
}

let modelsTimer;
async function refreshModels() {
  clearTimeout(modelsTimer);
  try {
    models = await api('/api/models');
    $('#models').replaceChildren(...models.map(modelCard));
    await refreshStatus();
  } catch (error) { if (error instanceof Unauthorised) return; }
  const busy = models.some((m) => m.loading || m.status === 'downloading');
  modelsTimer = setTimeout(refreshModels, busy ? 1000 : 5000);
}

// ---- transcripts -----------------------------------------------------------------------------

// A steady colour per speaker, so a call reads at a glance.
// The golden angle spreads neighbours ("Speaker 1", "Speaker 2") far apart on the wheel.
const hue = (id) => Math.round(([...id].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 9973, 7) * 137.508) % 360);

// Click a name to rename that speaker for the whole session (past and future lines).
async function renameSpeaker(session, u, current) {
  const name = prompt(`Rename ${current} in this session:`, current);
  if (!name?.trim() || name.trim() === current) return;
  try {
    await api(`/api/sessions/${session.id}/speakers/${encodeURIComponent(u.speaker_id)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() }),
    });
    showSpeakerName(u.speaker_id, name.trim());
  } catch (error) { if (!(error instanceof Unauthorised)) alert(error.message); }
}

function showSpeakerName(speakerId, name) {
  for (const node of document.querySelectorAll('.who')) {
    if (node.dataset.speaker === speakerId) { node.textContent = name; node.title = `${name} (click to rename)`; }
  }
}

const lineNode = (session, u) => el('div', { class: 'line' },
  el('time', {}, clock(u.started_at - session.started_at)),
  el('button', {
    class: 'who', type: 'button', title: `${u.speaker_name} (click to rename)`, 'data-speaker': u.speaker_id,
    style: `--hue:${hue(u.speaker_id)}`, onclick: (event) => renameSpeaker(session, u, event.currentTarget.textContent),
  }, u.speaker_name),
  el('p', { style: 'margin:0' }, u.text));

// ---- transcript viewer -----------------------------------------------------------------------

// Live and History share one viewer. It follows an open session as it is transcribed: the event
// stream delivers lines as they land, and a poll every few seconds catches anything the stream
// missed (a dropped connection, a proxy that buffers). Lines are keyed by id, so nothing doubles.
const POLL_MS = 4000;

async function renameSession(id, current) {
  const title = prompt('Rename this call:', current);
  if (!title?.trim() || title.trim() === current) return;
  try {
    await api(`/api/sessions/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: title.trim() }),
    });
    for (const viewer of [liveViewer, historyViewer]) viewer.retitle(id, title.trim());
    refreshLists();
  } catch (error) { if (!(error instanceof Unauthorised)) alert(error.message); }
}

class Viewer {
  constructor(box, { empty, tools } = {}) {
    this.box = box;
    this.empty = empty;
    this.tools = tools ?? (() => []);
    this.session = null;
    this.opening = 0;
  }

  get id() { return this.session?.id ?? null; }

  close() {
    this.source?.close();
    this.source = null;
    clearInterval(this.timer);
    this.timer = null;
    this.session = null;
    this.opening++;
  }

  showEmpty(message) {
    this.close();
    this.box.replaceChildren(el('p', { class: 'hint' }, message ?? this.empty));
  }

  async open(id) {
    this.close();
    const opening = this.opening;
    let session;
    try { session = await api(`/api/sessions/${id}`); } catch (error) {
      if (opening === this.opening && !(error instanceof Unauthorised)) this.showEmpty(`That call is gone (${error.message}).`);
      return false;
    }
    if (opening !== this.opening) return false; // a newer open won
    this.session = session;
    this.ids = new Set();
    this.header = el('div', { class: 'toolbar' });
    this.lines = el('div', { class: 'lines' }, el('p', { class: 'hint' }, 'No lines yet.'));
    this.box.replaceChildren(this.header, this.lines);
    this.renderHeader();
    this.merge(session.utterances);
    if (session.ended_at == null) this.follow();
    return true;
  }

  renderHeader() {
    const s = this.session;
    this.header.replaceChildren(
      el('h3', {}, el('button', { class: 'title', type: 'button', title: 'Click to rename this call', onclick: () => renameSession(s.id, s.title) }, s.title)),
      el('small', { class: s.ended_at == null ? 'state live' : 'state' }, s.ended_at == null ? '● live' : `ended ${when(s.ended_at)}`),
      ...this.tools(s),
    );
  }

  retitle(id, title) {
    if (this.id !== id) return;
    this.session.title = title;
    this.renderHeader();
  }

  merge(utterances) {
    const follow = window.innerHeight + window.scrollY >= document.body.scrollHeight - 80;
    let added = false;
    for (const u of utterances) {
      if (this.ids.has(u.id)) continue;
      this.ids.add(u.id);
      this.lines.querySelector(':scope > .hint')?.remove();
      const node = lineNode(this.session, u);
      node.dataset.started = u.started_at;
      // In time order: two speakers' lines can land out of order.
      const after = [...this.lines.children].find((n) => Number(n.dataset.started) > u.started_at);
      this.lines.insertBefore(node, after ?? null);
      added = true;
    }
    if (added && follow && this.session.ended_at == null) this.lines.lastElementChild?.scrollIntoView({ block: 'nearest' });
  }

  follow() {
    const id = this.session.id;
    const source = new EventSource(`/api/sessions/${id}/events`);
    this.source = source;
    let dropped = false;
    source.addEventListener('error', () => { dropped = true; });
    source.addEventListener('open', () => { if (dropped) { dropped = false; this.sync(); } });
    source.addEventListener('line', (event) => this.merge([JSON.parse(event.data)]));
    source.addEventListener('speaker', (event) => {
      const { speaker_id, name } = JSON.parse(event.data);
      showSpeakerName(speaker_id, name);
    });
    source.addEventListener('title', (event) => { this.retitle(id, JSON.parse(event.data).title); refreshLists(); });
    source.addEventListener('end', (event) => this.ended(JSON.parse(event.data)));
    this.timer = setInterval(() => this.sync(), POLL_MS);
  }

  async sync() {
    const id = this.id;
    if (!id) return;
    let s;
    try { s = await api(`/api/sessions/${id}`); } catch { return; }
    if (this.id !== id) return;
    if (s.title !== this.session.title) this.retitle(id, s.title);
    this.merge(s.utterances);
    // Renames made while the stream was down.
    for (const u of s.utterances) {
      const node = this.lines.querySelector(`.who[data-speaker="${CSS.escape(u.speaker_id)}"]`);
      if (node && node.textContent !== u.speaker_name) showSpeakerName(u.speaker_id, u.speaker_name);
    }
    if (s.ended_at != null) this.ended(s);
  }

  ended(s) {
    if (!this.session || this.session.ended_at != null) return;
    this.source?.close();
    this.source = null;
    clearInterval(this.timer);
    this.timer = null;
    this.session.ended_at = s.ended_at;
    this.renderHeader();
    refreshLists();
  }
}

// ---- routing: the tab and the open call live in the URL, so a refresh keeps them -------------

const TABS = ['models', 'live', 'history', 'try'];

function route() {
  const [tab, id] = location.hash.slice(1).split('/');
  return { tab: TABS.includes(tab) ? tab : 'models', id: id || null };
}

function setRoute(tab, id) {
  const hash = `#${tab}${id ? `/${id}` : ''}`;
  if (location.hash !== hash) history.replaceState(null, '', hash);
}

// ---- live ------------------------------------------------------------------------------------

const liveViewer = new Viewer($('#live-transcript'), {
  empty: 'No call is being transcribed. Start one with /scribe join in Discord, or the Chrome extension.',
});

async function refreshLive(wanted = null) {
  if (currentTab !== 'live') return;
  const open = (await api('/api/sessions')).filter((s) => s.ended_at == null);
  const pick = (id) => () => { setRoute('live', id); liveViewer.open(id).then(() => refreshLive()); };
  $('#live-sessions').replaceChildren(...open.map((s) =>
    el('button', { class: liveViewer.id === s.id ? 'on' : '', onclick: pick(s.id) }, s.title)));
  // Open the call asked for (from the URL); otherwise stay on whatever is shown, even once it has
  // ended, so the screen never jumps; only an empty screen picks the newest live call.
  const target = wanted ?? (liveViewer.id ? null : open[0]?.id);
  if (target && target !== liveViewer.id) {
    setRoute('live', target);
    if (await liveViewer.open(target)) refreshLive();
  } else if (!liveViewer.id && !open.length) {
    liveViewer.showEmpty();
  }
}

// ---- history ---------------------------------------------------------------------------------

const historyViewer = new Viewer($('#history-view'), {
  empty: 'Pick a call.',
  tools: (s) => [
    ...['md', 'srt', 'json'].map((f) => el('a', { href: `/api/sessions/${s.id}/export?format=${f}` }, f.toUpperCase())),
    el('button', { onclick: () => removeSession(s.id) }, 'Delete'),
  ],
});

async function removeSession(id) {
  if (!confirm('Delete this transcript?')) return;
  await api(`/api/sessions/${id}`, { method: 'DELETE' });
  historyViewer.showEmpty('Deleted.');
  setRoute('history');
  refreshHistory();
}

async function refreshHistory(wanted = null) {
  if (currentTab !== 'history') return;
  const sessions = await api('/api/sessions');
  $('#history').replaceChildren(...sessions.map((s) => el('li', {},
    el('button', {
      class: historyViewer.id === s.id ? 'on' : '',
      onclick: () => { setRoute('history', s.id); historyViewer.open(s.id).then(() => refreshHistory()); },
    }, s.title, el('small', {}, `${when(s.started_at)} · ${s.lines} line${s.lines === 1 ? '' : 's'}${s.ended_at == null ? ' · live' : ''}`)))));
  if (!sessions.length) $('#history').replaceChildren(el('li', { class: 'hint' }, 'Nothing yet.'));
  if (wanted && wanted !== historyViewer.id && (await historyViewer.open(wanted))) refreshHistory();
}

function refreshLists() {
  if (currentTab === 'live') refreshLive().catch(() => {});
  if (currentTab === 'history') refreshHistory().catch(() => {});
}

// ---- try it ----------------------------------------------------------------------------------

async function transcribeBlob(blob, name) {
  const result = $('#try-result');
  result.textContent = 'Transcribing…';
  const form = new FormData();
  form.append('file', blob, name);
  form.append('response_format', 'verbose_json');
  const speakers = $('#try-speakers').checked;
  if (speakers) form.append('diarize', 'true');
  try {
    const r = await api('/v1/audio/transcriptions', { method: 'POST', body: form });
    const model = models.find((m) => m.id === r.model)?.name ?? r.model ?? 'no model';
    const meta = `${model} · ${r.duration.toFixed(1)} s of audio in ${(r.asr_ms / 1000).toFixed(2)} s`;
    if (r.segments) {
      result.replaceChildren(
        el('p', { class: 'meta' }, `${meta} · ${r.speakers} speaker${r.speakers === 1 ? '' : 's'}`),
        el('div', { id: 'try-text', class: 'transcript' }, r.segments.length ? r.segments.map((seg) => el('div', { class: 'line' },
          el('time', {}, clock(seg.start)),
          el('span', { class: 'who', style: `--hue:${hue(seg.speaker)}` }, seg.speaker),
          el('p', { style: 'margin:0' }, seg.text))) : '(no speech)'));
    } else {
      result.replaceChildren(
        speakers ? el('p', { class: 'hint' }, 'Speaker separation is not ready yet: transcribed as one speaker.') : '',
        el('p', { class: 'meta' }, meta),
        el('p', { id: 'try-text' }, r.text || '(no speech)'));
    }
  } catch (error) { if (!(error instanceof Unauthorised)) result.replaceChildren(el('p', { class: 'error' }, error.message)); }
}

$('#try-file').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) transcribeBlob(file, file.name);
  // Clear it, so picking the same clip again (to compare models) fires another change.
  event.target.value = '';
});

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Blob([buffer], { type: 'audio/wav' });
}

let recording = null;
$('#try-record').addEventListener('click', async () => {
  const button = $('#try-record');
  if (recording) {
    const { stream, context, chunks } = recording;
    recording = null;
    stream.getTracks().forEach((t) => t.stop());
    await context.close();
    button.textContent = 'Record from mic';
    const length = chunks.reduce((n, c) => n + c.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;
    for (const c of chunks) { samples.set(c, offset); offset += c.length; }
    transcribeBlob(encodeWav(samples, context.sampleRate), 'mic.wav');
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) { $('#try-mic-note').hidden = false; return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  source.connect(processor);
  processor.connect(context.destination);
  recording = { stream, context, chunks };
  button.textContent = 'Stop and transcribe';
});

// ---- start -----------------------------------------------------------------------------------

let started = false;
function start() {
  const { tab, id } = route();
  showTab(tab, id); // also after signing in again, when the tabs were hidden
  if (started) return;
  started = true;
  refreshModels();
  // New and ended calls show up in the lists without a refresh.
  setInterval(refreshLists, POLL_MS);
}

(async () => {
  const me = await (await fetch('/api/whoami', { credentials: 'same-origin' })).json();
  if (!me.authorised) showLogin();
  else start();
})();
