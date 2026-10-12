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
function showTab(name) {
  currentTab = name;
  for (const button of document.querySelectorAll('nav button')) button.setAttribute('aria-selected', button.dataset.tab === name);
  for (const section of document.querySelectorAll('.tab')) section.hidden = section.id !== `tab-${name}`;
  if (name === 'live') refreshLive();
  if (name === 'history') refreshHistory();
}
for (const button of document.querySelectorAll('nav button')) button.addEventListener('click', () => showTab(button.dataset.tab));

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
    showTab(currentTab);
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
  const busy = m.loading || m.status === 'downloading';
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

const lineNode = (session, u) => el('div', { class: 'line' },
  el('time', {}, clock(u.started_at - session.started_at)),
  el('span', { class: 'who', title: u.speaker_name }, u.speaker_name),
  el('p', { style: 'margin:0' }, u.text));

// ---- live ------------------------------------------------------------------------------------

let liveSource = null;
let liveSession = null;
async function refreshLive() {
  if (currentTab !== 'live') return;
  const sessions = (await api('/api/sessions')).filter((s) => s.ended_at == null);
  $('#live-sessions').replaceChildren(...sessions.map((s) => el('button', { class: liveSession?.id === s.id ? 'on' : '', onclick: () => openLive(s.id) }, s.title)));
  if (!liveSession && sessions.length) openLive(sessions[0].id);
}

async function openLive(id) {
  liveSource?.close();
  const session = await api(`/api/sessions/${id}`);
  liveSession = session;
  const box = $('#live-transcript');
  box.replaceChildren(...session.utterances.map((u) => lineNode(session, u)));
  liveSource = new EventSource(`/api/sessions/${id}/events`);
  liveSource.addEventListener('line', (event) => {
    box.append(lineNode(session, JSON.parse(event.data)));
    box.lastElementChild.scrollIntoView({ block: 'nearest' });
  });
  liveSource.addEventListener('end', () => { liveSource.close(); box.append(el('p', { class: 'hint' }, 'Session ended.')); liveSession = null; });
  refreshLive();
}

// ---- history ---------------------------------------------------------------------------------

async function refreshHistory() {
  const sessions = await api('/api/sessions');
  $('#history').replaceChildren(...sessions.map((s) => el('li', {},
    el('button', { onclick: (event) => openHistory(s.id, event.currentTarget) },
      s.title, el('small', {}, `${when(s.started_at)} · ${s.lines} lines${s.ended_at == null ? ' · live' : ''}`)))));
  if (!sessions.length) $('#history').replaceChildren(el('li', { class: 'hint' }, 'Nothing yet.'));
}

async function openHistory(id, button) {
  for (const b of document.querySelectorAll('#history button')) b.classList.toggle('on', b === button);
  const session = await api(`/api/sessions/${id}`);
  const remove = async () => {
    if (!confirm('Delete this transcript?')) return;
    await api(`/api/sessions/${id}`, { method: 'DELETE' });
    $('#history-view').replaceChildren(el('p', { class: 'hint' }, 'Deleted.'));
    refreshHistory();
  };
  $('#history-view').replaceChildren(
    el('div', { class: 'toolbar' }, el('h3', {}, session.title),
      ...['md', 'srt', 'json'].map((f) => el('a', { href: `/api/sessions/${id}/export?format=${f}` }, f.toUpperCase())),
      el('button', { onclick: remove }, 'Delete')),
    ...(session.utterances.length ? session.utterances.map((u) => lineNode(session, u)) : [el('p', { class: 'hint' }, 'No lines.')]),
  );
}

// ---- try it ----------------------------------------------------------------------------------

async function transcribeBlob(blob, name) {
  const result = $('#try-result');
  result.textContent = 'Transcribing…';
  const form = new FormData();
  form.append('file', blob, name);
  form.append('response_format', 'verbose_json');
  try {
    const r = await api('/v1/audio/transcriptions', { method: 'POST', body: form });
    const model = models.find((m) => m.id === r.model)?.name ?? r.model ?? 'no model';
    result.replaceChildren(
      el('p', { class: 'meta' }, `${model} · ${r.duration.toFixed(1)} s of audio in ${(r.asr_ms / 1000).toFixed(2)} s`),
      el('p', { id: 'try-text' }, r.text || '(no speech)'));
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
  if (started) return;
  started = true;
  refreshModels();
  setInterval(() => { if (currentTab === 'live') refreshLive().catch(() => {}); }, 5000);
}

(async () => {
  const me = await (await fetch('/api/whoami', { credentials: 'same-origin' })).json();
  if (!me.authorised) showLogin();
  else start();
})();
