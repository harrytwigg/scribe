// scribe GUI: model picker, live transcripts, history, transcribe a file or podcast link, try-a-clip. No build step.

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

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: 'same-origin', ...options });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail ?? detail; } catch {}
    throw Object.assign(new Error(detail), { status: response.status });
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
  // Back on a tab: its viewer's call, unless the URL names another.
  id ??= name === 'live' ? liveViewer.current : name === 'history' ? historyViewer.current : null;
  for (const button of document.querySelectorAll('nav button')) button.setAttribute('aria-selected', button.dataset.tab === name);
  for (const section of document.querySelectorAll('.tab')) section.hidden = section.id !== `tab-${name}`;
  // Only the visible tab follows a call (the other keeps which call it was on).
  if (name !== 'live') liveViewer.close();
  if (name !== 'history') historyViewer.close();
  setRoute(name, id);
  if (name === 'live') refreshLive(id).catch(() => {});
  if (name === 'history') refreshHistory(id).catch(() => {});
}
for (const button of document.querySelectorAll('nav button')) button.addEventListener('click', () => showTab(button.dataset.tab));
window.addEventListener('hashchange', () => {
  const { tab, id } = route();
  if (tab !== currentTab || (id && id !== (tab === 'live' ? liveViewer.current : historyViewer.current))) showTab(tab, id);
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
  catch (error) { alertInline(`${label}: ${error.message}`); }
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
  } catch {}
  const busy = models.some((m) => m.loading || m.status === 'downloading');
  modelsTimer = setTimeout(refreshModels, busy ? 1000 : 5000);
}

// ---- transcripts -----------------------------------------------------------------------------

// A steady colour per speaker, so a call reads at a glance.
// The golden angle spreads neighbours ("Speaker 1", "Speaker 2") far apart on the wheel.
const hue = (id) => Math.round(([...id].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 9973, 7) * 137.508) % 360);

// Click a name to rename that speaker for the whole session (past and future lines).
async function renameSpeaker(session, u, button) {
  const current = button.textContent;
  const name = prompt(`Rename ${current} in this session:`, current);
  if (!name?.trim() || name.trim() === current) return;
  try {
    const renamed = await api(`/api/sessions/${session.id}/speakers/${encodeURIComponent(u.speaker_id)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() }),
    });
    showSpeakerName(button.closest('.transcript'), u.speaker_id, renamed.name);
  } catch (error) { alert(error.message); }
}

// Within one transcript only: speaker ids ("tab", "mic") repeat across calls.
function showSpeakerName(scope, speakerId, name) {
  for (const node of scope?.querySelectorAll('.who') ?? []) {
    if (node.dataset.speaker === speakerId && node.textContent !== name) {
      node.textContent = name;
      node.title = `${name} (click to rename)`;
    }
  }
}

const lineNode = (session, u) => el('div', { class: 'line', 'data-started': u.started_at },
  el('time', {}, clock(u.started_at - session.started_at)),
  el('button', {
    class: 'who', type: 'button', title: `${u.speaker_name} (click to rename)`, 'data-speaker': u.speaker_id,
    style: `--hue:${hue(u.speaker_id)}`, onclick: (event) => renameSpeaker(session, u, event.currentTarget),
  }, u.speaker_name),
  el('p', { style: 'margin:0' }, u.text));

// ---- transcript viewer -----------------------------------------------------------------------

// Live and History share one viewer. It follows an open call as it is transcribed: the event
// stream delivers lines as they land, and a poll every few seconds catches anything the stream
// missed (a dropped connection, a proxy that buffers). Lines are keyed by id, so nothing doubles.
//
// `current` is the call the viewer is on: shown, or still loading. It survives leaving the tab
// (the viewer stops following while hidden) so coming back reopens the same call, and a list
// refresh never takes the viewer to another call while one is loading.
const POLL_MS = 4000;

async function renameSession(id, current) {
  const title = prompt('Rename this call:', current);
  if (!title?.trim() || title.trim() === current) return;
  try {
    const session = await api(`/api/sessions/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: title.trim() }),
    });
    for (const viewer of [liveViewer, historyViewer]) viewer.retitle(id, session.title);
    refreshLists();
  } catch (error) { alert(error.message); }
}

class Viewer {
  constructor(box, { empty, tools } = {}) {
    this.box = box;
    this.empty = empty;
    this.tools = tools ?? (() => []);
    this.current = null;
    this.session = null;
    this.loading = false;
    this.opening = 0;
  }

  // Showing its call, or loading it (as opposed to stopped while its tab was hidden).
  get active() { return this.loading || this.session != null; }

  /** Stop following (the tab is hidden, or another call opens); remember which call it was. */
  close() {
    this.source?.close();
    this.source = null;
    clearInterval(this.timer);
    this.timer = null;
    this.session = null;
    this.loading = false;
    this.opening++;
  }

  showEmpty(message) {
    this.close();
    this.current = null;
    this.box.replaceChildren(el('p', { class: 'hint' }, message ?? this.empty));
  }

  async open(id) {
    this.close();
    this.current = id;
    this.loading = true;
    const opening = this.opening;
    let session;
    try { session = await api(`/api/sessions/${id}`); } catch (error) {
      if (opening !== this.opening) return false;
      this.showEmpty(error.status === 404 ? 'That call was deleted.' : `Cannot open that call: ${error.message}`);
      return false;
    }
    if (opening !== this.opening) return false; // a newer open (or a close) won
    this.loading = false;
    this.session = session;
    this.ids = new Set();
    this.header = el('div', { class: 'toolbar' });
    this.recording = el('div', { class: 'toolbar recording' });
    this.lines = el('div', { class: 'lines' }, el('p', { class: 'hint' }, 'No lines yet.'));
    this.box.replaceChildren(this.header, this.recording, this.lines);
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
    this.renderRecording();
  }

  /** The call's recording, if it has one: what is kept, and the buttons that redo the transcript from it. */
  renderRecording(note = this.note) {
    this.note = note;
    const s = this.session, r = s.recording;
    if (!r) return this.recording.replaceChildren();
    const ended = s.ended_at != null;
    const busy = r.job != null || this.working;
    const channels = Object.entries(r.channels).map(([name, c]) => `${name} ${clock((c.ended_at ?? 0) - (c.started_at ?? 0))}`).join(', ');
    this.recording.replaceChildren(
      el('small', { class: 'state' },
        `Audio kept: ${channels}, ${(r.bytes / 1048576).toFixed(1)} MB.`,
        r.gaps ? ` ${r.gaps} gap${r.gaps === 1 ? '' : 's'} in the transcript.` : ''),
      note ? el('small', { class: 'recording-note' }, note) : null,
      ended && r.gaps ? el('button', { disabled: busy, title: 'Transcribe the audio of each [unavailable] line and put the text in its place', onclick: () => reprocessSession(s.id, 'gaps') }, 'Fill gaps from the recording') : null,
      ended ? el('button', { disabled: busy, title: 'Transcribe the whole recording again, replacing this transcript', onclick: () => reprocessSession(s.id, 'all') }, 'Transcribe it again') : null,
      ...Object.keys(r.channels).map((name) => el('a', { href: `/api/sessions/${s.id}/recording/${name}/audio`, title: `The ${name} audio as one WAV file` }, `${name} audio`)),
    );
  }

  retitle(id, title) {
    if (this.session?.id !== id) return;
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
      // In time order (two speakers' lines can land out of order); the usual case is the newest.
      const last = this.lines.lastElementChild;
      if (!last || Number(last.dataset.started) <= u.started_at) this.lines.append(node);
      else this.lines.insertBefore(node, [...this.lines.children].find((n) => Number(n.dataset.started) > u.started_at));
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
      showSpeakerName(this.box, speaker_id, name);
    });
    source.addEventListener('title', (event) => { this.retitle(id, JSON.parse(event.data).title); refreshLists(); });
    source.addEventListener('end', (event) => this.ended(JSON.parse(event.data)));
    this.timer = setInterval(() => this.sync(), POLL_MS);
  }

  async sync() {
    const id = this.session?.id;
    if (!id) return;
    let s;
    try { s = await api(`/api/sessions/${id}`); } catch (error) {
      if (this.session?.id === id && error.status === 404) {
        this.showEmpty('That call was deleted.');
        refreshLists();
      }
      return;
    }
    if (this.session?.id !== id) return;
    if (s.title !== this.session.title) this.retitle(id, s.title);
    this.session.recording = s.recording;
    this.renderRecording();
    this.merge(s.utterances);
    // Speakers renamed while the stream was down.
    for (const [speakerId, name] of Object.entries(s.speakers ?? {})) showSpeakerName(this.box, speakerId, name);
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

const TABS = ['models', 'live', 'history', 'transcribe', 'try', 'extension'];

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
  if (currentTab !== 'live') return;
  $('#live-sessions').replaceChildren(...open.map((s) =>
    el('button', { class: liveViewer.current === s.id ? 'on' : '', onclick: () => showCall('live', s.id) }, s.title)));
  // The call asked for, else the one the viewer is on (even once it has ended, so the screen
  // never jumps), else the newest live call.
  const target = wanted ?? liveViewer.current ?? open[0]?.id;
  if (!target) {
    if (liveViewer.current !== null || !liveViewer.box.querySelector(':scope > .hint')) liveViewer.showEmpty();
  } else if (target !== liveViewer.current || !liveViewer.active) {
    showCall('live', target);
  }
}

// Open a call in a tab's viewer and put it in the URL.
async function showCall(tab, id) {
  const viewer = tab === 'live' ? liveViewer : historyViewer;
  setRoute(tab, id);
  if (await viewer.open(id)) (tab === 'live' ? refreshLive : refreshHistory)().catch(() => {});
}

// ---- history ---------------------------------------------------------------------------------

const historyViewer = new Viewer($('#history-view'), {
  empty: 'Pick a call.',
  tools: (s) => [
    ...['md', 'srt', 'json'].map((f) => el('a', { href: `/api/sessions/${s.id}/export?format=${f}` }, f.toUpperCase())),
    el('button', { onclick: () => removeSession(s.id) }, 'Delete'),
  ],
});

/** Fill a call's gaps from its recording, or transcribe all of it again, and show the result. */
async function reprocessSession(id, mode) {
  if (mode === 'all' && !confirm('Transcribe the whole recording again? The new transcript replaces this one. Names given to the other side\'s speakers are not kept.')) return;
  const viewer = historyViewer;
  const note = (text) => viewer.session?.id === id && viewer.renderRecording(text);
  viewer.working = true;
  note('Starting…');
  try {
    let job = await api(`/api/sessions/${id}/reprocess`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }),
    });
    while (['queued', 'fetching', 'decoding', 'transcribing'].includes(job.status)) {
      note(job.status === 'queued' ? 'Waiting its turn…' : `Transcribing… ${Math.round(job.progress * 100)}%`);
      await new Promise((resolve) => setTimeout(resolve, 1000));
      job = await api(`/api/transcriptions/${job.id}`);
    }
    viewer.working = false;
    const r = job.result ?? {};
    const text = job.status !== 'done' ? `Failed: ${job.error ?? job.status}`
      : mode === 'all' ? `Done: ${r.lines} lines from the recording.`
      : `Done: ${r.filled} of ${r.gaps} gaps filled${r.unrecoverable ? `, ${r.unrecoverable} with no audio` : ''}${r.failed ? `, ${r.failed} failed` : ''}.`;
    if (viewer.session?.id === id) { await viewer.open(id); note(text); }
  } catch (error) {
    viewer.working = false;
    note(`Failed: ${error.message}`);
  }
}

async function removeSession(id) {
  if (!confirm('Delete this transcript?')) return;
  await api(`/api/sessions/${id}`, { method: 'DELETE' });
  historyViewer.showEmpty('Deleted.');
  if (liveViewer.current === id) liveViewer.showEmpty();
  setRoute('history');
  refreshHistory();
}

async function refreshHistory(wanted = null) {
  if (currentTab !== 'history') return;
  const sessions = await api('/api/sessions');
  if (currentTab !== 'history') return;
  $('#history').replaceChildren(...sessions.map((s) => el('li', {},
    el('button', { class: historyViewer.current === s.id ? 'on' : '', onclick: () => showCall('history', s.id) },
      s.title, el('small', {}, `${when(s.started_at)} · ${s.lines} line${s.lines === 1 ? '' : 's'}${s.ended_at == null ? ' · live' : ''}`)))));
  if (!sessions.length) $('#history').replaceChildren(el('li', { class: 'hint' }, 'Nothing yet.'));
  const target = wanted ?? historyViewer.current;
  if (target && (target !== historyViewer.current || !historyViewer.active)) showCall('history', target);
}

function refreshLists() {
  if (currentTab === 'live') refreshLive().catch(() => {});
  if (currentTab === 'history') refreshHistory().catch(() => {});
}

// ---- transcribe: uploaded files and podcast links, queued as background jobs ----------------

const TX_JOBS = 'scribe-tx-jobs';
const remembered = {
  get() { try { return JSON.parse(localStorage.getItem(TX_JOBS) ?? '[]'); } catch { return []; } },
  set(ids) { try { localStorage.setItem(TX_JOBS, JSON.stringify(ids)); } catch {} },
};
const ACTIVE = ['queued', 'fetching', 'decoding', 'transcribing'];
const mb = (bytes) => (bytes / 1e6).toFixed(1);

// Every item in the list: a file still being sent, a link being handed over, or a server job.
// { key, label, id (server job), job (its last state), upload (XHR), pending, error, node, open }
const txItems = [];
let txTimer = null;

function txMessage(text, isError = true) {
  $('#tx-status').textContent = text;
  $('#tx-status').className = isError ? 'error' : 'hint';
}

/** POST the file itself as the body, with upload progress (fetch cannot report it). */
function uploadFile(item, file, params) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    item.upload = request;
    request.open('POST', `/api/transcriptions?${new URLSearchParams(params)}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) { item.uploaded = event.loaded / event.total; txRender(item); }
    };
    request.onabort = () => reject(Object.assign(new Error('Cancelled.'), { cancelled: true }));
    request.onerror = () => reject(new Error('The upload failed: the server could not be reached.'));
    request.onload = () => {
      item.upload = null;
      let body = {};
      try { body = JSON.parse(request.responseText); } catch {}
      if (request.status >= 200 && request.status < 300) resolve(body);
      else reject(new Error(body.detail ?? request.statusText));
    };
    request.send(file);
  });
}

function txPersist() {
  // Only what is still running: a reload picks those up; finished ones are in History.
  remembered.set(txItems.filter((item) => item.id && txActive(item)).map((item) => item.id));
}

function txState(item) {
  const job = item.job;
  if (item.error) return { text: item.error, bar: null, cls: item.cancelled ? '' : 'error' };
  if (item.upload || item.pending) {
    return item.upload && item.uploaded != null
      ? { text: `Uploading… ${Math.round(item.uploaded * 100)}%`, bar: item.uploaded }
      : { text: item.pending ? 'Waiting to be sent…' : 'Sending…', bar: null };
  }
  if (!job) return { text: 'Starting…', bar: null };
  if (job.status === 'queued') return { text: `Queued, ${job.queue_position === 1 ? 'next up' : `${job.queue_position - 1} ahead`}`, bar: 0 };
  if (job.status === 'fetching') {
    const d = job.download;
    if (d && d.total) return { text: `Downloading ${mb(d.done)} of ${mb(d.total)} MB`, bar: job.progress };
    return { text: d && d.done ? `Downloading ${mb(d.done)} MB` : 'Looking up the episode…', bar: null };
  }
  if (job.status === 'decoding') return { text: 'Decoding the audio…', bar: job.progress };
  if (job.status === 'transcribing') {
    return job.step
      ? { text: `${job.step[0].toUpperCase()}${job.step.slice(1)}…`, bar: null }
      : { text: `Transcribing ${Math.round(job.stage_progress * 100)}%`, bar: job.progress };
  }
  if (job.status === 'done') return { text: `Done: ${(job.result.duration / 60).toFixed(1)} min of audio in ${(job.result.asr_ms / 1000).toFixed(1)} s`, bar: 1 };
  if (job.status === 'cancelled') return { text: 'Cancelled.', bar: null };
  return { text: job.error ?? 'The transcription failed.', bar: null, cls: 'error' };
}

const txActive = (item) => !item.error && (item.pending || item.upload || !item.job || ACTIVE.includes(item.job.status));

function txRender(item) {
  const { text, bar, cls } = txState(item);
  const done = item.job?.status === 'done';
  if (!item.node) {
    item.node = el('li', { class: 'tx-item' });
    $('#tx-queue').append(item.node);
  }
  const title = item.job?.episode || item.job?.status === 'done' ? item.job.title : item.label;
  const progress = el('progress', { max: 1 });
  if (bar != null) progress.value = bar;
  item.node.replaceChildren(
    el('div', { class: 'toolbar' },
      el('strong', {}, title),
      el('span', { class: `state ${cls ?? ''}` }, text),
      done ? el('button', { type: 'button', onclick: () => txToggle(item) }, item.open ? 'Hide transcript' : 'Show transcript') : '',
      txActive(item)
        ? el('button', { type: 'button', onclick: () => txCancel(item) }, 'Cancel')
        : el('button', { type: 'button', title: 'Remove from this list (a transcript stays in History)', onclick: () => txDismiss(item) }, 'Remove')),
    txActive(item) ? progress : '',
    item.view ?? '');
  txSummary();
}

function txSummary() {
  const waiting = txItems.filter(txActive).length;
  $('#tx-summary').textContent = txItems.length ? `${txItems.length - waiting} of ${txItems.length} finished` : '';
}

async function txToggle(item) {
  item.open = !item.open;
  item.view = item.open ? await txTranscript(item.job) : null;
  txRender(item);
}

async function txCancel(item) {
  if (item.upload) item.upload.abort();
  else if (item.pending) { item.error = 'Cancelled.'; item.cancelled = true; item.pending = false; txRender(item); }
  else if (item.id) { try { await api(`/api/transcriptions/${item.id}`, { method: 'DELETE' }); } catch {} txPoll(); }
}

function txDismiss(item) {
  txItems.splice(txItems.indexOf(item), 1);
  item.node?.remove();
  txPersist();
  txSummary();
}

async function txTranscript(job) {
  const session = await api(`/api/sessions/${job.session_id}`);
  const r = job.result;
  const text = session.utterances.map((u) => u.text).join('\n');
  const copy = el('button', { type: 'button', onclick: async () => {
    try { await navigator.clipboard.writeText(r.text); copy.textContent = 'Copied'; } catch { copy.textContent = 'Copy failed'; }
  } }, 'Copy text');
  return el('div', { class: 'tx-view' },
    el('div', { class: 'toolbar' },
      el('span', { class: 'state' }, `${r.speakers} speaker${r.speakers === 1 ? '' : 's'}`),
      ...['md', 'srt', 'json'].map((f) => el('a', { href: `/api/sessions/${session.id}/export?format=${f}` }, f.toUpperCase())),
      copy,
      el('a', { href: `#history/${session.id}` }, 'Open in History')),
    r.note ? el('p', { class: 'hint' }, r.note) : '',
    el('div', { class: 'transcript tx-text' }, text ? session.utterances.map((u) => el('div', { class: 'line' },
      el('time', {}, clock(u.started_at - session.started_at)),
      el('span', { class: 'who', style: `--hue:${hue(u.speaker_id)}` }, u.speaker_name),
      el('p', { style: 'margin:0' }, u.text))) : el('p', { class: 'hint' }, '(no speech)')));
}

/** One poll of every job the server still works on. */
let txPolling = false;
async function txPoll() {
  if (txPolling) return; // the last round is still out
  txPolling = true;
  try {
    for (const item of txItems.filter((i) => i.id && !i.error && (!i.job || ACTIVE.includes(i.job.status)))) {
      try { item.job = await api(`/api/transcriptions/${item.id}`); } catch (error) {
        item.error = error.status === 404 ? 'No longer on the server (look in History).' : error.message;
      }
      if (item.job && !ACTIVE.includes(item.job.status)) txPersist();
      const solo = txItems.length === 1;
      if (item.job?.status === 'done' && solo && item.open == null) { item.open = true; item.view = await txTranscript(item.job); }
      txRender(item);
    }
  } finally { txPolling = false; }
  if (!txItems.some((item) => item.id && !item.error && (!item.job || ACTIVE.includes(item.job.status)))) {
    clearInterval(txTimer);
    txTimer = null;
  }
}

function txWatch() {
  txPersist();
  if (!txTimer) txTimer = setInterval(txPoll, 1500);
  txPoll();
}

function txAdd(label) {
  const item = { label, pending: true };
  txItems.push(item);
  txRender(item);
  return item;
}

$('#tx-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = [...$('#tx-file').files];
  const links = $('#tx-links').value.split('\n').map((l) => l.trim()).filter(Boolean);
  if (!files.length && !links.length) return txMessage('Choose audio files or paste podcast links (one per line).');
  txMessage('', false);
  const count = Number.parseInt($('#tx-count').value, 10);
  const speakers = count >= 1 ? { speakers: Math.min(count, 20) } : {}; // a known number implies separating
  const diarize = $('#tx-speakers').checked || 'speakers' in speakers;
  $('#tx-file').value = ''; // sent below; the next batch starts from an empty form
  $('#tx-links').value = '';
  const started = [...links.map((link) => ({ link, item: txAdd(link) })), ...files.map((file) => ({ file, item: txAdd(file.name) }))];
  // One after another: the server queues them and runs a few at a time.
  for (const { link, file, item } of started) {
    if (item.error) continue; // cancelled while waiting its turn
    item.pending = false;
    try {
      const job = link
        ? await api('/api/transcriptions/link', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: link, diarize, ...speakers }),
        })
        : await uploadFile(item, file, { filename: file.name, diarize, ...speakers });
      item.id = job.id;
      item.job = job;
      txWatch();
    } catch (error) {
      item.error = error.message;
      item.cancelled = error.cancelled;
      item.upload = null;
    }
    txRender(item);
  }
});

function txRestore() {
  for (const id of remembered.get()) {
    const item = { id, label: 'Transcription' };
    txItems.push(item);
    txRender(item);
  }
  if (txItems.length) txWatch();
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
  } catch (error) { result.replaceChildren(el('p', { class: 'error' }, error.message)); }
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
  showTab(tab, id);
  if (started) return;
  started = true;
  $('#ext-server').textContent = location.origin; // what the extension's server setting should say
  refreshModels();
  txRestore(); // transcriptions that were running when the page reloaded
  // New and ended calls show up in the lists without a refresh.
  setInterval(refreshLists, POLL_MS);
}

start();
