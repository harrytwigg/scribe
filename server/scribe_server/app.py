"""HTTP API and GUI (docs/spec.md, 3.3 and 3.4)."""

import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from . import exports
from .audio import RATE, AudioError, Vad, decode, speech_chunks, speech_spans
from .auth import COOKIE, Auth
from .diarize import DiarizerSlot, Speaker, SpeakerBook, Turn
from .models import Conflict, ModelManager, NoActiveModel, NotFound
from .store import Store
from .worker import GpuWorker, QueueFull

log = logging.getLogger("scribe.app")
STATIC = Path(__file__).parent / "static"


@dataclass
class Services:
    models: ModelManager
    store: Store
    worker: GpuWorker
    vad: Vad
    auth: Auth
    audio_dir: Path | None = None
    # Host headers the server answers to; None skips the check (tests). Stops DNS rebinding.
    allowed_hosts: list[str] | None = None
    max_upload_bytes: int = 512 * 1024 * 1024
    # Holds the number of open sessions, for the host's idle check to read without a token.
    open_sessions_file: Path | None = None
    # A browser session with no line or heartbeat for this long is ended: the browser quit or
    # crashed mid-call, and nothing else would end it (docs/spec.md, 4).
    browser_idle_s: float = 300
    # Splits a channel that mixes several voices into speakers (docs/spec.md, 3.8), and the CPU
    # thread it runs on, so it never holds up the GPU queue.
    diarization: DiarizerSlot = field(default_factory=DiarizerSlot)
    # Live calls and uploaded recordings diarize on separate threads (and models), so a long
    # upload never holds up a call. Built on first use when not given.
    diarize_worker: GpuWorker | None = None
    upload_worker: GpuWorker | None = None


@dataclass
class Transcript:
    text: str
    model_id: str | None
    asr_ms: int
    duration_s: float


class Hub:
    """Live transcript lines, fanned out to every SSE listener of a session."""

    def __init__(self) -> None:
        self._listeners: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._listeners[session_id].add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        self._listeners[session_id].discard(queue)

    def publish(self, session_id: str, event: str, data: dict) -> None:
        # asyncio queues are not thread-safe: publish only from `async def` handlers. This raises
        # RuntimeError when called from a threadpool (a plain `def` handler).
        asyncio.get_running_loop()
        for queue in list(self._listeners.get(session_id, ())):
            try:
                queue.put_nowait((event, data))
            except asyncio.QueueFull:
                pass


class Login(BaseModel):
    token: str


class NewSession(BaseModel):
    source: str = "manual"
    title: str | None = None
    guild: str | None = None
    channel: str | None = None


class Marker(BaseModel):
    speaker_id: str
    speaker_name: str
    started_at: float
    text: str


class Rename(BaseModel):
    name: str


class OptOut(BaseModel):
    source: str
    user_id: str
    opted_out: bool


def create_app(services: Services) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async def loop():
            while True:
                await asyncio.sleep(60)
                try:
                    await reap_idle()
                except Exception:
                    log.exception("reaping idle browser sessions failed")

        task = asyncio.create_task(loop())
        yield
        task.cancel()

    app = FastAPI(title="scribe", docs_url=None, redoc_url=None, lifespan=lifespan)
    hub = Hub()
    s = services
    # When each open session last showed signs of life. A session from before a restart counts
    # from the restart, so it gets the full grace period.
    last_seen: dict[str, float] = {}
    booted = time.time()

    def touch(session_id: str) -> None:
        last_seen[session_id] = time.time()

    async def reap_idle(now: float | None = None) -> list[str]:
        """End browser sessions that went quiet: no line and no heartbeat for `browser_idle_s`."""
        now = time.time() if now is None else now
        reaped = []
        for session in s.store.open_sessions("browser"):
            if now - last_seen.get(session["id"], booted) > s.browser_idle_s:
                s.store.end_session(session["id"])
                last_seen.pop(session["id"], None)
                forget(session["id"])
                hub.publish(session["id"], "end", s.store.get_session(session["id"]))
                reaped.append(session["id"])
        if reaped:
            log.info("ended %d abandoned browser session(s)", len(reaped))
            write_open_sessions()
        return reaped

    app.state.reap_idle = reap_idle
    if s.allowed_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=s.allowed_hosts)

    def guard(request: Request) -> None:
        s.auth.guard(request)

    def write_open_sessions() -> None:
        if s.open_sessions_file:
            tmp = s.open_sessions_file.with_suffix(".tmp")
            tmp.write_text(f"{s.store.count_open_sessions()}\n")
            tmp.replace(s.open_sessions_file)

    write_open_sessions()
    def cpu(attr: str) -> GpuWorker:
        if getattr(s, attr) is None:
            setattr(s, attr, GpuWorker(32, name=attr.replace("_", "-"), label="speaker separation"))
        return getattr(s, attr)

    # Each session's diarized speakers, loaded from the store on first use. Only the diarize
    # worker thread changes a book, one utterance at a time.
    books: dict[str, SpeakerBook] = {}
    # An utterance's turns and speakers, from diarizing it until its lines are stored: a retry
    # (after a busy GPU queue, or a client timeout) reuses them rather than diarizing again, so the
    # book never learns the same audio twice and the retry's lines get the same keys.
    pending: dict[tuple[str, str], list] = {}

    def forget(session_id: str) -> None:
        for key in [k for k in books if k.startswith(f"{session_id}/")]:
            books.pop(key, None)
        for key in [k for k in pending if k[0] == session_id]:
            pending.pop(key, None)

    def book_for(session_id: str, prefix: str) -> SpeakerBook:
        key = f"{session_id}/{prefix}"
        if key not in books:
            book = SpeakerBook()
            for speaker_id, voice, weight in s.store.voices(session_id, f"{prefix}#"):
                number = int(speaker_id.rsplit("#", 1)[1])
                book.speakers.append(Speaker(number, np.frombuffer(voice, dtype=np.float32).copy(), weight))
            book.speakers.sort(key=lambda sp: sp.number)
            books[key] = book
        return books[key]

    def run_asr(audio) -> Transcript:
        """Runs on the GPU worker thread."""
        started = time.monotonic()
        chunks = speech_chunks(audio, s.vad)
        if not chunks:
            return Transcript("", s.models.active_id, 0, len(audio) / RATE)
        model = s.models.acquire()
        try:
            text = " ".join(t for t in (model.engine.transcribe(chunk) for chunk in chunks) if t)
        finally:
            s.models.release(model)
        return Transcript(text.strip(), model.id, int((time.monotonic() - started) * 1000), len(audio) / RATE)

    async def read_upload(file: UploadFile) -> bytes:
        if file.size is not None and file.size > s.max_upload_bytes:
            raise HTTPException(413, f"upload larger than {s.max_upload_bytes // (1024 * 1024)} MB")
        return await file.read()

    async def decode_upload(data: bytes, model: str | None = None) -> np.ndarray:
        if model and model != s.models.active_id:
            raise HTTPException(
                409, f"model {model!r} is not active; the active model is {s.models.active_id!r} (pick one in the GUI)"
            )
        if s.models.active_id is None:
            raise HTTPException(503, "no model is active; pick one in the GUI")
        try:
            # Off the event loop: decoding a long upload must not stall live streams.
            return await asyncio.to_thread(decode, data)
        except AudioError as error:
            raise HTTPException(400, str(error)) from error

    async def on_worker(worker: GpuWorker, fn):
        try:
            return await worker.run(fn)
        except QueueFull as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error
        except NoActiveModel as error:
            raise HTTPException(503, str(error)) from error

    async def transcribe(data: bytes, model: str | None = None) -> Transcript:
        audio = await decode_upload(data, model)
        return await on_worker(s.worker, lambda: run_asr(audio))

    def cut(audio: np.ndarray, start: float, end: float) -> np.ndarray:
        return audio[int(start * RATE) : int(end * RATE)]

    def words_by_turn(audio: np.ndarray, spans: list[tuple[float, float]]) -> list[Transcript] | None:
        """Runs on the GPU worker. Transcribe the whole utterance once, as without diarization,
        with word timings, and give each word to the turn it falls in (or the nearest). None if
        the model has no word timings.

        Transcribing each turn on its own lost words: a short clip cut at a turn edge loses the
        context, and Parakeet returned nothing at all for one 3 s turn (docs/spec.md, 3.8).
        """
        started = time.monotonic()
        chunks = speech_spans(audio, s.vad)
        model = s.models.acquire()
        try:
            if not getattr(model.engine, "word_timings", False):
                return None
            words_of = model.engine.transcribe_words
            words = []
            for a, b in chunks:
                got = words_of(audio[a:b])
                if got is None:
                    return None
                # A word's place is near its start: onnx-asr's word ends run to the next word's
                # start, so a word before a pause would otherwise land in the pause (and maybe
                # with the next speaker).
                words += [(w.text, w.start + min(w.end - w.start, 0.3) / 2 + a / RATE) for w in got]
        finally:
            s.models.release(model)
        texts = [""] * len(spans)
        for text, t in words:
            inside = [i for i, (a, b) in enumerate(spans) if a <= t < b]
            index = inside[0] if inside else min(range(len(spans)), key=lambda i: min(abs(t - spans[i][0]), abs(t - spans[i][1])))
            texts[index] += text
        asr_ms = int((time.monotonic() - started) * 1000)
        return [Transcript(text.strip(), model.id, asr_ms, (b - a)) for text, (a, b) in zip(texts, spans)]

    def each_turn(audio: np.ndarray, spans: list[tuple[float, float]]) -> list[Transcript]:
        """Runs on the GPU worker, for a model without word timings (Canary): each turn on its
        own, reaching to the middle of the gaps around it (and the first and last to the ends), so
        speech the segmenter clipped or missed still reaches the model."""
        cuts = []
        for i, (a, b) in enumerate(spans):
            start = 0.0 if i == 0 else (spans[i - 1][1] + a) / 2
            end = len(audio) / RATE if i == len(spans) - 1 else (b + spans[i + 1][0]) / 2
            cuts.append((min(start, a), max(end, b)))
        return [run_asr(cut(audio, a, b)) for a, b in cuts]

    async def transcribe_turns(audio: np.ndarray, spans: list[tuple[float, float]]) -> list[Transcript]:
        """The text of each turn, in a single GPU job so a busy queue never splits an utterance."""
        if not spans:
            return []
        return await on_worker(s.worker, lambda: words_by_turn(audio, spans) or each_turn(audio, spans))

    # -- auth -----------------------------------------------------------------------------------

    @app.post("/api/login")
    def login(body: Login, request: Request, response: Response) -> dict:
        if not s.auth.same_origin(request):
            raise HTTPException(403, "cross-origin request refused")
        if not s.auth.check_token(body.token):
            raise HTTPException(401, "wrong token")
        response.set_cookie(COOKIE, s.auth.new_session(), httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return {"ok": True}

    @app.post("/api/logout")
    def logout(request: Request, response: Response) -> dict:
        s.auth.end_session(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/whoami")
    def whoami(request: Request) -> dict:
        return {"authorised": s.auth.authorised(request), "auth_enabled": s.auth.enabled}

    # -- models ---------------------------------------------------------------------------------

    @app.get("/api/status", dependencies=[Depends(guard)])
    def status() -> dict:
        return {
            "active_model": s.models.active_id,
            "loading_model": s.models.loading_id,
            "queue_depth": s.worker.depth,
            "open_sessions": s.store.count_open_sessions(),
            "diarization": s.diarization.status,
        }

    @app.get("/api/models", dependencies=[Depends(guard)])
    def list_models() -> list[dict]:
        return s.models.list()

    @app.post("/api/models/{model_id}/download", dependencies=[Depends(guard)], status_code=202)
    def download_model(model_id: str) -> dict:
        s.models.download(model_id)
        return s.models.status(model_id)

    @app.post("/api/models/{model_id}/activate", dependencies=[Depends(guard)], status_code=202)
    def activate_model(model_id: str) -> dict:
        s.models.activate(model_id)
        return s.models.status(model_id)

    @app.delete("/api/models/{model_id}", dependencies=[Depends(guard)])
    def delete_model(model_id: str) -> dict:
        s.models.delete(model_id)
        return s.models.status(model_id)

    @app.exception_handler(Conflict)
    def on_conflict(_: Request, error: Conflict) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(NotFound)
    def on_not_found(_: Request, error: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    # -- OpenAI-compatible transcription --------------------------------------------------------

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(guard)])
    async def openai_transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(None),
        response_format: str = Form("json"),
        diarize: bool = Form(False),
    ):
        """`diarize` (scribe's own field): label who spoke, as `segments` in verbose_json."""
        if diarize and (diarizer := s.diarization.ready()):
            return await diarized_file(await read_upload(file), model, response_format, diarizer)
        result = await transcribe(await read_upload(file), model)
        if response_format == "text":
            return PlainTextResponse(result.text)
        if response_format == "verbose_json":
            return {"text": result.text, "model": result.model_id, "duration": result.duration_s, "asr_ms": result.asr_ms}
        return {"text": result.text}

    async def diarized_file(data: bytes, model: str | None, response_format: str, diarizer):
        started = time.monotonic()
        audio = await decode_upload(data, model)
        try:
            turns = await on_worker(cpu("upload_worker"), lambda: diarizer.whole(audio))
        except HTTPException:
            raise
        except Exception:
            log.exception("diarizing an upload failed; transcribing it as one speaker")
            turns = []
        if not turns:  # none found (quiet or distant speech) or failed: one speaker, never no text
            turns = [(Turn(0, len(audio) / RATE, 0), 1)]
        results = await transcribe_turns(audio, [(t.start, t.end) for t, _ in turns])
        kept = [(t, n, r) for (t, n), r in zip(turns, results) if r.text]
        # Numbered after dropping silent turns, so the names run 1, 2, 3 with no gaps.
        numbers: dict[int, int] = {}
        segments = [
            {"speaker": f"Speaker {numbers.setdefault(n, len(numbers) + 1)}", "start": round(t.start, 2),
             "end": round(t.end, 2), "text": r.text}
            for t, n, r in kept
        ]
        text = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            model_id = next((r.model_id for r in results if r.model_id), s.models.active_id)
            return {
                "text": text, "model": model_id, "duration": len(audio) / RATE,
                "asr_ms": int((time.monotonic() - started) * 1000), "segments": segments,
                "speakers": len(numbers),
            }
        return {"text": text}

    @app.get("/v1/models", dependencies=[Depends(guard)])
    def openai_models() -> dict:
        active = s.models.active_id
        return {"object": "list", "data": [{"id": active, "object": "model", "owned_by": "scribe"}] if active else []}

    # -- sessions -------------------------------------------------------------------------------

    def session_or_404(session_id: str) -> dict:
        session = s.store.get_session(session_id)
        if session is None:
            raise HTTPException(404, "no such session")
        return session

    @app.post("/api/sessions", dependencies=[Depends(guard)], status_code=201)
    def create_session(body: NewSession) -> dict:
        title = body.title or time.strftime("%Y-%m-%d %H:%M ") + body.source
        session = s.store.create_session(body.source, title, body.guild, body.channel)
        touch(session["id"])
        write_open_sessions()
        return session

    @app.get("/api/sessions", dependencies=[Depends(guard)])
    def list_sessions() -> list[dict]:
        return s.store.list_sessions()

    @app.get("/api/sessions/{session_id}", dependencies=[Depends(guard)])
    def get_session(session_id: str) -> dict:
        return {**session_or_404(session_id), "utterances": s.store.utterances(session_id)}

    @app.post("/api/sessions/{session_id}/end", dependencies=[Depends(guard)])
    async def end_session(session_id: str) -> dict:  # async: publish on the event loop
        session_or_404(session_id)
        s.store.end_session(session_id)
        last_seen.pop(session_id, None)
        forget(session_id)
        write_open_sessions()
        session = s.store.get_session(session_id)
        hub.publish(session_id, "end", session)
        return session

    @app.post("/api/sessions/{session_id}/heartbeat", dependencies=[Depends(guard)])
    def heartbeat(session_id: str) -> dict:
        """A live client keeps a quiet call open (see `browser_idle_s`); 409 once it has ended."""
        if session_or_404(session_id)["ended_at"] is not None:
            raise HTTPException(409, "session has ended")
        touch(session_id)
        return {"ok": True}

    @app.delete("/api/sessions/{session_id}", dependencies=[Depends(guard)])
    def delete_session(session_id: str) -> dict:
        session_or_404(session_id)
        s.store.delete_session(session_id)
        last_seen.pop(session_id, None)
        forget(session_id)
        write_open_sessions()
        if s.audio_dir:
            for path in (s.audio_dir / session_id).glob("*.wav"):
                path.unlink()
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/utterances", dependencies=[Depends(guard)], status_code=201)
    async def add_utterance(
        session_id: str,
        file: UploadFile = File(...),
        speaker_id: str = Form(...),
        speaker_name: str = Form(...),
        started_at: float = Form(...),
        client_key: str | None = Form(None),
        diarize: bool = Form(False),
    ) -> dict:
        """`client_key` makes a retry safe: a key already stored returns the stored line.

        `diarize` marks a channel that mixes several voices (a browser call's tab): it is split
        into speakers `<speaker_id>#1`, `#2`, ... named "Speaker 1", ... until renamed, and the
        reply lists every line it became under `lines` (docs/spec.md, 3.8). Until diarization is
        ready, such a channel is transcribed as one speaker.
        """
        session = session_or_404(session_id)
        if session["ended_at"] is not None:
            raise HTTPException(409, "session has ended")
        touch(session_id)
        diarizer = s.diarization.ready() if diarize else None
        if diarizer and client_key and (existing := s.store.utterances_by_key_prefix(session_id, client_key)):
            return {"text": " ".join(line["text"] for line in existing), "lines": existing}
        if existing := s.store.utterance_by_key(session_id, client_key):
            return existing
        data = await read_upload(file)
        audio = await decode_upload(data)
        if diarizer:
            reply = await add_diarized(session_id, data, audio, speaker_id, started_at, client_key, diarizer)
            if reply is not None:
                return reply
        result = await on_worker(s.worker, lambda: run_asr(audio))
        if not result.text:
            return {"skipped": True, "reason": "no speech", "duration_s": result.duration_s}
        name = s.store.speaker_names(session_id).get(speaker_id, speaker_name)
        line, new = s.store.add_utterance(
            session_id, speaker_id, name, started_at, int(result.duration_s * 1000),
            result.text, result.model_id, result.asr_ms, client_key,
        )
        if new:
            keep_audio(session_id, line, data)
            hub.publish(session_id, "line", line)
        return line

    def keep_audio(session_id: str, line: dict, data: bytes) -> None:
        if s.audio_dir:
            folder = s.audio_dir / session_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{line['id']}.wav").write_bytes(data)

    async def diarize_once(key, session_id, audio, speaker_id, diarizer) -> list:
        """The utterance's (turn, speaker) pairs, diarized once per client_key. [] means "as one
        speaker". A retry, whether after a busy GPU queue or a client timeout while the first
        attempt is still running, waits for and reuses the first attempt's pairs: the book learns
        the audio once and the retry's lines get the same keys."""
        if key and (first := pending.get(key)) is not None:
            pairs = await asyncio.shield(first)
            if pairs is not None:
                return pairs
            # The first attempt hit a full queue and gave up before diarizing: go ahead.
        future = asyncio.get_running_loop().create_future() if key else None
        if key:
            pending[key] = future
            while len(pending) > 256:  # retries that never came
                pending.pop(next(iter(pending)))
        book = book_for(session_id, speaker_id)
        pairs = None  # None until diarized: a waiting retry then diarizes for itself
        try:
            try:
                pairs = await on_worker(cpu("diarize_worker"), lambda: diarizer.turns(audio, book))
            except HTTPException:  # a full queue: 503, which the client retries
                raise
            except Exception:
                log.exception("diarization failed; transcribing the utterance as one speaker")
                pairs = []
            if pairs:
                # Save the voices now, so the stored book always matches the one in memory.
                for speaker in {id(sp): sp for _, sp in pairs}.values():
                    s.store.save_voice(
                        session_id, f"{speaker_id}#{speaker.number}",
                        speaker.centroid.astype(np.float32).tobytes(), speaker.weight,
                    )
            return pairs
        finally:
            # Always resolved, even if this request is cancelled, so a waiting retry never hangs.
            if key:
                future.set_result(pairs)
                if not pairs:
                    pending.pop(key, None)

    async def add_diarized(session_id, data, audio, speaker_id, started_at, client_key, diarizer) -> dict | None:
        """Split one utterance into speaker turns and store a line per turn.

        None means "transcribe it as one speaker instead": no turns were found, or diarization
        failed. A failure here must never lose the speech.
        """
        key = (session_id, client_key) if client_key else None
        pairs = await diarize_once(key, session_id, audio, speaker_id, diarizer)
        if not pairs:
            return None
        results = await transcribe_turns(audio, [(turn.start, turn.end) for turn, _ in pairs])
        names = s.store.speaker_names(session_id)
        lines, kept = [], False
        for index, ((turn, speaker), result) in enumerate(zip(pairs, results)):
            if not result.text:
                continue
            line_speaker = f"{speaker_id}#{speaker.number}"
            line, new = s.store.add_utterance(
                session_id, line_speaker, names.get(line_speaker, f"Speaker {speaker.number}"),
                started_at + turn.start, int(turn.duration * 1000), result.text, result.model_id, result.asr_ms,
                f"{client_key}#{index}" if client_key else None,
            )
            if new:
                if not kept:
                    keep_audio(session_id, line, data)
                    kept = True
                hub.publish(session_id, "line", line)
            lines.append(line)
        if key:
            pending.pop(key, None)
        if not lines:
            return {"skipped": True, "reason": "no speech", "duration_s": len(audio) / RATE}
        return {"text": " ".join(line["text"] for line in lines), "lines": lines}

    @app.post("/api/sessions/{session_id}/speakers/{speaker_id}", dependencies=[Depends(guard)])
    async def rename_speaker(session_id: str, speaker_id: str, body: Rename) -> dict:  # async: publish
        """Name a speaker for this session: their lines so far and every later one."""
        session_or_404(session_id)
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "name is empty")
        if len(name) > 80:
            raise HTTPException(400, "name is longer than 80 characters")
        changed = s.store.rename_speaker(session_id, speaker_id, name)
        hub.publish(session_id, "speaker", {"speaker_id": speaker_id, "name": name})
        return {"speaker_id": speaker_id, "name": name, "lines": changed}

    @app.post("/api/sessions/{session_id}/markers", dependencies=[Depends(guard)], status_code=201)
    async def add_marker(session_id: str, body: Marker) -> dict:  # async: publish on the event loop
        """A line with no audio behind it, e.g. the bot recording audio it could not deliver."""
        session_or_404(session_id)
        touch(session_id)
        line, _ = s.store.add_utterance(session_id, body.speaker_id, body.speaker_name, body.started_at, 0, body.text, None, None)
        hub.publish(session_id, "line", line)
        return line

    @app.get("/api/sessions/{session_id}/events", dependencies=[Depends(guard)])
    async def session_events(session_id: str, request: Request) -> StreamingResponse:
        session_or_404(session_id)
        queue = hub.subscribe(session_id)

        async def stream():
            try:
                yield "retry: 2000\n\n"
                while not await request.is_disconnected():
                    try:
                        event, data = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
            finally:
                hub.unsubscribe(session_id, queue)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/sessions/{session_id}/export", dependencies=[Depends(guard)])
    def export_session(session_id: str, format: str = "md") -> Response:
        session = session_or_404(session_id)
        lines = s.store.utterances(session_id)
        try:
            body, media_type, ext = exports.render(format, session, lines)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        filename = f"scribe-{session_id}.{ext}"
        return Response(body, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    # -- opt-outs -------------------------------------------------------------------------------

    @app.get("/api/optouts/{source}", dependencies=[Depends(guard)])
    def list_optouts(source: str) -> list[str]:
        return s.store.optouts(source)

    @app.post("/api/optouts", dependencies=[Depends(guard)])
    def set_optout(body: OptOut) -> dict:
        s.store.set_optout(body.source, body.user_id, body.opted_out)
        return {"ok": True}

    # -- GUI ------------------------------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
