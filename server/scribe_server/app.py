"""HTTP API and GUI (docs/spec.md, 3.3 and 3.4)."""

import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from . import exports
from .audio import RATE, AudioError, Vad, decode, speech_chunks
from .auth import COOKIE, Auth
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

    async def transcribe(data: bytes, model: str | None = None) -> Transcript:
        if model and model != s.models.active_id:
            raise HTTPException(
                409, f"model {model!r} is not active; the active model is {s.models.active_id!r} (pick one in the GUI)"
            )
        if s.models.active_id is None:
            raise HTTPException(503, "no model is active; pick one in the GUI")
        try:
            # Off the event loop: decoding a long upload must not stall live streams.
            audio = await asyncio.to_thread(decode, data)
        except AudioError as error:
            raise HTTPException(400, str(error)) from error
        try:
            return await s.worker.run(lambda: run_asr(audio))
        except QueueFull as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error
        except NoActiveModel as error:
            raise HTTPException(503, str(error)) from error

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
    ):
        result = await transcribe(await read_upload(file), model)
        if response_format == "text":
            return PlainTextResponse(result.text)
        if response_format == "verbose_json":
            return {"text": result.text, "model": result.model_id, "duration": result.duration_s, "asr_ms": result.asr_ms}
        return {"text": result.text}

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
    ) -> dict:
        """`client_key` makes a retry safe: a key already stored returns the stored line."""
        session = session_or_404(session_id)
        if session["ended_at"] is not None:
            raise HTTPException(409, "session has ended")
        touch(session_id)
        if existing := s.store.utterance_by_key(session_id, client_key):
            return existing
        data = await read_upload(file)
        result = await transcribe(data)
        if not result.text:
            return {"skipped": True, "reason": "no speech", "duration_s": result.duration_s}
        line, new = s.store.add_utterance(
            session_id, speaker_id, speaker_name, started_at, int(result.duration_s * 1000),
            result.text, result.model_id, result.asr_ms, client_key,
        )
        if new:
            if s.audio_dir:
                folder = s.audio_dir / session_id
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{line['id']}.wav").write_bytes(data)
            hub.publish(session_id, "line", line)
        return line

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
