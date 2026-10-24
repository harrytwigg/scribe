"""HTTP API and GUI (docs/spec.md, 3.3 and 3.4)."""

import asyncio
import contextlib
import io
import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from . import exports, podcast
from .audio import RATE, AudioError, Vad, decode, speech_chunks, speech_spans
from .auth import COOKIE, Auth
from .diarize import DiarizerSlot, Speaker, SpeakerBook, Turn
from .models import Conflict, ModelManager, NoActiveModel, NotFound
from .store import Store
from .worker import GpuWorker, QueueFull

log = logging.getLogger("scribe.app")
STATIC = Path(__file__).parent / "static"
MAX_SPEAKERS = 20  # the most speakers a caller may ask for
# The Chrome extension, next to server/ in a checkout and in the deployed copy (deploy/install.sh).
EXTENSION = Path(__file__).resolve().parents[2] / "extension"


def extension_zip(folder: Path) -> bytes | None:
    """The extension as a zip to unpack and load, or None if this install has no extension folder.
    Only what Chrome runs: the manifest and src/, not the tests."""
    if not (folder / "manifest.json").is_file():
        return None
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zip_:
        for path in sorted([folder / "manifest.json", *(folder / "src").rglob("*")]):
            if path.is_file():
                zip_.write(path, path.relative_to(folder).as_posix())
    return out.getvalue()


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
    # Fetches a pasted podcast link for the GUI; None makes one that refuses non-public addresses.
    fetcher: podcast.Fetcher | None = None
    # Transcriptions running at once (each holds a decoded recording in memory); the rest queue.
    max_jobs: int = 2
    # Transcriptions waiting (beyond the running ones) before a new one is refused with 429.
    max_queued: int = 50
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


@dataclass
class Job:
    """A transcription in the background (an upload or a pasted link)."""

    id: str
    title: str
    status: str = "queued"  # queued, fetching, decoding, transcribing, done, error, cancelled
    stage_progress: float = 0.0  # of the transcribing, 0 to 1
    fetches: bool = False  # a link: it downloads the audio itself (an upload arrives complete)
    fetch_done: int = 0  # bytes downloaded so far, and how many there will be (0: not told)
    fetch_total: int = 0
    step: str | None = None  # what a stage with no measure is doing ("separating speakers")
    error: str | None = None
    session_id: str | None = None
    episode: dict | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None
    work: Path | None = None  # holds the audio file until it is decoded
    cancel: threading.Event = field(default_factory=threading.Event)  # for the thread that fetches

    @property
    def active(self) -> bool:
        return self.status in ("queued", "fetching", "decoding", "transcribing")

    @property
    def running(self) -> bool:  # has a slot: not waiting its turn
        return self.status in ("fetching", "decoding", "transcribing")

    @property
    def progress(self) -> float:
        """Overall, 0 to 1: the download (a link only), decoding, then transcribing, weighted by
        about how long each takes, so one bar covers the whole job."""
        fetch, decode = (0.15, 0.05) if self.fetches else (0.0, 0.05)
        if self.status == "done":
            return 1.0
        if self.status == "fetching":
            return fetch * (min(self.fetch_done / self.fetch_total, 1.0) if self.fetch_total else 0.0)
        if self.status == "decoding":
            return fetch
        if self.status == "transcribing":
            return fetch + decode + (1 - fetch - decode) * self.stage_progress
        return 0.0

    def public(self, position: int | None = None) -> dict:
        return {
            "id": self.id, "title": self.title, "status": self.status, "progress": round(self.progress, 3),
            "stage_progress": round(self.stage_progress, 3), "step": self.step,
            "download": {"done": self.fetch_done, "total": self.fetch_total} if self.fetches else None,
            "queue_position": position, "error": self.error, "session_id": self.session_id,
            "episode": self.episode, "result": self.result,
        }


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


def one_line(text: str) -> str:
    """A name or title as one plain line: runs of whitespace (newlines too) become one space and
    control characters go, so it cannot break the Markdown export or the page."""
    return " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())


class Rename(BaseModel):
    name: str


class SessionUpdate(BaseModel):
    title: str


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
        if close := getattr(s.diarization.value, "close", None):
            close()  # the speaker separation processes

    app = FastAPI(title="scribe", docs_url=None, redoc_url=None, lifespan=lifespan)
    hub = Hub()
    s = services
    # When each open session last showed signs of life. A session from before a restart counts
    # from the restart, so it gets the full grace period.
    last_seen: dict[str, float] = {}
    booted = time.time()
    jobs: dict[str, Job] = {}

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
            # A running transcription keeps the host awake like an open call does.
            tmp.write_text(f"{s.store.count_open_sessions() + sum(job.active for job in jobs.values())}\n")
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

    async def decode_upload(data: bytes | Path, model: str | None = None) -> np.ndarray:
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
            "transcriptions": {
                "running": sum(job.running for job in jobs.values()),
                "queued": sum(job.status == "queued" for job in jobs.values()),
                "max": s.max_jobs,
            },
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

    async def diarized_segments(audio: np.ndarray, diarizer, speakers: int | None = None) -> dict:
        """Who spoke when, with the text of each turn: `segments`, `text`, `model`, `duration`,
        `asr_ms` and `speakers`. Raises HTTPException when the GPU queue is full."""
        started = time.monotonic()
        cancel = threading.Event()
        try:
            turns = await on_worker(
                cpu("upload_worker"),
                lambda: diarizer.whole(audio, speakers, cancel),
            )
        except asyncio.CancelledError:
            cancel.set()  # a cancelled job stops its work in the separator, not only waiting for it
            raise
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
        return {
            "text": "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments),
            "model": next((r.model_id for r in results if r.model_id), s.models.active_id),
            "duration": len(audio) / RATE, "asr_ms": int((time.monotonic() - started) * 1000),
            "segments": segments, "speakers": len(numbers),
        }

    async def diarized_file(data: bytes, model: str | None, response_format: str, diarizer):
        out = await diarized_segments(await decode_upload(data, model), diarizer)
        if response_format == "text":
            return PlainTextResponse(out["text"])
        if response_format == "verbose_json":
            return out
        return {"text": out["text"]}

    @app.get("/v1/models", dependencies=[Depends(guard)])
    def openai_models() -> dict:
        active = s.models.active_id
        return {"object": "list", "data": [{"id": active, "object": "model", "owned_by": "scribe"}] if active else []}

    # -- background transcriptions: an uploaded recording or a pasted podcast link ----------------
    # The result is stored as a session (source "upload") when it is done, so History, its exports
    # and the speaker names work as for a call. The job itself lives in memory for an hour so a
    # client can poll it; a restart forgets running jobs.

    JOB_KEEP_S = 3600
    VAD_WINDOW_S = 600  # speech detection runs over this much audio per GPU-queue turn

    def prune_jobs() -> None:
        now = time.time()
        for old in [j for j in jobs.values() if not j.active and now - (j.finished or now) > JOB_KEEP_S]:
            jobs.pop(old.id, None)

    def new_job(title: str, fetches: bool = False) -> Job:
        """A queued job, or 429 when `max_queued` are already waiting."""
        prune_jobs()
        if sum(job.active for job in jobs.values()) >= s.max_jobs + s.max_queued:
            raise HTTPException(
                429, f"{s.max_queued} transcriptions are already waiting; try again when some finish",
                headers={"Retry-After": "60"},
            )
        job = Job(id=uuid.uuid4().hex[:12], title=one_line(title)[:120] or "Recording", fetches=fetches)
        job.work = Path(tempfile.mkdtemp(prefix="scribe-job-"))
        jobs[job.id] = job
        write_open_sessions()
        return job

    def queue_position(job: Job) -> int | None:
        """1 for the next job to start; None once it has a slot."""
        if job.status != "queued":
            return None
        waiting = [j.id for j in jobs.values() if j.status == "queued"]  # in the order they came
        return waiting.index(job.id) + 1

    async def wait_for_slot(job: Job) -> None:
        """First come, first served, at most `max_jobs` at once."""
        while True:
            first = next((j for j in jobs.values() if j.status == "queued"), None)
            if first is job and sum(j.running for j in jobs.values()) < s.max_jobs:
                job.status = "fetching"
                return
            await asyncio.sleep(0.25)

    def discard_job(job: Job) -> None:
        """Forget a job that never started (its request was refused)."""
        jobs.pop(job.id, None)
        shutil.rmtree(job.work, ignore_errors=True)
        write_open_sessions()

    def job_or_404(job_id: str) -> Job:
        prune_jobs()
        if job_id not in jobs:
            raise HTTPException(404, "no such transcription (finished ones are forgotten after an hour)")
        return jobs[job_id]

    async def patient(worker: GpuWorker, fn, wait_s: float = 900):
        """Run on a worker, waiting for room in its queue: a background job yields to live calls
        rather than failing when they fill it."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                return await worker.run(fn)
            except QueueFull:
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(0.5)

    def asr_chunk(chunk: np.ndarray) -> tuple[str, str]:
        model = s.models.acquire()
        try:
            return model.engine.transcribe(chunk), model.id
        finally:
            s.models.release(model)

    async def plain_segments(audio: np.ndarray, job: Job) -> dict:
        """One speaker: each stretch of speech (up to 25 s) with its start and end. Every chunk is
        its own turn at the GPU queue, so a long recording never holds up a live call."""
        started = time.monotonic()
        window = VAD_WINDOW_S * RATE
        spans: list[tuple[int, int]] = []
        for offset in range(0, len(audio), window):
            part = audio[offset : offset + window]
            found = await patient(s.worker, lambda part=part: speech_spans(part, s.vad))
            spans += [(a + offset, b + offset) for a, b in found]
            job.stage_progress = 0.1 * min(offset + window, len(audio)) / len(audio)  # finding the speech: the first tenth
        segments, model_id = [], s.models.active_id
        for index, (a, b) in enumerate(spans, 1):
            text, model_id = await patient(s.worker, lambda a=a, b=b: asr_chunk(audio[a:b]))
            if text.strip():
                segments.append({"speaker": "Speaker 1", "start": round(a / RATE, 2), "end": round(b / RATE, 2), "text": text.strip()})
            job.stage_progress = 0.1 + 0.9 * index / len(spans)
        return {
            "text": " ".join(seg["text"] for seg in segments), "model": model_id, "duration": len(audio) / RATE,
            "asr_ms": int((time.monotonic() - started) * 1000), "segments": segments, "speakers": 1 if segments else 0,
        }

    def store_result(job: Job) -> str:
        """The finished job as an ended session; returns its id."""
        result = job.result
        session = s.store.create_session("upload", job.title)
        names = {}
        for index, seg in enumerate(result["segments"]):
            key = names.setdefault(seg["speaker"], f"s{len(names) + 1}")
            s.store.add_utterance(
                session["id"], key, seg["speaker"], session["started_at"] + seg["start"],
                int(max(seg["end"] - seg["start"], 0) * 1000), seg["text"], result["model"], None, f"seg{index}",
            )
        s.store.end_session(session["id"])
        return session["id"]

    async def run_job(job: Job, source, diarize: bool, speakers: int | None = None) -> None:
        """`source` is an awaitable giving the path of the audio file in `job.work` (it may fetch it first)."""
        try:
            await wait_for_slot(job)
            path = await source(job)
            job.status = "decoding"
            audio = await decode_upload(path)
            shutil.rmtree(job.work, ignore_errors=True)  # no audio is kept
            job.status = "transcribing"
            diarizer = s.diarization.ready() if diarize else None
            job.step = "separating speakers" if diarizer else None
            for attempt in range(60):  # a full queue is not the job's failure: wait for room
                try:
                    job.result = await (diarized_segments(audio, diarizer, speakers) if diarizer else plain_segments(audio, job))
                    break
                except HTTPException as busy:
                    if busy.status_code != 503 or "Retry-After" not in (busy.headers or {}) or attempt == 59:
                        raise
                    await asyncio.sleep(2)
            if diarize and not diarizer:
                job.result["note"] = "Speaker separation is not ready yet: transcribed as one speaker."
            job.session_id = await asyncio.to_thread(store_result, job)
            job.status = "done"
            job.stage_progress = 1.0
        except asyncio.CancelledError:
            job.status, job.error = "cancelled", "cancelled"
            raise
        except HTTPException as error:
            job.status, job.error = "error", str(error.detail)
        except (podcast.PodcastError, AudioError, NoActiveModel, QueueFull) as error:
            job.status, job.error = "error", str(error)
        except Exception as error:
            log.exception("transcription %s failed", job.id)
            job.status, job.error = "error", f"transcription failed: {error}"
        finally:
            shutil.rmtree(job.work, ignore_errors=True)
            job.finished = time.time()
            job.task = None
            write_open_sessions()

    def start_job(job: Job, source, diarize: bool, speakers: int | None = None) -> dict:
        job.task = asyncio.create_task(run_job(job, source, diarize, speakers))
        return job.public(queue_position(job))

    def require_model() -> None:
        if s.models.active_id is None:
            raise HTTPException(503, "no model is active; pick one in the GUI")

    @app.post("/api/transcriptions", dependencies=[Depends(guard)], status_code=202)
    async def start_upload(
        request: Request, filename: str = "", title: str = "", diarize: bool = False, speakers: int | None = Query(None, ge=1, le=MAX_SPEAKERS)
    ) -> dict:
        """The request body is the audio file itself (any format `decode` reads). Returns the job
        to poll at GET /api/transcriptions/{id}. `speakers`, the number of people in it if known,
        implies `diarize` and makes the speaker separation find exactly that many."""
        require_model()
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > s.max_upload_bytes:
            raise HTTPException(413, f"upload larger than {s.max_upload_bytes // (1024 * 1024)} MB")
        job = new_job(title or Path(filename).stem or "Recording")
        path = job.work / "upload"
        try:
            size = 0
            with path.open("wb") as out:  # to disk: the audio is only held decoded, once
                async for part in request.stream():
                    size += len(part)
                    if size > s.max_upload_bytes:
                        raise HTTPException(413, f"upload larger than {s.max_upload_bytes // (1024 * 1024)} MB")
                    out.write(part)
            if not size:
                raise HTTPException(400, "the request body is empty: send the audio file as the body")
        except BaseException:
            discard_job(job)
            raise

        async def source(_: Job) -> Path:
            return path

        return start_job(job, source, diarize or speakers is not None, speakers)

    class LinkJob(BaseModel):
        url: str
        diarize: bool = False
        speakers: int | None = Field(None, ge=1, le=MAX_SPEAKERS)  # people in it, if known; implies diarize
        episode: int = 1  # of a show or feed link: 1 is the newest

    @app.post("/api/transcriptions/link", dependencies=[Depends(guard)], status_code=202)
    async def start_link(body: LinkJob) -> dict:
        """Fetch the podcast episode a link names (Spotify, Apple Podcasts, an RSS feed or an audio
        URL; docs/cli.md) and transcribe it. Returns the job to poll."""
        require_model()
        fetcher = s.fetcher or podcast.Fetcher(public_only=True)
        job = new_job("Podcast", fetches=True)

        async def source(job: Job) -> Path:
            def aborted(done: int = 0, total: int | None = None) -> None:
                if job.cancel.is_set():  # called as the download streams: stops it
                    raise podcast.PodcastError("cancelled")
                job.fetch_done, job.fetch_total = done, total or 0

            def fetch() -> Path:
                episode = podcast.resolve(fetcher, body.url.strip(), body.episode)
                aborted()
                job.episode = episode.to_json()
                job.title = one_line(f"{episode.show}: {episode.title}" if episode.show else episode.title)[:120]
                return podcast.download(fetcher, episode.audio_url, job.work, "episode", aborted, limit=s.max_upload_bytes)

            return await asyncio.to_thread(fetch)

        return start_job(job, source, body.diarize or body.speakers is not None, body.speakers)

    @app.get("/api/transcriptions/{job_id}", dependencies=[Depends(guard)])
    def get_job(job_id: str) -> dict:
        job = job_or_404(job_id)
        return job.public(queue_position(job))

    @app.delete("/api/transcriptions/{job_id}", dependencies=[Depends(guard)])
    async def cancel_job(job_id: str) -> dict:  # async: runs on the event loop with the jobs it stops
        job = job_or_404(job_id)
        if job.active:
            job.cancel.set()  # the fetching thread notices at its next chunk
            if job.task:
                job.task.cancel()
            job.status, job.error, job.finished = "cancelled", "cancelled", time.time()
            shutil.rmtree(job.work, ignore_errors=True)
            write_open_sessions()
        return {"ok": True}

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
        """The call, its lines, and `speakers`: the names given to speakers in this call."""
        return {
            **session_or_404(session_id),
            "utterances": s.store.utterances(session_id),
            "speakers": s.store.speaker_names(session_id),
        }

    @app.patch("/api/sessions/{session_id}", dependencies=[Depends(guard)])
    async def update_session(session_id: str, body: SessionUpdate) -> dict:  # async: publish
        """Rename a call (its title in the GUI, exports and the Live tab)."""
        session_or_404(session_id)
        title = one_line(body.title)
        if not title:
            raise HTTPException(400, "title is empty")
        if len(title) > 120:
            raise HTTPException(400, "title is longer than 120 characters")
        s.store.rename_session(session_id, title)
        hub.publish(session_id, "title", {"title": title})
        return s.store.get_session(session_id)

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
        name = one_line(body.name)
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

    # -- Chrome extension -----------------------------------------------------------------------

    @app.get("/api/extension.zip", dependencies=[Depends(guard)])
    def download_extension() -> Response:
        """The Chrome extension (extension/), to unzip and load unpacked on any machine."""
        body = extension_zip(EXTENSION)
        if body is None:
            raise HTTPException(404, "this install has no extension folder")
        return Response(
            body, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="scribe-extension.zip"'}
        )

    # -- GUI ------------------------------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
