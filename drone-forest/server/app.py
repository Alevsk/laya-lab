"""The decision microservice: FastAPI over HTTP for tools, WebSocket for the game.

Threading model. Engines are synchronous (Laya is a blocking GPU forward pass), so every
`decide()` runs in the default threadpool via `asyncio.to_thread`; the event loop stays free
for other connections. Per connection at most one decision is in flight: a frame that arrives
while one is being computed replaces any frame still waiting, because the game only ever
wants an answer to its *latest* frame. Decision ids increase per connection in send order.

Engines are looked up through `server.engines` (name -> factory) and cached per process, so
switching engines never reloads a model. `server.app` knows no concrete engine.

Security: none. This is a local development tool bound to loopback with CORS open to any
origin so the Vite dev server on :5173 can reach it. Do not expose it.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from . import engines
from .config import Settings, get_settings
from .engines.base import DecisionEngine
from .schemas import (
    Decision,
    DecisionMessage,
    ErrorMessage,
    EventMessage,
    FrameMessage,
    GameEvent,
    InfoMessage,
    SensorFrame,
    SetEngineMessage,
)
from .telemetry import Telemetry

log = logging.getLogger("drone-forest")

ClientMessage = Annotated[FrameMessage | EventMessage | SetEngineMessage, Field(discriminator="type")]
_client_message: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


class EngineCache:
    """One instance per engine name for the life of the process.

    `instance()` only constructs (cheap: engines defer model loading to `warmup()`), so listing
    and describing engines never loads a model. `ready()` warms once and is what every decision
    path goes through; switching engines afterwards is a dictionary lookup.
    """

    def __init__(self) -> None:
        self._instances: dict[str, DecisionEngine] = {}
        self._warm: set[str] = set()
        self._lock = threading.Lock()

    def instance(self, name: str) -> DecisionEngine:
        with self._lock:
            engine = self._instances.get(name)
            if engine is None:
                engine = self._instances[name] = engines.create(name)  # KeyError for unknown names
            return engine

    def ready(self, name: str) -> DecisionEngine:
        engine = self.instance(name)
        with self._lock:
            if name not in self._warm:
                engine.warmup()
                self._warm.add(name)
        return engine

    def loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._warm)


ENGINE_CACHE = EngineCache()


class DecideRequest(BaseModel):
    frame: SensorFrame
    engine: str | None = None


class EventRequest(BaseModel):
    event: GameEvent


class OkResponse(BaseModel):
    ok: Literal[True] = True


class HealthResponse(BaseModel):
    ok: bool
    engine: str
    engines: list[str]
    loaded: list[str]
    laya_import_error: str | None
    telemetry_file: str


class EngineInfo(BaseModel):
    name: str
    describe: dict


def _resolve_default_engine(settings: Settings, names: list[str]) -> str:
    if settings.default_engine in names:
        return settings.default_engine
    fallback = "heuristic" if "heuristic" in names else names[0]
    log.warning("DEFAULT_ENGINE=%r is not available (%s); using %r", settings.default_engine, names, fallback)
    return fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    names = engines.load_builtin_engines()
    for mod, err in engines.IMPORT_ERRORS.items():
        log.warning("engine module %s did not load: %s", mod, err)
    if not names:
        raise RuntimeError("no decision engines could be loaded")
    cache = ENGINE_CACHE
    default = _resolve_default_engine(settings, names)
    await asyncio.to_thread(cache.ready, default)
    telemetry = Telemetry(settings.telemetry_dir)
    log.info("engines %s; default %r; telemetry -> %s", names, default, telemetry.path)

    app.state.settings = settings
    app.state.engine_cache = cache
    app.state.default_engine = default
    app.state.telemetry = telemetry
    app.state.http_decision_id = 0
    try:
        yield
    finally:
        telemetry.close()


app = FastAPI(title="drone-forest decision service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---- HTTP -------------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cache: EngineCache = app.state.engine_cache
    return HealthResponse(
        ok=True,
        engine=app.state.default_engine,
        engines=engines.available(),
        loaded=cache.loaded(),
        laya_import_error=engines.IMPORT_ERRORS.get("laya_engine"),
        telemetry_file=str(app.state.telemetry.path),
    )


@app.get("/engines", response_model=list[EngineInfo])
async def list_engines() -> list[EngineInfo]:
    cache: EngineCache = app.state.engine_cache
    out: list[EngineInfo] = []
    for name in engines.available():
        out.append(EngineInfo(name=name, describe=cache.instance(name).describe()))
    return out


@app.get("/stats")
async def stats() -> dict:
    return app.state.telemetry.stats()


@app.post("/decide", response_model=Decision)
async def decide(req: DecideRequest) -> Decision:
    name = req.engine or app.state.default_engine
    if name not in engines.available():
        raise HTTPException(status_code=404, detail=f"unknown engine {name!r}; available: {engines.available()}")
    engine = await asyncio.to_thread(app.state.engine_cache.ready, name)
    app.state.http_decision_id += 1
    decision = await asyncio.to_thread(engine.decide, req.frame, app.state.http_decision_id)
    app.state.telemetry.record_decision(req.frame, decision, name)
    return decision


@app.post("/event", response_model=OkResponse)
async def post_event(req: EventRequest) -> OkResponse:
    app.state.telemetry.record_event(req.event, app.state.default_engine)
    return OkResponse()


# ---- WebSocket --------------------------------------------------------------------------------


class Connection:
    """State for one game client: its engine, its decision counter, and the one-slot frame queue."""

    def __init__(self, ws: WebSocket, engine_name: str) -> None:
        self.ws = ws
        self.engine_name = engine_name
        self.decision_id = 0
        self.pending: SensorFrame | None = None
        self.worker: asyncio.Task[None] | None = None
        self.dropped = 0
        self._send_lock = asyncio.Lock()

    async def send(self, message: BaseModel) -> None:
        async with self._send_lock:
            await self.ws.send_text(message.model_dump_json())

    def submit(self, frame: SensorFrame) -> None:
        """Queue a frame; if one is already waiting it is replaced (the game wants the latest)."""
        if self.pending is not None:
            self.dropped += 1
        self.pending = frame
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        cache: EngineCache = app.state.engine_cache
        telemetry: Telemetry = app.state.telemetry
        while self.pending is not None:
            frame, self.pending = self.pending, None
            name = self.engine_name
            self.decision_id += 1
            try:
                engine = await asyncio.to_thread(cache.ready, name)
                decision = await asyncio.to_thread(engine.decide, frame, self.decision_id)
            except Exception as e:  # an engine that raises must not take the socket down
                log.exception("engine %s failed on frame %s", name, frame.frame_id)
                await self.send(ErrorMessage(message=f"engine {name!r} failed: {type(e).__name__}: {e}"))
                continue
            telemetry.record_decision(frame, decision, name)
            await self.send(DecisionMessage(decision=decision))

    async def close(self) -> None:
        if self.worker is not None and not self.worker.done():
            self.worker.cancel()
            try:
                await self.worker
            except (asyncio.CancelledError, Exception):
                pass


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, engine: str | None = None) -> None:
    await ws.accept()
    names = engines.available()
    requested = engine or app.state.default_engine
    conn = Connection(ws, requested if requested in names else app.state.default_engine)
    telemetry: Telemetry = app.state.telemetry
    try:
        if requested not in names:
            await conn.send(ErrorMessage(message=f"unknown engine {requested!r}; using {conn.engine_name!r}"))
        # Warm the engine before announcing it: `info` means "ready", so the game never flies
        # blind through a model load (a 2-3 s Laya load would otherwise cost it ~30 m of forest).
        try:
            await asyncio.to_thread(app.state.engine_cache.ready, conn.engine_name)
        except Exception as e:
            log.exception("engine %s failed to load", conn.engine_name)
            await conn.send(ErrorMessage(message=f"engine {conn.engine_name!r} failed to load: {type(e).__name__}: {e}"))
            conn.engine_name = app.state.default_engine
        await conn.send(InfoMessage(engine=conn.engine_name, engines=names))
        while True:
            raw = await ws.receive_text()
            try:
                msg = _client_message.validate_json(raw)
            except ValidationError as e:
                await conn.send(ErrorMessage(message=f"bad message: {e.errors()[0]['msg']} at {e.errors()[0]['loc']}"))
                continue
            if isinstance(msg, FrameMessage):
                conn.submit(msg.frame)
            elif isinstance(msg, EventMessage):
                telemetry.record_event(msg.event, conn.engine_name)
            else:
                await _set_engine(conn, msg)
    except WebSocketDisconnect:
        pass
    finally:
        await conn.close()
        if conn.dropped:
            log.info("connection closed; %d stale frames dropped", conn.dropped)


async def _set_engine(conn: Connection, msg: SetEngineMessage) -> None:
    names = engines.available()
    if msg.engine not in names:
        await conn.send(ErrorMessage(message=f"unknown engine {msg.engine!r}; available: {names}"))
        return
    cache: EngineCache = app.state.engine_cache
    try:
        await asyncio.to_thread(cache.ready, msg.engine)  # warm it before switching so no frame stalls
    except Exception as e:
        log.exception("engine %s failed to load", msg.engine)
        await conn.send(ErrorMessage(message=f"engine {msg.engine!r} failed to load: {type(e).__name__}: {e}"))
        return
    conn.engine_name = msg.engine
    await conn.send(InfoMessage(engine=conn.engine_name, engines=names))
