/**
 * `RemoteEngineSource`: a DecisionSource over a WebSocket to the Python service, speaking the
 * envelope messages in `server/schemas.py` (frame / event / set_engine / decision / info /
 * error).
 *
 * Latency policy, because it is what makes engine comparisons fair: exactly one frame is in
 * flight at a time. If the game asks while one is pending, the request is answered at once
 * with the last decision and nothing is sent — a slow engine loses ticks, it never builds a
 * queue on the server. Round trip is measured here (send → decision) and reported separately
 * from the engine's own `latency_ms`, so network cost and think time can be told apart.
 * Reconnects with capped exponential backoff; events are buffered briefly while disconnected
 * so score events survive a blip.
 */
import type { Decision, DecisionSource, GameEvent, SensorFrame } from '../core/types';
import { LatencyWindow, RateMeter, nowMs } from './stats';
import type { ExtendedDecisionStats } from './stats';

export interface RemoteOptions {
  /** give up on an in-flight decision after this long and answer with the last one */
  timeoutMs?: number;
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
  /** how long `connect()` waits for the first socket before resolving anyway */
  connectWaitMs?: number;
  eventBufferSize?: number;
  /** injectable for tests; defaults to the page's WebSocket */
  WebSocketImpl?: typeof WebSocket;
  onInfo?: (engine: string, engines: string[]) => void;
  onError?: (message: string) => void;
}

interface ServerInfo { type: 'info'; engine: string; engines: string[]; protocol?: number }
interface ServerDecision { type: 'decision'; decision: Decision }
interface ServerError { type: 'error'; message: string }
type ServerMessage = ServerInfo | ServerDecision | ServerError;

interface Pending {
  frameId: number;
  sentAt: number;
  resolve: (d: Decision) => void;
  timer: ReturnType<typeof setTimeout>;
}

const FALLBACK_ID = -1;

export class RemoteEngineSource implements DecisionSource {
  private engine: string;
  private readonly url: URL;
  private ws: WebSocket | null = null;
  private pending: Pending | null = null;
  private last: Decision | null = null;
  private disposed = false;
  private attempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly firstOpen: Promise<void>;
  private resolveFirstOpen: () => void = () => {};
  private readonly eventBuffer: GameEvent[] = [];

  private readonly rtt = new LatencyWindow();
  private readonly think = new LatencyWindow();
  private readonly rate = new RateMeter();
  private decisions = 0;
  private skipped = 0;
  private errors = 0;
  private timeouts = 0;
  private reconnects = 0;
  private stale = 0;
  private engines: string[] = [];
  private lastError: string | null = null;

  private readonly timeoutMs: number;
  private readonly baseMs: number;
  private readonly maxMs: number;
  private readonly connectWaitMs: number;
  private readonly bufferSize: number;
  private readonly WS: typeof WebSocket;

  constructor(url: string, engine: string, private readonly opts: RemoteOptions = {}) {
    this.engine = engine;
    this.url = new URL(url);
    this.url.searchParams.set('engine', engine);
    this.timeoutMs = opts.timeoutMs ?? 10_000;
    this.baseMs = opts.reconnectBaseMs ?? 300;
    this.maxMs = opts.reconnectMaxMs ?? 5_000;
    this.connectWaitMs = opts.connectWaitMs ?? 2_000;
    this.bufferSize = opts.eventBufferSize ?? 256;
    this.WS = opts.WebSocketImpl ?? WebSocket;
    this.firstOpen = new Promise<void>((resolve) => {
      this.resolveFirstOpen = resolve;
    });
  }

  get name(): string {
    return `remote:${this.engine}`;
  }

  get connected(): boolean {
    return this.ws !== null && this.ws.readyState === this.WS.OPEN;
  }

  /** Resolves when the first socket opens, or after `connectWaitMs` — reconnection keeps going in the background either way. */
  connect(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    if (!this.ws) this.open();
    const wait = new Promise<void>((resolve) => setTimeout(resolve, this.connectWaitMs));
    return Promise.race([this.firstOpen, wait]);
  }

  decide(frame: SensorFrame): Promise<Decision> {
    if (this.pending || !this.connected) {
      this.skipped++;
      return Promise.resolve(this.fallback(frame, this.pending ? 'engine still thinking' : 'not connected'));
    }
    return new Promise<Decision>((resolve) => {
      const timer = setTimeout(() => this.onTimeout(frame), this.timeoutMs);
      this.pending = { frameId: frame.frame_id, sentAt: nowMs(), resolve, timer };
      this.send({ type: 'frame', frame });
    });
  }

  report(event: GameEvent): void {
    if (this.connected) {
      this.send({ type: 'event', event });
      return;
    }
    if (this.eventBuffer.length >= this.bufferSize) this.eventBuffer.shift();
    this.eventBuffer.push(event);
  }

  /** Ask the service to switch engines on this connection; the `info` reply confirms it. */
  setEngine(engine: string): void {
    this.engine = engine;
    this.url.searchParams.set('engine', engine);
    this.last = null;
    if (this.connected) this.send({ type: 'set_engine', engine });
  }

  stats(): ExtendedDecisionStats {
    return {
      engine: this.engine,
      decisions: this.decisions,
      p50_ms: this.rtt.percentile(0.5),
      p95_ms: this.rtt.percentile(0.95),
      think_p50_ms: this.think.percentile(0.5),
      think_p95_ms: this.think.percentile(0.95),
      decisions_per_s: this.rate.perSecond(nowMs()),
      inflight: this.pending !== null,
      connected: this.connected,
      skipped: this.skipped,
      errors: this.errors,
      timeouts: this.timeouts,
      stale: this.stale,
      reconnects: this.reconnects,
      engines: this.engines,
      last_error: this.lastError,
    };
  }

  lastDecision(): Decision | null {
    return this.last;
  }

  dispose(): void {
    this.disposed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.settlePending(null, 'disposed');
    if (this.ws) {
      this.ws.onopen = this.ws.onmessage = this.ws.onclose = this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }
  }

  // ---------------------------------------------------------------- internals

  private open(): void {
    if (this.disposed) return;
    let ws: WebSocket;
    try {
      ws = new this.WS(this.url.toString());
    } catch (e) {
      this.fail(`websocket: ${e instanceof Error ? e.message : String(e)}`);
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      if (this.attempt > 0) this.reconnects++;
      this.attempt = 0;
      this.resolveFirstOpen();
      for (const ev of this.eventBuffer.splice(0)) this.send({ type: 'event', event: ev });
    };
    ws.onmessage = (m: MessageEvent) => this.onMessage(m);
    ws.onerror = () => {
      this.fail('websocket error');
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.settlePending(null, 'connection closed');
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.disposed || this.reconnectTimer) return;
    const backoff = Math.min(this.maxMs, this.baseMs * 2 ** this.attempt);
    const jitter = backoff * (0.8 + Math.random() * 0.4);
    this.attempt++;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.open();
    }, jitter);
  }

  private send(payload: unknown): void {
    if (!this.connected || !this.ws) return;
    this.ws.send(JSON.stringify(payload));
  }

  private onMessage(m: MessageEvent): void {
    let msg: ServerMessage;
    try {
      msg = JSON.parse(typeof m.data === 'string' ? m.data : String(m.data)) as ServerMessage;
    } catch {
      this.fail('unparseable message from service');
      return;
    }
    switch (msg.type) {
      case 'decision':
        this.onDecision(msg.decision);
        break;
      case 'info':
        this.engine = msg.engine;
        this.engines = msg.engines;
        this.opts.onInfo?.(msg.engine, msg.engines);
        break;
      case 'error':
        this.fail(msg.message);
        this.settlePending(null, msg.message);
        break;
      default:
        this.fail(`unknown message type ${String((msg as { type?: unknown }).type)}`);
    }
  }

  private onDecision(d: Decision): void {
    const p = this.pending;
    if (!p || p.frameId !== d.frame_id) {
      // answer to a frame that already timed out (or was superseded): keep it, do not resolve
      this.stale++;
      this.last = d;
      return;
    }
    const rtt = nowMs() - p.sentAt;
    this.rtt.push(rtt);
    this.think.push(d.latency_ms);
    this.rate.mark(nowMs());
    this.decisions++;
    this.last = d;
    this.settlePending(d, '');
  }

  private onTimeout(frame: SensorFrame): void {
    if (!this.pending || this.pending.frameId !== frame.frame_id) return;
    this.timeouts++;
    this.fail(`decision for frame ${frame.frame_id} timed out after ${this.timeoutMs} ms`);
    this.settlePending(null, 'timed out');
  }

  private settlePending(d: Decision | null, why: string): void {
    const p = this.pending;
    if (!p) return;
    this.pending = null;
    clearTimeout(p.timer);
    p.resolve(d ?? this.fallback({ frame_id: p.frameId }, why));
  }

  private fallback(frame: { frame_id: number }, why: string): Decision {
    if (this.last) return { ...this.last, frame_id: frame.frame_id, reason: `${this.last.reason ?? ''} (stale: ${why})` };
    return {
      action: 'forward',
      engine: this.engine,
      frame_id: frame.frame_id,
      decision_id: FALLBACK_ID,
      latency_ms: 0,
      reason: `no decision yet (${why})`,
    };
  }

  private fail(message: string): void {
    this.errors++;
    this.lastError = message;
    this.opts.onError?.(message);
  }
}

export function createRemoteSource(url: string, engine: string, opts?: RemoteOptions): RemoteEngineSource {
  return new RemoteEngineSource(url, engine, opts);
}
