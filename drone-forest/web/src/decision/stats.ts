/**
 * Small measurement helpers shared by every DecisionSource: a bounded latency window with
 * percentiles, and a sliding-window rate meter. Pure, allocation-light, node-testable.
 */
import type { DecisionStats } from '../core/types';

/** What every source in this module reports. A superset of the contract's `DecisionStats`. */
export interface ExtendedDecisionStats extends DecisionStats {
  think_p95_ms: number;
  decisions_per_s: number;
  /** requests answered with a stale decision because one was already in flight or the link was down */
  skipped: number;
  errors: number;
  timeouts: number;
  /** answers that arrived for a frame no longer pending (after a timeout) */
  stale: number;
  reconnects: number;
  engines: string[];
  last_error: string | null;
}

export class LatencyWindow {
  private readonly buf: Float64Array;
  private head = 0;
  private size = 0;
  private total = 0;

  constructor(capacity = 256) {
    this.buf = new Float64Array(capacity);
  }

  push(ms: number): void {
    this.buf[this.head] = ms;
    this.head = (this.head + 1) % this.buf.length;
    if (this.size < this.buf.length) this.size++;
    this.total++;
  }

  get count(): number {
    return this.total;
  }

  /** Nearest-rank percentile over the retained window; 0 when empty. */
  percentile(q: number): number {
    if (this.size === 0) return 0;
    const sorted = Array.from(this.buf.subarray(0, this.size)).sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
    return sorted[idx] ?? 0;
  }

  last(): number {
    if (this.size === 0) return 0;
    return this.buf[(this.head - 1 + this.buf.length) % this.buf.length] ?? 0;
  }

  reset(): void {
    this.head = 0;
    this.size = 0;
    this.total = 0;
  }
}

export class RateMeter {
  private readonly marks: number[] = [];

  constructor(private readonly windowMs = 5000) {}

  mark(nowMs: number): void {
    this.marks.push(nowMs);
    this.prune(nowMs);
  }

  perSecond(nowMs: number): number {
    this.prune(nowMs);
    if (this.marks.length < 2) return this.marks.length === 1 ? 1000 / this.windowMs : 0;
    const span = Math.max(1, nowMs - (this.marks[0] ?? nowMs));
    return ((this.marks.length - 1) * 1000) / span;
  }

  private prune(nowMs: number): void {
    const cutoff = nowMs - this.windowMs;
    let drop = 0;
    while (drop < this.marks.length && (this.marks[drop] ?? Infinity) < cutoff) drop++;
    if (drop > 0) this.marks.splice(0, drop);
  }

  reset(): void {
    this.marks.length = 0;
  }
}

export const nowMs = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
