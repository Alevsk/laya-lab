/**
 * The decision loop: the only thing the game calls each frame. It asks the source at most once
 * per `intervalMs`, keeps the drone on the last action until the next answer arrives, and
 * keeps score of how many decision slots the engine met.
 *
 * Met vs missed: every request has one outcome. It is MET when the answer arrives before the
 * next slot would have been due, and MISSED otherwise (late or timed out) — the "could not
 * keep up" mechanic. Slots that pass while a request is still pending are counted separately
 * as `slots_skipped`; nothing is ever queued.
 *
 * Modes: 'realtime' (default) uses the wall clock and never waits. 'quality' uses simulation
 * time for the cadence and exposes `awaitDecision()` so an arena runner can step the world
 * deterministically, waiting for every answer — latency removed, only the policy measured.
 */
import { ACTIONS } from '../core/types';
import type { ActionName, Decision, DecisionSource, SensorFrame } from '../core/types';
import { nowMs } from './stats';

export type LoopMode = 'realtime' | 'quality';

export interface DecisionLoopOptions {
  mode?: LoopMode;
  /** wall clock in ms, injectable for tests (realtime mode only) */
  now?: () => number;
  initialAction?: ActionName;
  onDecision?: (decision: Decision, meta: DecisionMeta) => void;
}

export interface DecisionMeta {
  /** round trip in loop-clock ms (wall ms in realtime, sim ms in quality) */
  rtt_ms: number;
  /** wall-clock round trip, always measured */
  wall_ms: number;
  met: boolean;
}

export interface LoopStats {
  mode: LoopMode;
  interval_ms: number;
  requests: number;
  ticks_met: number;
  ticks_missed: number;
  slots_skipped: number;
  decisions: number;
  action_histogram: Record<ActionName, number>;
  option_index_hist: Record<number, number> | null;
  inflight: boolean;
  last_rtt_ms: number;
}

export interface DecisionLoop {
  /** Feed the frame; returns true when a request was sent this tick. */
  tick(frame: SensorFrame): boolean;
  /** Whether the next `tick` would ask, given the frame's time — lets the caller skip building a frame. */
  due(simTimeS: number): boolean;
  current(): ActionName;
  last(): Decision | null;
  /** Resolves when the in-flight request (if any) has been applied. */
  awaitDecision(): Promise<Decision | null>;
  stats(): LoopStats;
  mode(): LoopMode;
  setMode(mode: LoopMode): void;
  /** Swap the source; an answer still pending from the old one is ignored. */
  setSource(source: DecisionSource): void;
  source(): DecisionSource;
  /** New episode: forget the pending request, return to the initial action, clear counters. */
  reset(): void;
}

const emptyHistogram = (): Record<ActionName, number> => {
  const h = {} as Record<ActionName, number>;
  for (const a of ACTIONS) h[a] = 0;
  return h;
};

export function createDecisionLoop(
  initialSource: DecisionSource,
  intervalMs: number,
  opts: DecisionLoopOptions = {},
): DecisionLoop {
  let source = initialSource;
  let mode: LoopMode = opts.mode ?? 'realtime';
  const wall = opts.now ?? nowMs;
  const initial = opts.initialAction ?? 'forward';

  let action: ActionName = initial;
  let lastDecision: Decision | null = null;
  let lastCountedId: number | null = null;
  let generation = 0;

  let nextDueAt: number | null = null; // in loop-clock ms
  let inflight: { askedAt: number; askedWall: number; promise: Promise<Decision | null> } | null = null;

  let requests = 0;
  let met = 0;
  let missed = 0;
  let skipped = 0;
  let decisions = 0;
  let lastRtt = 0;
  let histogram = emptyHistogram();
  let optionHist: Record<number, number> | null = null;

  const clock = (frame: SensorFrame): number => (mode === 'quality' ? frame.t * 1000 : wall());
  const isDue = (now: number): boolean => nextDueAt === null || now >= nextDueAt - 1e-6;

  const apply = (d: Decision, now: number, askedAt: number, askedWall: number): void => {
    const rtt = now - askedAt;
    const wallRtt = wall() - askedWall;
    const ok = mode === 'quality' || rtt <= intervalMs;
    if (ok) met++;
    else missed++;
    lastRtt = wallRtt;
    action = d.action;
    lastDecision = d;
    if (d.decision_id >= 0 && d.decision_id !== lastCountedId) {
      lastCountedId = d.decision_id;
      decisions++;
      histogram[d.action] = (histogram[d.action] ?? 0) + 1;
      if (d.option_index !== null && d.option_index !== undefined) {
        optionHist ??= {};
        optionHist[d.option_index] = (optionHist[d.option_index] ?? 0) + 1;
      }
    }
    opts.onDecision?.(d, { rtt_ms: rtt, wall_ms: wallRtt, met: ok });
  };

  const tick = (frame: SensorFrame): boolean => {
    const now = clock(frame);
    if (!isDue(now)) return false;
    // advance the cadence in whole slots so a stall does not produce a burst of requests
    nextDueAt = (nextDueAt ?? now) + intervalMs;
    while (nextDueAt <= now) nextDueAt += intervalMs;

    if (inflight) {
      skipped++;
      return false;
    }

    const gen = generation;
    const askedAt = now;
    const askedWall = wall();
    requests++;
    const promise = source
      .decide(frame)
      .then((d) => {
        if (gen !== generation) return null;
        inflight = null;
        // in quality mode the answer lands at the same sim instant it was asked
        apply(d, mode === 'quality' ? askedAt : wall(), askedAt, askedWall);
        return d;
      })
      .catch((e: unknown) => {
        if (gen !== generation) return null;
        inflight = null;
        missed++;
        const reason = e instanceof Error ? e.message : String(e);
        opts.onDecision?.(
          { action, engine: source.name, frame_id: frame.frame_id, decision_id: -1, latency_ms: 0, reason: `source failed: ${reason}` },
          { rtt_ms: wall() - askedWall, wall_ms: wall() - askedWall, met: false },
        );
        return null;
      });
    inflight = { askedAt, askedWall, promise };
    return true;
  };

  const reset = (): void => {
    generation++;
    inflight = null;
    nextDueAt = null;
    action = initial;
    lastDecision = null;
    lastCountedId = null;
    requests = met = missed = skipped = decisions = lastRtt = 0;
    histogram = emptyHistogram();
    optionHist = null;
  };

  return {
    tick,
    due: (t) => isDue(mode === 'quality' ? t * 1000 : wall()),
    current: () => action,
    last: () => lastDecision,
    awaitDecision: () => (inflight ? inflight.promise : Promise.resolve(lastDecision)),
    stats: () => ({
      mode,
      interval_ms: intervalMs,
      requests,
      ticks_met: met,
      ticks_missed: missed,
      slots_skipped: skipped,
      decisions,
      action_histogram: { ...histogram },
      option_index_hist: optionHist ? { ...optionHist } : null,
      inflight: inflight !== null,
      last_rtt_ms: lastRtt,
    }),
    mode: () => mode,
    setMode: (m) => {
      mode = m;
      nextDueAt = null;
    },
    setSource: (s) => {
      generation++;
      inflight = null;
      source = s;
    },
    source: () => source,
    reset,
  };
}
