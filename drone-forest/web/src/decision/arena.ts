/**
 * Arena: a benchmark run on a fixed seed for a fixed number of *simulated* seconds, so engines
 * can be compared on identical worlds. main.ts invokes it from URL params, e.g.
 * `?arena=quality&seconds=60&engine=laya&seed=7`.
 *
 * 'quality' mode steps the world at a fixed dt and waits for every decision — latency is taken
 * out of the picture and only the policy is measured. 'realtime' mode still steps at the same
 * fixed dt, but how many steps happen per iteration follows the real elapsed time, so a slow
 * engine really does lose slots exactly as it would in play. Either way the result carries
 * real, measured numbers: nothing here is estimated.
 *
 * While a run is in progress the caller must not step the world or the drone itself; the
 * arena owns the simulation until it resolves.
 */
import type * as THREE from 'three';

import type { ActionName, CollisionResult, DecisionSource, GameEvent, Obstacle, SensorFrame, WorldContext } from '../core/types';
import { createDecisionLoop } from './loop';
import type { DecisionLoop, LoopMode } from './loop';
import { nowMs } from './stats';

export type ArenaMode = LoopMode;

export interface ArenaParams {
  mode: ArenaMode;
  seconds: number;
  engine: string;
  seed: number;
}

export interface ArenaResult {
  engine: string;
  mode: ArenaMode;
  seed: number;
  seconds: number;
  /** horizontal path length flown, metres */
  distance: number;
  /** straight-line horizontal distance from the start, metres */
  displacement: number;
  collisions: number;
  near_misses: number;
  ticks_met: number;
  ticks_missed: number;
  slots_skipped: number;
  decisions: number;
  decision_p50_ms: number;
  decision_p95_ms: number;
  think_p50_ms: number;
  action_histogram: Record<ActionName, number>;
  option_index_hist?: Record<number, number>;
  sim_steps: number;
  wall_ms: number;
}

export interface ArenaProgress {
  t: number;
  seconds: number;
  collisions: number;
  near_misses: number;
  distance: number;
  action: ActionName;
}

export interface ArenaWorld {
  readonly obstacles: readonly Obstacle[];
  readonly ctx: WorldContext;
  update(dt: number): void;
}

export interface ArenaDrone {
  readonly position: THREE.Vector3;
  applyAction(action: ActionName, dt: number): void;
  update(dt: number, ctx: WorldContext): void;
  collide(obstacles: readonly Obstacle[]): CollisionResult;
  reset(): void;
}

export interface ArenaDeps {
  world: ArenaWorld;
  drone: ArenaDrone;
  sensors: { frame(frameId: number, t: number, lastAction: ActionName | null): SensorFrame };
  source: DecisionSource;
  /** use this loop (switched to the arena's mode) instead of a private one, so a HUD can watch it */
  loop?: DecisionLoop;
  /** decision cadence; default 100 ms */
  intervalMs?: number;
  /** fixed physics step; default 1/60 */
  dt?: number;
  /** realtime mode only: called once per iteration so the run is visible */
  render?: () => void;
  onProgress?: (p: ArenaProgress) => void;
  onFrame?: (frame: SensorFrame) => void;
  /** yield to the event loop; defaults to requestAnimationFrame in a page, setTimeout elsewhere */
  yieldFrame?: () => Promise<number>;
  now?: () => number;
}

export const ARENA_RESULT_PREFIX = 'ARENA_RESULT';

export function parseArenaParams(search: string): ArenaParams | null {
  const q = new URLSearchParams(search);
  const arena = q.get('arena');
  if (arena !== 'quality' && arena !== 'realtime') return null;
  const seconds = Number(q.get('seconds') ?? 60);
  const seed = Number(q.get('seed') ?? 7);
  return {
    mode: arena,
    seconds: Number.isFinite(seconds) && seconds > 0 ? seconds : 60,
    engine: q.get('engine') ?? 'local',
    seed: Number.isFinite(seed) ? Math.floor(seed) : 7,
  };
}

const defaultYield = (): Promise<number> =>
  typeof requestAnimationFrame === 'function'
    ? new Promise<number>((resolve) => requestAnimationFrame(resolve))
    : new Promise<number>((resolve) => setTimeout(() => resolve(nowMs()), 0));

export async function runArena(params: ArenaParams, deps: ArenaDeps): Promise<ArenaResult> {
  const dt = deps.dt ?? 1 / 60;
  const intervalMs = deps.intervalMs ?? 100;
  const now = deps.now ?? nowMs;
  const yieldFrame = deps.yieldFrame ?? defaultYield;
  const { world, drone, sensors, source } = deps;
  const loop = deps.loop ?? createDecisionLoop(source, intervalMs, { mode: params.mode, now });
  loop.setMode(params.mode);

  const totalSteps = Math.round(params.seconds / dt);
  let step = 0;
  let frameId = 0;
  let t = 0;
  let collisions = 0;
  let nearMisses = 0;
  let distance = 0;
  const activeHits = new Set<number>();
  const activeNear = new Set<number>();

  drone.reset();
  loop.reset();
  const startX = drone.position.x;
  const startZ = drone.position.z;
  let prevX = startX;
  let prevZ = startZ;
  source.report({ type: 'reset', frame_id: 0, t: 0, details: { arena: params } });

  const event = (type: GameEvent['type'], obstacle: Obstacle | null, details: Record<string, unknown> = {}): void => {
    source.report({ type, frame_id: frameId, t, obstacle_kind: obstacle?.kind ?? null, details: { id: obstacle?.id, ...details } });
  };

  const stepOnce = async (): Promise<void> => {
    if (loop.due(t)) {
      const frame = sensors.frame(frameId++, t, loop.last()?.action ?? null);
      deps.onFrame?.(frame);
      loop.tick(frame);
      if (params.mode === 'quality') await loop.awaitDecision();
    }
    drone.applyAction(loop.current(), dt);
    drone.update(dt, world.ctx);
    world.update(dt);
    t += dt;
    step++;

    const dx = drone.position.x - prevX;
    const dz = drone.position.z - prevZ;
    distance += Math.sqrt(dx * dx + dz * dz);
    prevX = drone.position.x;
    prevZ = drone.position.z;

    const res = drone.collide(world.obstacles);
    if (res.hit) {
      if (!activeHits.has(res.hit.id)) {
        activeHits.add(res.hit.id);
        collisions++;
        event('collision', res.hit, { action: loop.current() });
      }
    } else {
      activeHits.clear();
    }
    if (res.nearMiss && !activeHits.has(res.nearMiss.id)) {
      if (!activeNear.has(res.nearMiss.id)) {
        activeNear.add(res.nearMiss.id);
        nearMisses++;
        event('near_miss', res.nearMiss, { action: loop.current() });
      }
    } else if (!res.nearMiss) {
      activeNear.clear();
    }
  };

  const progress = (): void =>
    deps.onProgress?.({ t, seconds: params.seconds, collisions, near_misses: nearMisses, distance, action: loop.current() });

  const wallStart = now();
  if (params.mode === 'quality') {
    while (step < totalSteps) {
      await stepOnce();
      if (step % 60 === 0) {
        progress();
        deps.render?.();
        await yieldFrame();
      }
    }
  } else {
    let lastWall = now();
    let acc = 0;
    while (step < totalSteps) {
      await yieldFrame();
      const wallNow = now();
      acc += Math.min(0.25, (wallNow - lastWall) / 1000);
      lastWall = wallNow;
      while (acc >= dt && step < totalSteps) {
        await stepOnce();
        acc -= dt;
      }
      progress();
      deps.render?.();
    }
  }
  const wallMs = now() - wallStart;

  const ls = loop.stats();
  const ss = source.stats();
  const dx = drone.position.x - startX;
  const dz = drone.position.z - startZ;
  const result: ArenaResult = {
    engine: ss.engine,
    mode: params.mode,
    seed: params.seed,
    seconds: params.seconds,
    distance: round(distance),
    displacement: round(Math.sqrt(dx * dx + dz * dz)),
    collisions,
    near_misses: nearMisses,
    ticks_met: ls.ticks_met,
    ticks_missed: ls.ticks_missed,
    slots_skipped: ls.slots_skipped,
    decisions: ls.decisions,
    decision_p50_ms: round(ss.p50_ms),
    decision_p95_ms: round(ss.p95_ms),
    think_p50_ms: round(ss.think_p50_ms),
    action_histogram: ls.action_histogram,
    ...(ls.option_index_hist ? { option_index_hist: ls.option_index_hist } : {}),
    sim_steps: step,
    wall_ms: round(wallMs),
  };

  console.log(`${ARENA_RESULT_PREFIX} ${JSON.stringify(result)}`);
  source.report({ type: 'score', frame_id: frameId, t, details: { ...result } });
  return result;
}

const round = (x: number): number => Math.round(x * 100) / 100;
