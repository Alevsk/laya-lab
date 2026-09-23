/**
 * The forest spawner. It lays obstacle ROWS across the corridor at a fixed spacing along the
 * flight axis, always far enough ahead that new rows appear inside the fog, and it recycles
 * whatever has fallen well behind the drone.
 *
 * Guarantees
 *   - flyable: every row has a clear gap whose centre performs a bounded random walk from row
 *     to row, so a continuous path always exists and never jumps further sideways per row than
 *     a banking drone can follow (`gapDrift`); nothing with `respectsGap` may touch the band;
 *   - bounded: each kind is capped at `profile.maxCount`, and the per-row budget is derived
 *     from that cap and the populated span so the caps are respected in steady state rather
 *     than hit and starved;
 *   - spaced: surface-to-surface clearance of `profile.minSpacing` to every live obstacle;
 *   - deterministic: every draw comes from the rng handed to `createSpawner`, and rows are
 *     keyed by distance not by time, so the same seed and drone path produce the same forest;
 *   - kind-agnostic: the spawner reads each factory's `SpawnProfile` and never names a kind.
 *
 * Difficulty ramps the obstacle density from `baseDensity` to 1.0 over `rampDistance` metres
 * flown and narrows the clear gap from `clearGap[0]` to `clearGap[1]`.
 *
 * Contract note: `update` splices expired obstacles OUT of the array it is given (disposing
 * them) and RETURNS the newly spawned ones; the world appends those to its list. Every spawned
 * obstacle has already been added to `ctx.scene` by its factory.
 */
import * as THREE from 'three';
import type { Obstacle, ObstacleFactory, ObstacleKind, Rng, Spawner, WorldContext } from '../core/types';
import { EXPIRE_BEHIND_M } from './base';
import { DEFAULT_CORRIDOR, type CorridorFrame } from './corridor';
import { profileOf, type SpawnProfile } from './profile';
import { defaultFactories } from './registry';

export interface SpawnerOptions {
  corridor?: CorridorFrame;
  /** Distance between rows along the corridor, metres. */
  rowSpacing?: number;
  /** Each placement is jittered ±this along the corridor so rows do not read as rows. */
  rowJitter?: number;
  /** Rows are kept populated to max(minHorizon, bounds.corridor_length) ahead of the drone. */
  minHorizon?: number;
  /** Nothing spawns closer than this ahead of a fresh or reset drone. */
  startClear?: number;
  /** Half-width of the guaranteed clear gap at difficulty 0 and 1. */
  clearGap?: readonly [easy: number, hard: number];
  /** Maximum lateral drift of the gap centre from one row to the next. */
  gapDrift?: number;
  /** Density multiplier at distance 0; ramps linearly to 1 at `rampDistance`. */
  baseDensity?: number;
  rampDistance?: number;
  /** Placement attempts before a spot is given up. */
  placementAttempts?: number;
}

export interface RowRecord {
  readonly along: number;
  readonly gapCentre: number;
  readonly gapHalfWidth: number;
}

export interface ForestSpawner extends Spawner {
  /** Forget all rows; the next `update` disposes every obstacle in its array and refills. */
  reset(): void;
  /** Rows currently alive (from just behind the drone to the horizon), oldest first. */
  rows(): readonly RowRecord[];
  /** 0 at the start, 1 once `rampDistance` metres have been flown. */
  difficulty(): number;
}

/** A backwards jump this large along the corridor means the drone was reset. */
const RESET_JUMP_M = 50;
const MAX_ROWS_PER_UPDATE = 64;

interface Resolved {
  corridor: CorridorFrame;
  rowSpacing: number;
  rowJitter: number;
  minHorizon: number;
  startClear: number;
  clearGap: readonly [number, number];
  gapDrift: number;
  baseDensity: number;
  rampDistance: number;
  placementAttempts: number;
}

export function createSpawner(rng: Rng, factories?: ObstacleFactory[], options: SpawnerOptions = {}): ForestSpawner {
  const o: Resolved = {
    corridor: options.corridor ?? DEFAULT_CORRIDOR,
    rowSpacing: options.rowSpacing ?? 6,
    rowJitter: options.rowJitter ?? 1.5,
    minHorizon: options.minHorizon ?? 220,
    startClear: options.startClear ?? 30,
    clearGap: options.clearGap ?? [5, 3.5],
    gapDrift: options.gapDrift ?? 1.8,
    baseDensity: options.baseDensity ?? 0.55,
    rampDistance: options.rampDistance ?? 1500,
    placementAttempts: options.placementAttempts ?? 8,
  };
  const kinds = (factories ?? defaultFactories({ corridor: o.corridor })).map((factory) => ({
    factory,
    profile: profileOf(factory),
  }));
  const c = o.corridor;

  let rows: RowRecord[] = [];
  let startAlong: number | undefined;
  let nextRowAlong = 0;
  let gapCentre = 0;
  let lastDroneAlong = -Infinity;
  let pendingReset = false;

  const anchor = new THREE.Vector3();

  const difficulty = (droneAlong: number): number =>
    startAlong === undefined ? 0 : Math.min(1, Math.max(0, (droneAlong - startAlong) / o.rampDistance));

  const clearAll = (obstacles: Obstacle[]): void => {
    for (const obstacle of obstacles) obstacle.dispose();
    obstacles.length = 0;
    rows = [];
    startAlong = undefined;
  };

  const expire = (ctx: WorldContext, obstacles: Obstacle[], droneAlong: number): void => {
    for (let i = obstacles.length - 1; i >= 0; i--) {
      const obstacle = obstacles[i]!;
      if (obstacle.isExpired(ctx)) {
        obstacle.dispose();
        obstacles.splice(i, 1);
      }
    }
    const keepFrom = droneAlong - EXPIRE_BEHIND_M - o.rowJitter;
    while (rows.length > 0 && rows[0]!.along < keepFrom) rows.shift();
  };

  const countByKind = (obstacles: readonly Obstacle[]): Map<ObstacleKind, number> => {
    const counts = new Map<ObstacleKind, number>();
    for (const obstacle of obstacles) counts.set(obstacle.kind, (counts.get(obstacle.kind) ?? 0) + 1);
    return counts;
  };

  const tooClose = (obstacles: readonly Obstacle[], along: number, across: number, radius: number, spacing: number): boolean => {
    for (const other of obstacles) {
      const da = c.along(other.position) - along;
      const dx = c.across(other.position) - across;
      const need = radius + other.radius + spacing;
      if (da * da + dx * dx < need * need) return true;
    }
    return false;
  };

  const rowCount = (profile: SpawnProfile, ctx: WorldContext, span: number, d: number, live: number): number => {
    const area = 2 * ctx.bounds.half_width * o.rowSpacing;
    const expected = (profile.density * area * (o.baseDensity + (1 - o.baseDensity) * d)) / 100;
    const budget = (profile.maxCount * o.rowSpacing) / span;
    const n = Math.min(expected, budget);
    const whole = Math.floor(n);
    const count = whole + (rng.next() < n - whole ? 1 : 0);
    return Math.min(count, Math.max(0, profile.maxCount - live));
  };

  const spawnRow = (ctx: WorldContext, obstacles: Obstacle[], along: number, span: number, d: number): Obstacle[] => {
    const hw = ctx.bounds.half_width;
    const gapHalf = o.clearGap[0] + (o.clearGap[1] - o.clearGap[0]) * d;
    gapCentre = clamp(gapCentre + rng.range(-o.gapDrift, o.gapDrift), -hw + gapHalf, hw - gapHalf);
    rows.push({ along, gapCentre, gapHalfWidth: gapHalf });

    const counts = countByKind(obstacles);
    const spawned: Obstacle[] = [];
    for (const { factory, profile } of kinds) {
      const wanted = rowCount(profile, ctx, span, d, counts.get(factory.kind) ?? 0);
      for (let i = 0; i < wanted; i++) {
        for (let attempt = 0; attempt < o.placementAttempts; attempt++) {
          const scale = rng.range(profile.scale[0], profile.scale[1]);
          const radius = profile.baseRadius * scale;
          const across = rng.range(-hw + radius, hw - radius);
          const rowAlong = along + rng.range(-o.rowJitter, o.rowJitter);
          if (profile.respectsGap && Math.abs(across - gapCentre) < gapHalf + radius) continue;
          if (tooClose(obstacles, rowAlong, across, radius, profile.minSpacing)) continue;
          if (tooClose(spawned, rowAlong, across, radius, profile.minSpacing)) continue;
          const [altMin, altMax] = profile.altitude ?? [ctx.bounds.altitude_min, ctx.bounds.altitude_max];
          const height = profile.layer === 'air' ? rng.range(altMin, altMax) : 0;
          c.compose(rowAlong, across, height, anchor);
          spawned.push(factory.create(ctx, { position: anchor.clone(), scale }));
          break;
        }
      }
    }
    return spawned;
  };

  return {
    reset() {
      pendingReset = true;
    },
    rows: () => rows,
    difficulty: () => difficulty(lastDroneAlong),

    update(ctx, obstacles) {
      const droneAlong = c.along(ctx.drone.position);
      if (pendingReset || droneAlong < lastDroneAlong - RESET_JUMP_M) {
        clearAll(obstacles);
        pendingReset = false;
      }
      lastDroneAlong = droneAlong;

      if (startAlong === undefined) {
        startAlong = droneAlong;
        nextRowAlong = droneAlong + o.startClear;
        gapCentre = c.across(ctx.drone.position);
      }

      expire(ctx, obstacles, droneAlong);

      const horizon = Math.max(o.minHorizon, ctx.bounds.corridor_length);
      const span = horizon + EXPIRE_BEHIND_M;
      const d = difficulty(droneAlong);
      // Budget starvation or a stall must never let rows pop in close to the drone.
      nextRowAlong = Math.max(nextRowAlong, droneAlong + o.startClear);

      const spawned: Obstacle[] = [];
      for (let n = 0; n < MAX_ROWS_PER_UPDATE && nextRowAlong <= droneAlong + horizon; n++) {
        spawned.push(...spawnRow(ctx, obstacles, nextRowAlong, span, d));
        nextRowAlong += o.rowSpacing;
      }
      return spawned;
    },
  };
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
