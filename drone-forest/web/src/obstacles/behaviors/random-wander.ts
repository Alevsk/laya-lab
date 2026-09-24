/**
 * Piecewise-constant random flight: every 0.8–2.5 s a new velocity is drawn from the world rng
 * (random horizontal heading, speed and a small vertical component), integrated each frame,
 * and reflected off the corridor walls and an altitude band so the obstacle never leaves the
 * playable volume. A sinusoidal vertical bob rides on top of the wander velocity so the motion
 * reads as flight rather than a sliding token.
 *
 * All randomness comes from `ctx.rng`, so a seeded world replays identically at a fixed step.
 * One instance per obstacle: the redraw timer and current velocity are per-obstacle state.
 */
import type { Obstacle, ObstacleBehavior, WorldContext } from '../../core/types';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';

export interface RandomWanderOptions {
  /** Metres per second, drawn uniformly each redraw. */
  speed?: readonly [number, number];
  /** Vertical velocity component drawn each redraw, m/s. */
  verticalSpeed?: readonly [number, number];
  /** Seconds between redraws, drawn uniformly. */
  redrawInterval?: readonly [number, number];
  /** Height band (metres above ground). Defaults to the world's altitude bounds. */
  altitudeBand?: readonly [number, number];
  /** Keep this far inside the corridor walls. */
  wallMargin?: number;
  /** Vertical bob amplitude in metres (0 disables). */
  bobAmplitude?: number;
  bobHz?: number;
  corridor?: CorridorFrame;
}

export class RandomWanderBehavior implements ObstacleBehavior {
  readonly name = 'random_wander';
  private readonly speed: readonly [number, number];
  private readonly verticalSpeed: readonly [number, number];
  private readonly redrawInterval: readonly [number, number];
  private readonly altitudeBand: readonly [number, number] | undefined;
  private readonly wallMargin: number;
  private readonly bobAmplitude: number;
  private readonly bobOmega: number;
  private readonly corridor: CorridorFrame;
  private nextRedrawAt = -Infinity;
  private bobPhase = 0;

  constructor(options: RandomWanderOptions = {}) {
    this.speed = options.speed ?? [2, 6];
    this.verticalSpeed = options.verticalSpeed ?? [-1, 1];
    this.redrawInterval = options.redrawInterval ?? [0.8, 2.5];
    this.altitudeBand = options.altitudeBand;
    this.wallMargin = options.wallMargin ?? 1;
    this.bobAmplitude = options.bobAmplitude ?? 0.25;
    this.bobOmega = 2 * Math.PI * (options.bobHz ?? 1.4);
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
  }

  update(obstacle: Obstacle, ctx: WorldContext): void {
    const { rng, dt } = ctx;
    const c = this.corridor;
    const v = obstacle.velocity;
    const p = obstacle.position;

    if (ctx.time >= this.nextRedrawAt) {
      const heading = rng.range(0, 2 * Math.PI);
      const speed = rng.range(this.speed[0], this.speed[1]);
      const vertical = rng.range(this.verticalSpeed[0], this.verticalSpeed[1]);
      c.compose(Math.cos(heading) * speed, Math.sin(heading) * speed, vertical, v);
      this.nextRedrawAt = ctx.time + rng.range(this.redrawInterval[0], this.redrawInterval[1]);
      this.bobPhase = rng.range(0, 2 * Math.PI);
    }

    const [altMin, altMax] = this.altitudeBand ?? [ctx.bounds.altitude_min, ctx.bounds.altitude_max];
    const wall = ctx.bounds.half_width - this.wallMargin;
    const bob = this.bobAmplitude * this.bobOmega * Math.cos(this.bobOmega * ctx.time + this.bobPhase);

    // Integrate, then reflect any component that carried the obstacle past a boundary.
    p.addScaledVector(v, dt).addScaledVector(c.up, bob * dt);

    // The walls and the altitude band are hard limits: always clamp the position back inside,
    // and reflect the velocity only when it points outward. (Clamping only while moving
    // outward let the bob term creep an obstacle a few decimetres past the band.)
    const across = c.across(p);
    const vAcross = c.across(v);
    if (across > wall || across < -wall) {
      if ((across > wall && vAcross > 0) || (across < -wall && vAcross < 0)) v.addScaledVector(c.right, -2 * vAcross);
      p.addScaledVector(c.right, clamp(across, -wall, wall) - across);
    }
    const height = p.dot(c.up);
    const vUp = v.dot(c.up);
    if (height > altMax || height < altMin) {
      if ((height > altMax && vUp > 0) || (height < altMin && vUp < 0)) v.addScaledVector(c.up, -2 * vUp);
      p.addScaledVector(c.up, clamp(height, altMin, altMax) - height);
    }
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
