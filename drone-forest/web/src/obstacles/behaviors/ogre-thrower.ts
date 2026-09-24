/**
 * The ogre's policy: stand still, face the drone, and when it comes into range wind up and
 * throw a rock at where the drone is GOING to be. Everything that makes the ogre dangerous
 * scales with `ctx.difficulty` and is read live, so changing the level in the HUD changes the
 * next throw:
 *
 *   level      1      3      5
 *   range     35 m   50 m   65 m      engagement distance (never closer than 14 m)
 *   cooldown  5.0 s  3.7 s  2.4 s     between throws (x0.8..1.2 jitter)
 *   speed     12     17     22 m/s    launch speed
 *   aim error 12°    7.5°    3°       random cone around the solved launch direction
 *   lead       0     0.43   0.85      0 = aim at the drone's current position, 1 = full intercept
 *   wind-up   1.0 s  0.78 s 0.55 s    the telegraph before release (arm raised)
 *
 * Throws come out of `throwRock`, injected by the factory, and enter the world through
 * `ctx.emit` — the behaviour never touches the scene or the obstacle list itself.
 */
import * as THREE from 'three';
import type { Obstacle, ObstacleBehavior, WorldContext } from '../../core/types';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import { perturbDirection, predictAimPoint, solveLaunch } from './ballistics';

export interface OgreTuning {
  range: number;
  cooldown: number;
  rockSpeed: number;
  aimErrorDeg: number;
  lead: number;
  windup: number;
}

const EASY: OgreTuning = { range: 35, cooldown: 5.0, rockSpeed: 12, aimErrorDeg: 12, lead: 0, windup: 1.0 };
const BRUTAL: OgreTuning = { range: 65, cooldown: 2.4, rockSpeed: 22, aimErrorDeg: 3, lead: 0.85, windup: 0.55 };
/** No throws at a drone closer than this: a point-blank rock cannot be dodged by anyone. */
const MIN_THROW_DISTANCE = 14;

/** Linear blend of the level-1 and level-5 tunings; levels outside 1..5 are clamped. */
export function tuningFor(level: number): OgreTuning {
  const k = Math.min(1, Math.max(0, (level - 1) / 4));
  const mix = (a: number, b: number): number => a + (b - a) * k;
  return {
    range: mix(EASY.range, BRUTAL.range),
    cooldown: mix(EASY.cooldown, BRUTAL.cooldown),
    rockSpeed: mix(EASY.rockSpeed, BRUTAL.rockSpeed),
    aimErrorDeg: mix(EASY.aimErrorDeg, BRUTAL.aimErrorDeg),
    lead: mix(EASY.lead, BRUTAL.lead),
    windup: mix(EASY.windup, BRUTAL.windup),
  };
}

export interface OgreThrowerOptions {
  corridor?: CorridorFrame;
  /** Build a projectile at `origin` with launch `velocity`; the behaviour emits it into the world. */
  throwRock: (ctx: WorldContext, origin: THREE.Vector3, velocity: THREE.Vector3) => Obstacle;
  /** Height of the throwing hand above the obstacle's sphere centre, metres (scaled by the factory). */
  handHeight?: number;
  /** How far past the ogre the drone may be before it stops bothering, metres along the corridor. */
  giveUpBehind?: number;
}

export type OgrePhase = 'idle' | 'windup' | 'cooldown';

export class OgreThrowerBehavior implements ObstacleBehavior {
  readonly name = 'ogre_thrower';
  phase: OgrePhase = 'idle';
  /** 0..1 how far the throwing arm is raised - the factory animates from this */
  armRaise = 0;
  throws = 0;
  /** unit vector toward the drone, horizontal - the factory turns the model to face it */
  readonly facing = new THREE.Vector3(0, 0, -1);
  private readonly corridor: CorridorFrame;
  private readonly throwRock: OgreThrowerOptions['throwRock'];
  private readonly handHeight: number;
  private readonly giveUpBehind: number;
  private nextThrowAt: number | null = null;
  private windupStarted = 0;
  private windupFor = 0;
  private readonly tmp = new THREE.Vector3();

  constructor(options: OgreThrowerOptions) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    this.throwRock = options.throwRock;
    this.handHeight = options.handHeight ?? 1.1;
    this.giveUpBehind = options.giveUpBehind ?? 8;
  }

  update(obstacle: Obstacle, ctx: WorldContext): void {
    const c = this.corridor;
    const tuning = tuningFor(ctx.difficulty);
    const drone = ctx.drone;
    const toDrone = this.tmp.copy(drone.position).sub(obstacle.position);
    const distance = toDrone.length();
    const horizontal = toDrone.clone().addScaledVector(c.up, -toDrone.dot(c.up));
    if (horizontal.lengthSq() > 1e-6) this.facing.copy(horizontal.normalize());
    // the drone is a target while it is within range and has not flown far past the ogre
    const aheadOfMe = c.along(obstacle.position) - c.along(drone.position);
    const engaged = distance <= tuning.range && distance >= MIN_THROW_DISTANCE && aheadOfMe > -this.giveUpBehind;

    if (this.nextThrowAt === null) {
      // first throw after a short, seeded delay so a row of ogres does not fire in unison
      this.nextThrowAt = ctx.time + ctx.rng.range(0.2, 1.2);
    }

    switch (this.phase) {
      case 'idle':
        if (engaged && ctx.time >= this.nextThrowAt) {
          this.phase = 'windup';
          this.windupStarted = ctx.time;
          this.windupFor = tuning.windup;
        }
        break;
      case 'windup': {
        this.armRaise = Math.min(1, (ctx.time - this.windupStarted) / this.windupFor);
        if (!engaged) {
          this.phase = 'idle';
          this.armRaise = 0;
          break;
        }
        if (this.armRaise >= 1) {
          this.release(obstacle, ctx, tuning);
          this.phase = 'cooldown';
          this.nextThrowAt = ctx.time + tuning.cooldown * ctx.rng.range(0.8, 1.2);
        }
        break;
      }
      case 'cooldown':
        this.armRaise = Math.max(0, this.armRaise - ctx.dt * 3);
        if (ctx.time >= (this.nextThrowAt ?? 0)) this.phase = 'idle';
        break;
    }
  }

  private release(obstacle: Obstacle, ctx: WorldContext, tuning: OgreTuning): void {
    const c = this.corridor;
    const origin = obstacle.position
      .clone()
      .addScaledVector(c.up, this.handHeight)
      .addScaledVector(this.facing, obstacle.radius * 0.6);
    const aim = predictAimPoint(origin, ctx.drone.position, ctx.drone.velocity, tuning.rockSpeed, tuning.lead, undefined, c.up);
    const solved = solveLaunch(origin, aim, tuning.rockSpeed, undefined, c.up);
    // triangular error distribution: mostly near the solved direction, occasionally wide
    const errorRad = ((ctx.rng.next() + ctx.rng.next() - 1) * tuning.aimErrorDeg * Math.PI) / 180;
    const velocity = perturbDirection(solved.velocity, Math.abs(errorRad), ctx.rng);
    ctx.emit(this.throwRock(ctx, origin, velocity));
    this.throws += 1;
    this.armRaise = 0;
  }
}
