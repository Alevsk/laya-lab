/**
 * A thrown object: integrate gravity, stop on the ground. The projectile's obstacle owns the
 * launch velocity (set at creation); this behaviour only bends it downward and detects the
 * landing, after which the obstacle lies still until it expires.
 */
import type { Obstacle, ObstacleBehavior, WorldContext } from '../../core/types';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import { GRAVITY } from './ballistics';

export class BallisticBehavior implements ObstacleBehavior {
  readonly name = 'ballistic';
  landed = false;
  landedAt = 0;
  private readonly corridor: CorridorFrame;
  private readonly gravity: number;

  constructor(options: { corridor?: CorridorFrame; gravity?: number } = {}) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    this.gravity = options.gravity ?? GRAVITY;
  }

  update(obstacle: Obstacle, ctx: WorldContext): void {
    if (this.landed) return;
    const up = this.corridor.up;
    obstacle.velocity.addScaledVector(up, -this.gravity * ctx.dt);
    obstacle.position.addScaledVector(obstacle.velocity, ctx.dt);
    const height = obstacle.position.dot(up);
    if (height <= obstacle.radius) {
      obstacle.position.addScaledVector(up, obstacle.radius - height);
      obstacle.velocity.set(0, 0, 0);
      this.landed = true;
      this.landedAt = ctx.time;
    }
  }
}
