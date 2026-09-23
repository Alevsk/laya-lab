/** The no-op policy for scenery that never moves. One shared instance is enough: it holds no state. */
import type { Obstacle, ObstacleBehavior, WorldContext } from '../../core/types';

export class StaticBehavior implements ObstacleBehavior {
  readonly name = 'static';
  update(_obstacle: Obstacle, _ctx: WorldContext): void {}
}

export const STATIC_BEHAVIOR: ObstacleBehavior = new StaticBehavior();
