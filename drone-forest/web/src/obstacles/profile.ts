/**
 * What the spawner needs to know about a kind WITHOUT knowing the kind. A factory that carries
 * a `SpawnProfile` describes its own placement rules; the spawner reads the profile and never
 * branches on `kind`. That is what keeps "add an obstacle kind" to one new file: the placement
 * knowledge travels with the factory instead of accumulating as cases inside the spawner.
 *
 * A factory without a profile (a bare `ObstacleFactory` from the contract) is spawned with
 * `DEFAULT_PROFILE` — a sparse ground object that keeps out of the clear path.
 */
import type { ObstacleFactory } from '../core/types';

export interface SpawnProfile {
  /**
   * `ground`: the spawn anchor is the ground contact point (y = 0) and the factory lifts its
   * sphere centre to wherever the model needs it. `air`: the anchor is the sphere centre, at a
   * height drawn from `altitude`.
   */
  readonly layer: 'ground' | 'air';
  /** Hard cap on live instances of this kind. */
  readonly maxCount: number;
  /** Expected instances per 100 m² of corridor at full difficulty (before caps). */
  readonly density: number;
  /** Upper bound of `radius / scale`, used to reserve space before the instance exists. */
  readonly baseRadius: number;
  /** Per-instance scale range handed to `create` as `opts.scale`. */
  readonly scale: readonly [number, number];
  /** Minimum surface-to-surface clearance to any other obstacle at spawn time. */
  readonly minSpacing: number;
  /** When true the instance must not intersect the row's guaranteed clear gap. */
  readonly respectsGap: boolean;
  /** Height band for `air` spawns (metres above ground). */
  readonly altitude?: readonly [number, number];
}

export interface SpawnableFactory extends ObstacleFactory {
  readonly profile: SpawnProfile;
  /** Frees the GPU resources shared by every instance this factory produced. */
  dispose?(): void;
}

export const DEFAULT_PROFILE: SpawnProfile = {
  layer: 'ground',
  maxCount: 30,
  density: 0.4,
  baseRadius: 1.5,
  scale: [1, 1],
  minSpacing: 1,
  respectsGap: true,
};

export function profileOf(factory: ObstacleFactory): SpawnProfile {
  return 'profile' in factory && factory.profile !== undefined
    ? (factory as SpawnableFactory).profile
    : DEFAULT_PROFILE;
}
