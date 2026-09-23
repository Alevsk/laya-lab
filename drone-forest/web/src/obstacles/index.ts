/**
 * Obstacles: factories (what a kind looks like and how it is placed), behaviours (how it
 * moves), the registry (which kinds exist) and the spawner (where and when they appear).
 * The world depends only on `Spawner`, `Obstacle` and `ObstacleFactory` from `core/types`.
 *
 * Extending — each is ONE new file plus ONE registration line, nothing else changes:
 *
 *   A new kind, say a drone swarm:
 *     1. `factories/drone-swarm.ts`
 *          export class DroneSwarmFactory implements SpawnableFactory {
 *            readonly kind = 'drone_swarm';                       // (see note below)
 *            readonly profile: SpawnProfile = { layer: 'air', maxCount: 8, density: 0.1,
 *              baseRadius: 0.6, scale: [1, 1], minSpacing: 3, respectsGap: false, altitude: [4, 20] };
 *            create(ctx, opts) { ...build a Group, return new BaseObstacle({ ..., behavior }) }
 *            dispose() { ...free shared geometry/materials }
 *          }
 *     2. `registry.ts`: `registerFactory((o) => new DroneSwarmFactory(o));`
 *     The spawner needs no change: placement comes from the profile, not from the kind.
 *     Note: `ObstacleKind` in `core/types.ts` is a closed union mirrored by `server/schemas.py`,
 *     so a genuinely new kind NAME is also one word added to each of those two contract files
 *     (the sensors report it, and the server validates it).
 *
 *   A new behaviour, say an AI flocking policy:
 *     1. `behaviors/flock-ai.ts` implementing `ObstacleBehavior` (shape sketched in
 *        `behaviors/index.ts`).
 *     2. `registry.ts`: `registerFactory((o) => new BirdFactory({ ...o, behavior: (c) => new FlockAiBehavior(c, transport) }));`
 *     Trees, rocks, the spawner and the world are untouched.
 *
 * Axis convention: the corridor runs along -Z with +X right and y = 0 as the ground, matching
 * `world/`; pass `{ corridor: createCorridorFrame(forward) }` to change it.
 */
export { createSpawner, type SpawnerOptions, type ForestSpawner, type RowRecord } from './spawner';
export { registerFactory, defaultFactories, disposeFactories, type FactoryProvider } from './registry';
export { BaseObstacle, EXPIRE_BEHIND_M, type BaseObstacleInit } from './base';
export { type SpawnProfile, type SpawnableFactory, DEFAULT_PROFILE, profileOf } from './profile';
export { createCorridorFrame, DEFAULT_CORRIDOR, type CorridorFrame } from './corridor';
export { StaticBehavior, STATIC_BEHAVIOR, RandomWanderBehavior, type RandomWanderOptions } from './behaviors';
export { TreeFactory, RockFactory, BirdFactory, type FactoryOptions, type BirdFactoryOptions } from './factories';
