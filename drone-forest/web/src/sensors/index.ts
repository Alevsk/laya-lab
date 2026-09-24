/**
 * Sensors: turns the world + drone into the `SensorFrame` an engine decides from.
 *
 * The world and drone are consumed through structural types (position/velocity/heading and an
 * obstacle list) so this module depends on `core/types` only. Rays are cast analytically
 * against obstacle bounding spheres, the ground plane, the altitude ceiling and the corridor
 * sides — see `raycast.ts` for the coordinate convention, and `debug.ts` for the on-screen
 * ray view. Cost is linear in obstacles with no allocation per obstacle.
 */
import type * as THREE from 'three';

import { RAY_MAX_RANGE } from '../core/rays';
import { PROTOCOL_VERSION } from '../core/types';
import type { ActionName, Obstacle, SensorFrame, WorldContext } from '../core/types';
import { castFan, headingToBasis, mostUrgentThreat, nearestObstacle } from './raycast';
import type { Basis } from './raycast';

export { castFan, headingToBasis, mostUrgentThreat, nearestObstacle, rayDirection, raySphere, rayPlaneX, rayPlaneY } from './raycast';
export type { Basis, SphereLike } from './raycast';
export { createRayDebug, RAY_HIT_COLORS } from './debug';
export type { RayDebug } from './debug';

export interface SensorWorld {
  readonly obstacles: readonly Obstacle[];
  readonly ctx: WorldContext;
}

export interface SensorDrone {
  readonly position: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  readonly headingDeg: number;
}

export interface SensorOptions {
  /** Heading → (forward, right, up). Defaults to the convention documented in raycast.ts. */
  basis?: (headingDeg: number) => Basis;
  maxRange?: number;
  /** report the corridor sides (x = ±half_width) as 'wall' hits; default true */
  sideWalls?: boolean;
}

export interface Sensors {
  /** Build the frame for this instant. `lastAction` is what the drone was doing when it was taken. */
  frame(frameId: number, t: number, lastAction: ActionName | null): SensorFrame;
  /** The most recent frame built, for the HUD and the ray debug view. */
  lastFrame(): SensorFrame | null;
  /** Basis of the last frame (or the drone's current heading when none was built yet). */
  basis(): Basis;
}

export function createSensors(world: SensorWorld, drone: SensorDrone, opts: SensorOptions = {}): Sensors {
  const toBasis = opts.basis ?? headingToBasis;
  const maxRange = opts.maxRange ?? RAY_MAX_RANGE;
  const sideWalls = opts.sideWalls ?? true;
  let last: SensorFrame | null = null;
  let lastBasis: Basis = toBasis(drone.headingDeg);

  const frame = (frameId: number, t: number, lastAction: ActionName | null): SensorFrame => {
    const p = drone.position;
    const v = drone.velocity;
    const origin = { x: p.x, y: p.y, z: p.z };
    const velocity = { x: v.x, y: v.y, z: v.z };
    const heading = ((drone.headingDeg % 360) + 360) % 360;
    const basis = toBasis(heading);
    const bounds = {
      altitude_min: world.ctx.bounds.altitude_min,
      altitude_max: world.ctx.bounds.altitude_max,
      speed_max: world.ctx.bounds.speed_max,
    };

    last = {
      protocol: PROTOCOL_VERSION,
      frame_id: frameId,
      t,
      drone: {
        position: origin,
        velocity,
        heading_deg: heading,
        altitude: p.y,
        speed: Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),
      },
      rays: castFan(origin, basis, world.obstacles, bounds, maxRange, sideWalls ? world.ctx.bounds.half_width : undefined),
      nearest: nearestObstacle(origin, velocity, basis, world.obstacles),
      threat: mostUrgentThreat(origin, velocity, basis, world.obstacles),
      bounds,
      last_action: lastAction,
    };
    lastBasis = basis;
    return last;
  };

  return {
    frame,
    lastFrame: () => last,
    basis: () => lastBasis,
  };
}
