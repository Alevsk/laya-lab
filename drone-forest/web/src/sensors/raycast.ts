/**
 * Analytic ray casting for the sensor fan. Pure math with no Three.js dependency so it runs in
 * node for tests and stays cheap enough for every simulation step (11 rays against a few
 * hundred bounding spheres is a few thousand multiply-adds, no allocation on the hot path).
 *
 * Coordinate convention (the debug view uses the same one; `createSensors` accepts a `basis`
 * override if the drone module chooses another): world up is +Y and the ground is the plane
 * y = 0. Heading 0° looks down -Z, heading 90° looks down +X — compass headings turn clockwise
 * when seen from above. Ray bearing is right(+)/left(-) of the heading, elevation up(+)/down(-).
 */
import { RAY_MAX_RANGE, RAY_SPEC } from '../core/rays';
import type { Bounds, Nearest, ObstacleKind, Ray, Vec3 } from '../core/types';

export interface Basis { forward: Vec3; right: Vec3; up: Vec3 }

/** What the caster needs from an obstacle. `Obstacle` from core/types satisfies it structurally. */
export interface SphereLike {
  readonly kind: ObstacleKind;
  readonly radius: number;
  readonly position: Vec3;
  readonly velocity?: Vec3;
}

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

export function headingToBasis(headingDeg: number): Basis {
  const h = headingDeg * DEG;
  const s = Math.sin(h);
  const c = Math.cos(h);
  return {
    forward: { x: s, y: 0, z: -c },
    right: { x: c, y: 0, z: s },
    up: { x: 0, y: 1, z: 0 },
  };
}

/** Unit direction for a ray at (bearing, elevation) relative to `basis`. */
export function rayDirection(basis: Basis, bearingDeg: number, elevationDeg: number): Vec3 {
  const b = bearingDeg * DEG;
  const e = elevationDeg * DEG;
  const ce = Math.cos(e);
  const fx = ce * Math.cos(b);
  const rx = ce * Math.sin(b);
  const ux = Math.sin(e);
  return {
    x: basis.forward.x * fx + basis.right.x * rx + basis.up.x * ux,
    y: basis.forward.y * fx + basis.right.y * rx + basis.up.y * ux,
    z: basis.forward.z * fx + basis.right.z * rx + basis.up.z * ux,
  };
}

/**
 * Distance along the ray (unit `dir`) to the first intersection with the sphere, or +Infinity.
 * An origin inside the sphere reports 0 — the drone is already touching it.
 */
export function raySphere(origin: Vec3, dir: Vec3, center: Vec3, radius: number): number {
  const mx = origin.x - center.x;
  const my = origin.y - center.y;
  const mz = origin.z - center.z;
  const b = mx * dir.x + my * dir.y + mz * dir.z;
  const c = mx * mx + my * my + mz * mz - radius * radius;
  if (c > 0 && b > 0) return Infinity; // outside and pointing away
  const disc = b * b - c;
  if (disc < 0) return Infinity;
  const t = -b - Math.sqrt(disc);
  return t < 0 ? 0 : t;
}

/** Distance along the ray to the horizontal plane y = planeY, or +Infinity when parallel / behind. */
export function rayPlaneY(origin: Vec3, dir: Vec3, planeY: number): number {
  if (Math.abs(dir.y) < 1e-9) return Infinity;
  const t = (planeY - origin.y) / dir.y;
  return t < 0 ? Infinity : t;
}

// Scratch buffers for the culled sphere set; grown on demand, never shrunk.
let cullX = new Float64Array(256);
let cullY = new Float64Array(256);
let cullZ = new Float64Array(256);
let cullR = new Float64Array(256);
let cullKind: ObstacleKind[] = [];

function grow(n: number): void {
  if (cullX.length >= n) return;
  const cap = Math.max(n, cullX.length * 2);
  cullX = new Float64Array(cap);
  cullY = new Float64Array(cap);
  cullZ = new Float64Array(cap);
  cullR = new Float64Array(cap);
}

/** Distance along the ray to the vertical plane x = planeX, or +Infinity when parallel / behind. */
export function rayPlaneX(origin: Vec3, dir: Vec3, planeX: number): number {
  if (Math.abs(dir.x) < 1e-9) return Infinity;
  const t = (planeX - origin.x) / dir.x;
  return t < 0 ? Infinity : t;
}

/**
 * Cast the whole RAY_SPEC fan from `origin`. Each ray reports the nearest of: obstacle
 * bounding spheres ('tree' | 'rock' | 'bird'), the ground plane y = 0 ('ground'), the altitude
 * ceiling ('wall') and, when `halfWidth` is given, the corridor sides x = ±halfWidth ('wall' —
 * the drone is clamped there, so they are as real as a tree), clamped to `maxRange`. Rays are
 * returned in RAY_SPEC order.
 */
export function castFan(
  origin: Vec3,
  basis: Basis,
  spheres: readonly SphereLike[],
  bounds: Bounds,
  maxRange: number = RAY_MAX_RANGE,
  halfWidth?: number,
): Ray[] {
  grow(spheres.length);
  let n = 0;
  for (const s of spheres) {
    const dx = s.position.x - origin.x;
    const dy = s.position.y - origin.y;
    const dz = s.position.z - origin.z;
    const reach = maxRange + s.radius;
    if (dx * dx + dy * dy + dz * dz > reach * reach) continue;
    cullX[n] = s.position.x;
    cullY[n] = s.position.y;
    cullZ[n] = s.position.z;
    cullR[n] = s.radius;
    cullKind[n] = s.kind;
    n++;
  }

  const out: Ray[] = new Array(RAY_SPEC.length);
  for (let i = 0; i < RAY_SPEC.length; i++) {
    const spec = RAY_SPEC[i]!;
    const dir = rayDirection(basis, spec[1], spec[2]);
    let best = maxRange;
    let hit: ObstacleKind | null = null;

    for (let j = 0; j < n; j++) {
      const mx = origin.x - cullX[j]!;
      const my = origin.y - cullY[j]!;
      const mz = origin.z - cullZ[j]!;
      const b = mx * dir.x + my * dir.y + mz * dir.z;
      const r = cullR[j]!;
      const c = mx * mx + my * my + mz * mz - r * r;
      if (c > 0 && b > 0) continue;
      const disc = b * b - c;
      if (disc < 0) continue;
      let t = -b - Math.sqrt(disc);
      if (t < 0) t = 0;
      if (t < best) {
        best = t;
        hit = cullKind[j]!;
      }
    }

    const tGround = rayPlaneY(origin, dir, 0);
    if (tGround < best) {
      best = tGround;
      hit = 'ground';
    }
    const tCeiling = rayPlaneY(origin, dir, bounds.altitude_max);
    if (tCeiling < best) {
      best = tCeiling;
      hit = 'wall';
    }
    if (halfWidth !== undefined) {
      const tSide = Math.min(rayPlaneX(origin, dir, halfWidth), rayPlaneX(origin, dir, -halfWidth));
      if (tSide < best) {
        best = tSide;
        hit = 'wall';
      }
    }

    out[i] = {
      name: spec[0],
      bearing_deg: spec[1],
      elevation_deg: spec[2],
      distance: best,
      hit,
      max_range: maxRange,
    };
  }
  return out;
}

/**
 * The closest obstacle by surface distance (centre distance minus radius, floored at 0), with
 * its bearing/elevation relative to `basis` and the closing speed along the line of sight.
 * Ground and ceiling are deliberately not candidates: the 'down'/'up' rays and the frame's
 * altitude + bounds already carry them, and letting the floor win here would hide the tree.
 */
function relativeClosing(dx: number, dy: number, dz: number, len: number, s: SphereLike, droneVelocity: Vec3): number {
  if (len <= 1e-9) return 0;
  const ov = s.velocity ?? { x: 0, y: 0, z: 0 };
  // d/dt |c - p| = unit(c - p) · (v_obstacle - v_drone); the gap shrinks when that is negative
  return -(dx * (ov.x - droneVelocity.x) + dy * (ov.y - droneVelocity.y) + dz * (ov.z - droneVelocity.z)) / len;
}

function describe(best: SphereLike, bdx: number, bdy: number, bdz: number, gap: number, closing: number, basis: Basis): Nearest {
  const horiz = Math.sqrt(bdx * bdx + bdz * bdz);
  const along = bdx * basis.forward.x + bdy * basis.forward.y + bdz * basis.forward.z;
  const across = bdx * basis.right.x + bdy * basis.right.y + bdz * basis.right.z;
  const bearing = horiz < 1e-9 ? 0 : Math.atan2(across, along) * RAD;
  const elevation = Math.atan2(bdy, horiz) * RAD;
  return { kind: best.kind, distance: Math.max(0, gap), bearing_deg: bearing, elevation_deg: elevation, closing_speed: closing };
}

/** The closest obstacle by surface gap - scenery proximity, whatever it is doing. */
export function nearestObstacle(
  origin: Vec3,
  droneVelocity: Vec3,
  basis: Basis,
  spheres: readonly SphereLike[],
): Nearest | null {
  let best: SphereLike | null = null;
  let bestGap = Infinity;
  let bdx = 0;
  let bdy = 0;
  let bdz = 0;
  let bLen = 0;
  for (const s of spheres) {
    const dx = s.position.x - origin.x;
    const dy = s.position.y - origin.y;
    const dz = s.position.z - origin.z;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const gap = len - s.radius;
    if (gap < bestGap) {
      bestGap = gap;
      best = s;
      bdx = dx;
      bdy = dy;
      bdz = dz;
      bLen = len;
    }
  }
  if (!best) return null;
  return describe(best, bdx, bdy, bdz, bestGap, relativeClosing(bdx, bdy, bdz, bLen, best, droneVelocity), basis);
}

/** Seconds of time-to-collision beyond which a moving object is not reported as a threat. */
export const THREAT_TTC_S = 4;

/**
 * The most urgent MOVING object: among obstacles with their own velocity (thrown rocks, birds),
 * the one with the least time-to-collision (gap / closing speed), provided it is closing and
 * would arrive within THREAT_TTC_S. Static scenery never appears here - that is `nearest`.
 */
export function mostUrgentThreat(
  origin: Vec3,
  droneVelocity: Vec3,
  basis: Basis,
  spheres: readonly SphereLike[],
): Nearest | null {
  let best: SphereLike | null = null;
  let bestTtc = THREAT_TTC_S;
  let bdx = 0;
  let bdy = 0;
  let bdz = 0;
  let bGap = 0;
  let bClosing = 0;
  for (const s of spheres) {
    const ov = s.velocity;
    if (!ov || (ov.x === 0 && ov.y === 0 && ov.z === 0)) continue;
    const dx = s.position.x - origin.x;
    const dy = s.position.y - origin.y;
    const dz = s.position.z - origin.z;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const closing = relativeClosing(dx, dy, dz, len, s, droneVelocity);
    if (closing <= 0) continue;
    const gap = Math.max(0, len - s.radius);
    const ttc = gap / closing;
    if (ttc < bestTtc) {
      bestTtc = ttc;
      best = s;
      bdx = dx;
      bdy = dy;
      bdz = dz;
      bGap = gap;
      bClosing = closing;
    }
  }
  if (!best) return null;
  return describe(best, bdx, bdy, bdz, bGap, bClosing, basis);
}
