/**
 * Pure ballistics for thrown projectiles: given a launch point, a target, a launch speed and
 * gravity, the launch velocity that passes through the target on the LOW arc; and the
 * lead-predicted aim point for a moving target. No scene, no rng, no side effects, so the
 * ogre's aiming can be unit-checked on its own (`ballistics.check.ts`).
 *
 * Gravity acts along -`up` (world +Y by default). Horizontal is everything perpendicular to it.
 */
import * as THREE from 'three';

export const GRAVITY = 9.81;

const UP = new THREE.Vector3(0, 1, 0);

export interface LaunchSolution {
  velocity: THREE.Vector3;
  /** seconds until the projectile reaches the target's horizontal distance */
  flightTime: number;
  /** false when the target is out of range at this speed: the velocity is then the max-range shot toward it */
  reachable: boolean;
}

/**
 * Launch velocity of magnitude `speed` from `origin` that hits `target` (low trajectory).
 * Standard closed form: tan θ = (s² − √(s⁴ − g(g·d² + 2·h·s²))) / (g·d).
 */
export function solveLaunch(
  origin: THREE.Vector3,
  target: THREE.Vector3,
  speed: number,
  gravity = GRAVITY,
  up = UP,
): LaunchSolution {
  const delta = target.clone().sub(origin);
  const h = delta.dot(up);
  const horizontal = delta.clone().addScaledVector(up, -h);
  const d = horizontal.length();
  const dir = d > 1e-6 ? horizontal.multiplyScalar(1 / d) : new THREE.Vector3(0, 0, -1);
  const s2 = speed * speed;
  const disc = s2 * s2 - gravity * (gravity * d * d + 2 * h * s2);
  let theta: number;
  let reachable = true;
  if (disc < 0 || d < 1e-6) {
    // out of range: the 45° shot carries furthest; aim it at the target anyway
    theta = Math.PI / 4;
    reachable = d < 1e-6 ? true : false;
  } else {
    theta = Math.atan((s2 - Math.sqrt(disc)) / (gravity * d));
  }
  const velocity = dir.multiplyScalar(speed * Math.cos(theta)).addScaledVector(up, speed * Math.sin(theta));
  const vh = speed * Math.cos(theta);
  return { velocity, flightTime: vh > 1e-6 ? d / vh : 0, reachable };
}

/**
 * Where to aim at a target moving at constant velocity: iterate the intercept time a few
 * times (flight time depends on where the target will be, which depends on flight time).
 * `lead` in [0, 1] blends between aiming at the current position (0) and the full intercept (1).
 */
export function predictAimPoint(
  origin: THREE.Vector3,
  target: THREE.Vector3,
  targetVelocity: THREE.Vector3,
  speed: number,
  lead = 1,
  gravity = GRAVITY,
  up = UP,
): THREE.Vector3 {
  // Fixed-point iteration on the flight time, damped so a target moving away at close to the
  // rock's horizontal speed still converges instead of oscillating; capped so an unreachable
  // target (which the solver answers with the 45° max-range shot) cannot loop forever.
  let t = 0;
  let aim = target.clone();
  for (let i = 0; i < 24; i++) {
    const next = solveLaunch(origin, aim, speed, gravity, up).flightTime;
    const tNew = t + 0.6 * (next - t);
    aim = target.clone().addScaledVector(targetVelocity, tNew * lead);
    if (Math.abs(tNew - t) < 1e-4) {
      t = tNew;
      break;
    }
    t = tNew;
  }
  return aim;
}

/** Position of a projectile launched from `origin` with `velocity` after `t` seconds (no drag). */
export function positionAt(origin: THREE.Vector3, velocity: THREE.Vector3, t: number, gravity = GRAVITY, up = UP): THREE.Vector3 {
  return origin.clone().addScaledVector(velocity, t).addScaledVector(up, -0.5 * gravity * t * t);
}

/** Rotate `v` by a random small angle about a random axis perpendicular to it (aim error). */
export function perturbDirection(v: THREE.Vector3, angleRad: number, rng: { next(): number }): THREE.Vector3 {
  if (angleRad <= 0) return v.clone();
  const axis = new THREE.Vector3(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5);
  axis.sub(v.clone().normalize().multiplyScalar(axis.dot(v.clone().normalize())));
  if (axis.lengthSq() < 1e-9) axis.set(0, 1, 0);
  axis.normalize();
  return v.clone().applyAxisAngle(axis, angleRad);
}
