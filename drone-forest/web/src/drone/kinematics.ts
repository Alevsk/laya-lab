/**
 * Flight model. Kinematic, not dynamic: an action names a TARGET lateral velocity, vertical
 * velocity and speed multiplier, and the actual velocity relaxes toward it with a first-order
 * exponential lag. That gives the drone visible inertia (an engine that flips actions every
 * tick still gets a smooth, physically plausible path) while keeping the response to a
 * decision fully predictable, which matters when comparing engines.
 *
 * Axes: the corridor runs along -Z, so cruise velocity is (0, 0, -cruise). Lateral is x,
 * vertical is y. bank_left → -x, bank_right → +x, climb → +y, descend → -y.
 */
import type * as THREE from 'three';
import type { ActionName } from '../core/types';

export interface FlightParams {
  cruiseSpeed: number;
  lateralSpeed: number;
  verticalSpeed: number;
  brakeMultiplier: number;
  /** hard ceiling on commanded forward speed, m/s */
  maxSpeed: number;
  /** time constant of the exponential lag, seconds */
  tau: number;
}

export const DEFAULT_FLIGHT: FlightParams = {
  cruiseSpeed: 12,
  lateralSpeed: 7,
  verticalSpeed: 5,
  brakeMultiplier: 0.4,
  maxSpeed: 18,
  tau: 0.25,
};

export interface FlightTargets {
  vx: number;
  vy: number;
  speedMultiplier: number;
  /** absolute forward speed the engine asked for, m/s; null = cruise */
  speed: number | null;
}

export function targetsFor(action: ActionName, p: FlightParams = DEFAULT_FLIGHT): FlightTargets {
  switch (action) {
    case 'bank_left':
      return { vx: -p.lateralSpeed, vy: 0, speedMultiplier: 1, speed: null };
    case 'bank_right':
      return { vx: p.lateralSpeed, vy: 0, speedMultiplier: 1, speed: null };
    case 'climb':
      return { vx: 0, vy: p.verticalSpeed, speedMultiplier: 1, speed: null };
    case 'descend':
      return { vx: 0, vy: -p.verticalSpeed, speedMultiplier: 1, speed: null };
    case 'brake':
      return { vx: 0, vy: 0, speedMultiplier: p.brakeMultiplier, speed: null };
    case 'forward':
      return { vx: 0, vy: 0, speedMultiplier: 1, speed: null };
  }
}

/** Fraction of the remaining gap closed in `dt` for a first-order lag with time constant `tau`. */
export function lagFactor(dt: number, tau: number): number {
  return 1 - Math.exp(-dt / tau);
}

/** Relax `velocity` toward the targets in place. Returns the same vector. */
export function relaxVelocity(
  velocity: THREE.Vector3,
  targets: FlightTargets,
  dt: number,
  p: FlightParams = DEFAULT_FLIGHT,
): THREE.Vector3 {
  const k = lagFactor(dt, p.tau);
  const base = Math.min(p.maxSpeed, Math.max(0, targets.speed ?? p.cruiseSpeed));
  const vzTarget = -base * targets.speedMultiplier;
  velocity.x += (targets.vx - velocity.x) * k;
  velocity.y += (targets.vy - velocity.y) * k;
  velocity.z += (vzTarget - velocity.z) * k;
  return velocity;
}
