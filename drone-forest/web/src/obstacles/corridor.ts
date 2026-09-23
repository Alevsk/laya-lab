/**
 * The corridor frame: which world axis the drone flies along, which is lateral, which is up.
 * Every obstacle-side computation ("how far ahead is this row", "is the bird still inside the
 * corridor") goes through this object so the axis convention lives in exactly one place.
 *
 * Default matches the world module: the drone flies toward -Z, +X is right, +Y is up, y = 0 is
 * the collision ground. A world that runs its corridor along another axis passes
 * `createCorridorFrame(forward)` to `createSpawner` / `defaultFactories`.
 */
import * as THREE from 'three';

export interface CorridorFrame {
  readonly forward: THREE.Vector3;
  readonly right: THREE.Vector3;
  readonly up: THREE.Vector3;
  /** Signed progress along the corridor (metres flown). */
  along(p: THREE.Vector3): number;
  /** Signed lateral offset from the corridor centre line (right is positive). */
  across(p: THREE.Vector3): number;
  /** Writes `out = along·forward + across·right + height·up`. */
  compose(along: number, across: number, height: number, out: THREE.Vector3): THREE.Vector3;
}

export function createCorridorFrame(forward: THREE.Vector3, up = new THREE.Vector3(0, 1, 0)): CorridorFrame {
  const f = forward.clone().normalize();
  const u = up.clone().normalize();
  const r = new THREE.Vector3().crossVectors(f, u).normalize();
  return {
    forward: f,
    right: r,
    up: u,
    along: (p) => p.dot(f),
    across: (p) => p.dot(r),
    compose: (along, across, height, out) =>
      out.set(0, 0, 0).addScaledVector(f, along).addScaledVector(r, across).addScaledVector(u, height),
  };
}

export const DEFAULT_CORRIDOR: CorridorFrame = createCorridorFrame(new THREE.Vector3(0, 0, -1));
