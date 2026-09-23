/**
 * The drone: model + kinematics + collision. It knows nothing about who decides — it is handed
 * an ActionName each step and moves accordingly. Collision is sphere-vs-sphere against the
 * `Obstacle` contract (each obstacle publishes a bounding-sphere radius), so a new obstacle
 * kind needs no change here.
 */
import * as THREE from 'three';
import type { ActionName, CollisionResult, Obstacle, WorldContext } from '../core/types';
import { DEFAULT_FLIGHT, relaxVelocity, targetsFor, type FlightParams, type FlightTargets } from './kinematics';
import { buildDroneModel } from './model';

export { createChaseRig, type CameraMode, type CameraRig } from './camera';
export { DEFAULT_FLIGHT, targetsFor, relaxVelocity, type FlightParams, type FlightTargets } from './kinematics';

export interface Drone {
  readonly object3d: THREE.Object3D;
  /** The tilting sub-group (bank / pitch); the camera rig mounts the nose cam here. */
  readonly body: THREE.Object3D;
  /** Live reference — the world and sensors read it, never copy it lazily. */
  readonly position: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  /** Compass heading in degrees. The corridor is straight, so this is 0 (looking down -Z). */
  readonly headingDeg: number;
  readonly speed: number;
  readonly action: ActionName;
  readonly radius: number;
  readonly nearMissRadius: number;
  /** Set the flight targets for `a` and relax the velocity toward them over `dt`. */
  applyAction(a: ActionName, dt: number): void;
  /** Engine-recommended forward speed in m/s; null returns to cruise. Takes effect through the same lag as actions. */
  setTargetSpeed(mps: number | null): void;
  readonly targetSpeed: number | null;
  /** Integrate position, clamp to the corridor bounds, animate rotors, lights and tilt. */
  update(dt: number, ctx: WorldContext): void;
  collide(obstacles: readonly Obstacle[]): CollisionResult;
  reset(): void;
  dispose(): void;
}

export interface DroneOptions {
  flight?: Partial<FlightParams>;
  radius?: number;
  nearMissRadius?: number;
  start?: THREE.Vector3;
}

export const DRONE_RADIUS = 0.6;
export const NEAR_MISS_RADIUS = 2.0;
export const START_POSITION = new THREE.Vector3(0, 8, 0);

const MAX_ROLL = THREE.MathUtils.degToRad(30);
const ROTOR_RATE = 58;
const TILT_TAU = 0.12;

export function createDrone(scene: THREE.Scene, opts: DroneOptions = {}): Drone {
  const flight: FlightParams = { ...DEFAULT_FLIGHT, ...opts.flight };
  const radius = opts.radius ?? DRONE_RADIUS;
  const nearMissRadius = opts.nearMissRadius ?? NEAR_MISS_RADIUS;
  const start = (opts.start ?? START_POSITION).clone();

  const model = buildDroneModel();
  scene.add(model.root);

  const position = model.root.position;
  const velocity = new THREE.Vector3(0, 0, -flight.cruiseSpeed);
  let action: ActionName = 'forward';
  let targetSpeed: number | null = null;
  let targets: FlightTargets = targetsFor(action, flight);
  let clock = 0;
  const tilt = new THREE.Euler();

  const reset = (): void => {
    position.copy(start);
    velocity.set(0, 0, -flight.cruiseSpeed);
    action = 'forward';
    targetSpeed = null;
    targets = targetsFor(action, flight);
    tilt.set(0, 0, 0);
    model.body.rotation.set(0, 0, 0);
  };
  reset();

  const animate = (dt: number): void => {
    clock += dt;
    const k = 1 - Math.exp(-dt / TILT_TAU);
    const roll = THREE.MathUtils.clamp(-velocity.x / flight.lateralSpeed, -1, 1) * MAX_ROLL;
    const pitch = -(0.06 + 0.1 * (-velocity.z / flight.cruiseSpeed)) - velocity.y * 0.03;
    tilt.z += (roll - tilt.z) * k;
    tilt.x += (pitch - tilt.x) * k;
    tilt.y = tilt.z * 0.15;
    model.body.rotation.copy(tilt);

    model.rotors.forEach((r, i) => {
      r.rotation.y += (i % 2 === 0 ? 1 : -1) * ROTOR_RATE * dt;
    });

    const blink = (clock % 1) < 0.65 ? 3.5 : 0;
    model.navLeft.emissiveIntensity = blink;
    model.navRight.emissiveIntensity = blink;
    const phase = clock % 1.6;
    model.strobe.emissiveIntensity = phase < 0.06 || (phase > 0.16 && phase < 0.22) ? 8 : 0;
  };

  return {
    object3d: model.root,
    body: model.body,
    position,
    velocity,
    headingDeg: 0,
    radius,
    nearMissRadius,
    get speed() {
      return velocity.length();
    },
    get action() {
      return action;
    },
    applyAction(a, dt) {
      if (a !== action) {
        action = a;
        targets = { ...targetsFor(a, flight), speed: targetSpeed };
      }
      relaxVelocity(velocity, targets, dt, flight);
    },
    setTargetSpeed(mps) {
      if (mps === targetSpeed) return;
      targetSpeed = mps;
      targets = { ...targets, speed: mps };
    },
    get targetSpeed() {
      return targetSpeed;
    },
    update(dt, ctx) {
      position.addScaledVector(velocity, dt);
      const { altitude_min, altitude_max, half_width } = ctx.bounds;
      if (position.y < altitude_min) {
        position.y = altitude_min;
        if (velocity.y < 0) velocity.y = 0;
      } else if (position.y > altitude_max) {
        position.y = altitude_max;
        if (velocity.y > 0) velocity.y = 0;
      }
      if (position.x < -half_width) {
        position.x = -half_width;
        if (velocity.x < 0) velocity.x = 0;
      } else if (position.x > half_width) {
        position.x = half_width;
        if (velocity.x > 0) velocity.x = 0;
      }
      animate(dt);
    },
    collide(obstacles) {
      let hit: Obstacle | null = null;
      let nearMiss: Obstacle | null = null;
      let hitD = Infinity;
      let nearD = Infinity;
      for (const o of obstacles) {
        const d = position.distanceTo(o.position);
        if (d < radius + o.radius) {
          if (d < hitD) {
            hitD = d;
            hit = o;
          }
        } else if (d < nearMissRadius + o.radius && d < nearD) {
          nearD = d;
          nearMiss = o;
        }
      }
      return { hit, nearMiss: hit ? null : nearMiss };
    },
    reset,
    dispose() {
      scene.remove(model.root);
      model.dispose();
    },
  };
}
