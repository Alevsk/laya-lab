/**
 * Camera rig. Chase mode: a spring-lagged third-person camera behind and above the drone that
 * drifts opposite to lateral motion and widens its FOV with speed. Nose mode: mounted on the
 * camera pod, inheriting the body's bank so the horizon rolls.
 */
import * as THREE from 'three';
import { DEFAULT_FLIGHT } from './kinematics';
import type { Drone } from './index';

export type CameraMode = 'chase' | 'nose';

export interface CameraRig {
  readonly mode: CameraMode;
  setMode(mode: CameraMode): void;
  toggle(): CameraMode;
  update(dt: number): void;
  /** Snap to the target with no lag (after a reset). */
  snap(): void;
}

const CHASE_OFFSET = new THREE.Vector3(0, 2.4, 6);
const LOOK_AHEAD = new THREE.Vector3(0, 0.2, -10);
const NOSE_OFFSET = new THREE.Vector3(0, 0.08, -0.5);
const POSITION_TAU = 0.2;
const LOOK_TAU = 0.1;
const CHASE_FOV = 62;
const NOSE_FOV = 82;

export function createChaseRig(camera: THREE.PerspectiveCamera, drone: Drone): CameraRig {
  let mode: CameraMode = 'chase';
  const eye = new THREE.Vector3();
  const look = new THREE.Vector3();
  const wantEye = new THREE.Vector3();
  const wantLook = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const q = new THREE.Quaternion();

  const desired = (): void => {
    wantEye.copy(drone.position).add(CHASE_OFFSET);
    wantEye.x -= drone.velocity.x * 0.12;
    wantEye.y += drone.velocity.y * 0.06;
    wantLook.copy(drone.position).add(LOOK_AHEAD);
    wantLook.x += drone.velocity.x * 0.25;
  };

  const snap = (): void => {
    desired();
    eye.copy(wantEye);
    look.copy(wantLook);
  };
  snap();

  return {
    get mode() {
      return mode;
    },
    setMode(m) {
      mode = m;
      camera.fov = m === 'nose' ? NOSE_FOV : CHASE_FOV;
      camera.updateProjectionMatrix();
    },
    toggle() {
      const next: CameraMode = mode === 'chase' ? 'nose' : 'chase';
      camera.fov = next === 'nose' ? NOSE_FOV : CHASE_FOV;
      camera.updateProjectionMatrix();
      mode = next;
      return next;
    },
    snap,
    update(dt) {
      if (mode === 'nose') {
        drone.body.getWorldQuaternion(q);
        tmp.copy(NOSE_OFFSET).applyQuaternion(q);
        camera.position.copy(drone.position).add(tmp);
        camera.quaternion.copy(q);
        return;
      }
      desired();
      const kp = 1 - Math.exp(-dt / POSITION_TAU);
      const kl = 1 - Math.exp(-dt / LOOK_TAU);
      eye.lerp(wantEye, kp);
      look.lerp(wantLook, kl);
      camera.position.copy(eye);
      camera.lookAt(look);
      const kick = THREE.MathUtils.clamp((drone.speed - DEFAULT_FLIGHT.cruiseSpeed) * 0.9, -8, 10);
      const fov = CHASE_FOV + kick;
      if (Math.abs(camera.fov - fov) > 0.05) {
        camera.fov += (fov - camera.fov) * kp;
        camera.updateProjectionMatrix();
      }
    },
  };
}
