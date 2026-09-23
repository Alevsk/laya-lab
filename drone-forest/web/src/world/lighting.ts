/**
 * Hemisphere fill + a shadow-casting sun. The sun's shadow frustum is a 150 m box that
 * travels with the drone, so shadow texels are spent only where the camera looks.
 */
import * as THREE from 'three';

export interface Lighting {
  readonly sun: THREE.DirectionalLight;
  readonly hemi: THREE.HemisphereLight;
  /** Re-centre the shadow frustum around the drone each frame. */
  update(anchor: THREE.Vector3, forward: THREE.Vector3): void;
}

const SHADOW_HALF = 75;
const SUN_DISTANCE = 160;

export function createLighting(scene: THREE.Scene, sunDirection: THREE.Vector3): Lighting {
  const hemi = new THREE.HemisphereLight(0xbcd3ec, 0x3a4a26, 0.5);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xffd9a8, 2.4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = SUN_DISTANCE * 2 + SHADOW_HALF;
  sun.shadow.camera.left = -SHADOW_HALF;
  sun.shadow.camera.right = SHADOW_HALF;
  sun.shadow.camera.top = SHADOW_HALF;
  sun.shadow.camera.bottom = -SHADOW_HALF;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.04;
  sun.shadow.radius = 3;
  scene.add(sun);
  scene.add(sun.target);

  const focus = new THREE.Vector3();

  return {
    sun,
    hemi,
    update(anchor, forward) {
      // Look-ahead so most of the frustum covers what is in front of the camera.
      focus.copy(anchor).addScaledVector(forward, 35);
      focus.y = 0;
      sun.target.position.copy(focus);
      sun.position.copy(focus).addScaledVector(sunDirection, SUN_DISTANCE);
    },
  };
}
