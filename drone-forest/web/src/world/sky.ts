/**
 * Sky dome, sun direction and fog. The sun sits low off the right shoulder, slightly ahead, so
 * trees throw long shadows across the corridor without the camera staring into the glare.
 * Fog is exponential and tuned to the horizon haze so the far end of the corridor dissolves
 * before obstacles pop in.
 */
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { markScenery } from './layers';

export interface SkyRig {
  readonly sky: Sky;
  /** Unit vector pointing FROM the scene TOWARD the sun. */
  readonly sunDirection: THREE.Vector3;
  readonly fog: THREE.FogExp2;
  /** Keep the dome centred on the camera; the corridor is endless. */
  update(cameraPosition: THREE.Vector3): void;
}

export const FOG_COLOR = 0x8ea287;
export const FOG_DENSITY = 0.0055;

const SUN_ELEVATION_DEG = 24;
const SUN_AZIMUTH_DEG = 118;

export function createSky(scene: THREE.Scene): SkyRig {
  const sky = new Sky();
  sky.scale.setScalar(3000);
  markScenery(sky);
  scene.add(sky);

  const u = sky.material.uniforms as Record<string, THREE.IUniform>;
  u['turbidity']!.value = 5;
  u['rayleigh']!.value = 3.0;
  u['mieCoefficient']!.value = 0.004;
  u['mieDirectionalG']!.value = 0.8;

  const sunDirection = new THREE.Vector3().setFromSphericalCoords(
    1,
    THREE.MathUtils.degToRad(90 - SUN_ELEVATION_DEG),
    THREE.MathUtils.degToRad(SUN_AZIMUTH_DEG),
  );
  (u['sunPosition']!.value as THREE.Vector3).copy(sunDirection);

  const fog = new THREE.FogExp2(FOG_COLOR, FOG_DENSITY);
  scene.fog = fog;

  return {
    sky,
    sunDirection,
    fog,
    update(cameraPosition) {
      sky.position.copy(cameraPosition);
    },
  };
}
