/**
 * On-screen view of the sensor fan: one line segment per ray from the drone, coloured by what
 * it hit, so a human can see exactly what the engine was told. Toggled by main.ts ('R' key).
 */
import * as THREE from 'three';

import { RAY_SPEC } from '../core/rays';
import type { ObstacleKind, SensorFrame } from '../core/types';
import { headingToBasis, rayDirection } from './raycast';
import type { Basis } from './raycast';

export const RAY_HIT_COLORS: Record<ObstacleKind | 'clear', number> = {
  clear: 0x3ddc84,
  tree: 0xff8c42,
  rock: 0xb0b8c0,
  bird: 0xffe94d,
  ground: 0x4aa3ff,
  wall: 0xe040fb,
};

export interface RayDebug {
  readonly object3d: THREE.LineSegments;
  update(frame: SensorFrame, basis?: Basis): void;
  setVisible(visible: boolean): void;
  toggle(): boolean;
  isVisible(): boolean;
  dispose(): void;
}

export function createRayDebug(scene: THREE.Scene, opts: { basis?: (headingDeg: number) => Basis } = {}): RayDebug {
  const toBasis = opts.basis ?? headingToBasis;
  const n = RAY_SPEC.length;
  const positions = new Float32Array(n * 2 * 3);
  const colors = new Float32Array(n * 2 * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthTest: false });
  const lines = new THREE.LineSegments(geometry, material);
  lines.frustumCulled = false;
  lines.renderOrder = 999;
  lines.visible = false;
  scene.add(lines);

  const color = new THREE.Color();

  const update = (frame: SensorFrame, basis?: Basis): void => {
    if (!lines.visible) return;
    const b = basis ?? toBasis(frame.drone.heading_deg);
    const o = frame.drone.position;
    for (let i = 0; i < n; i++) {
      const ray = frame.rays[i];
      if (!ray) continue;
      const d = rayDirection(b, ray.bearing_deg, ray.elevation_deg);
      const k = i * 6;
      positions[k] = o.x;
      positions[k + 1] = o.y;
      positions[k + 2] = o.z;
      positions[k + 3] = o.x + d.x * ray.distance;
      positions[k + 4] = o.y + d.y * ray.distance;
      positions[k + 5] = o.z + d.z * ray.distance;
      color.setHex(RAY_HIT_COLORS[ray.hit ?? 'clear']);
      colors[k] = color.r;
      colors[k + 1] = color.g;
      colors[k + 2] = color.b;
      colors[k + 3] = color.r;
      colors[k + 4] = color.g;
      colors[k + 5] = color.b;
    }
    geometry.getAttribute('position').needsUpdate = true;
    geometry.getAttribute('color').needsUpdate = true;
  };

  return {
    object3d: lines,
    update,
    setVisible: (visible) => {
      lines.visible = visible;
    },
    toggle: () => {
      lines.visible = !lines.visible;
      return lines.visible;
    },
    isVisible: () => lines.visible,
    dispose: () => {
      scene.remove(lines);
      geometry.dispose();
      material.dispose();
    },
  };
}
