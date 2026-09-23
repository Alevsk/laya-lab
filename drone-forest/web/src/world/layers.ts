/**
 * Render layers. Everything that is NOT an obstacle (terrain, scenery treeline, particles,
 * the drone itself) lives on SCENERY_LAYER. A default `THREE.Raycaster` only tests layer 0,
 * so a sensor implementation that raycasts against the scene sees obstacles and nothing else,
 * and the sensor "ground" stays the analytic y = 0 plane even though the terrain has bumps.
 * The main camera enables both layers so everything still renders (and casts shadows).
 */
import type * as THREE from 'three';

export const OBSTACLE_LAYER = 0;
export const SCENERY_LAYER = 1;

/** Move an object and all its descendants onto the scenery layer only. */
export function markScenery(root: THREE.Object3D): void {
  root.traverse((o) => {
    o.layers.set(SCENERY_LAYER);
  });
}
