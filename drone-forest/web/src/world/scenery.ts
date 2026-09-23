/**
 * Non-interactive dressing: a dense instanced treeline on both sides of the corridor and low
 * undergrowth across the floor. Two draw calls for ~1000 trees, one for the bushes. Placement
 * is a pure function of (row, instance index) so a recycled tile is always dressed the same.
 *
 * Scenery lives on SCENERY_LAYER: it is never an Obstacle, sensors never see it, and the
 * spawner is free to place real obstacles anywhere inside ±half_width.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { markScenery } from './layers';
import { hash2 } from './noise';
import type { Terrain } from './terrain';
import { TILE_COUNT, TILE_LENGTH } from './terrain';

const TREES_PER_TILE = 170;
const BUSHES_PER_TILE = 90;
const TREELINE_GAP = 5;
const TREELINE_DEPTH = 95;
const BUSH_SPREAD_BEYOND = 30;

export interface Scenery {
  readonly group: THREE.Group;
  dispose(): void;
}

function conifer(): { trunk: THREE.BufferGeometry; canopy: THREE.BufferGeometry } {
  const trunk = new THREE.CylinderGeometry(0.22, 0.42, 5.5, 7);
  trunk.translate(0, 2.75, 0);
  const tiers = [
    new THREE.ConeGeometry(2.6, 4.2, 8).translate(0, 4.6, 0),
    new THREE.ConeGeometry(2.0, 3.6, 8).translate(0, 7.0, 0),
    new THREE.ConeGeometry(1.35, 3.0, 8).translate(0, 9.1, 0),
  ];
  const canopy = mergeGeometries(tiers);
  for (const t of tiers) t.dispose();
  return { trunk, canopy };
}

export function createScenery(scene: THREE.Scene, terrain: Terrain, halfWidth: number): Scenery {
  const group = new THREE.Group();
  group.name = 'scenery';
  scene.add(group);

  const { trunk, canopy } = conifer();
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x4b3423, roughness: 0.95 });
  const canopyMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
  const bushGeo = new THREE.IcosahedronGeometry(1, 1);
  bushGeo.translate(0, 0.35, 0);
  const bushMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 });

  const treeCount = TREES_PER_TILE * TILE_COUNT;
  const bushCount = BUSHES_PER_TILE * TILE_COUNT;
  const trunks = new THREE.InstancedMesh(trunk, trunkMat, treeCount);
  const canopies = new THREE.InstancedMesh(canopy, canopyMat, treeCount);
  const bushes = new THREE.InstancedMesh(bushGeo, bushMat, bushCount);
  for (const m of [trunks, canopies, bushes]) {
    m.castShadow = true;
    m.receiveShadow = true;
    m.frustumCulled = false;
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    markScenery(m);
    group.add(m);
  }

  const m4 = new THREE.Matrix4();
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  const color = new THREE.Color();
  const up = new THREE.Vector3(0, 1, 0);

  const dress = (slot: number, row: number): void => {
    const z0 = terrain.rowCenterZ(row) - TILE_LENGTH / 2;
    for (let i = 0; i < TREES_PER_TILE; i++) {
      const idx = slot * TREES_PER_TILE + i;
      const side = hash2(row, i, 1) < 0.5 ? -1 : 1;
      // Bias toward the corridor edge so the treeline reads as a wall from inside.
      const depth = Math.pow(hash2(row, i, 2), 1.6) * TREELINE_DEPTH;
      const x = side * (halfWidth + TREELINE_GAP + depth);
      const z = z0 + hash2(row, i, 3) * TILE_LENGTH;
      const s = 0.85 + hash2(row, i, 4) * 0.95;
      pos.set(x, terrain.heightAt(x, z) - 0.3, z);
      quat.setFromAxisAngle(up, hash2(row, i, 5) * Math.PI * 2);
      scl.set(s, s * (0.9 + hash2(row, i, 6) * 0.4), s);
      m4.compose(pos, quat, scl);
      trunks.setMatrixAt(idx, m4);
      canopies.setMatrixAt(idx, m4);
      color.setHSL(0.27 + hash2(row, i, 7) * 0.08, 0.45, 0.16 + hash2(row, i, 8) * 0.14);
      canopies.setColorAt(idx, color);
    }
    for (let i = 0; i < BUSHES_PER_TILE; i++) {
      const idx = slot * BUSHES_PER_TILE + i;
      const span = halfWidth + BUSH_SPREAD_BEYOND;
      const x = (hash2(row, i, 21) * 2 - 1) * span;
      const z = z0 + hash2(row, i, 22) * TILE_LENGTH;
      const s = 0.5 + hash2(row, i, 23) * 0.7;
      pos.set(x, terrain.heightAt(x, z), z);
      quat.setFromAxisAngle(up, hash2(row, i, 24) * Math.PI * 2);
      scl.set(s * 1.3, s * 0.8, s * 1.3);
      m4.compose(pos, quat, scl);
      bushes.setMatrixAt(idx, m4);
      color.setHSL(0.24 + hash2(row, i, 25) * 0.1, 0.5, 0.14 + hash2(row, i, 26) * 0.1);
      bushes.setColorAt(idx, color);
    }
    trunks.instanceMatrix.needsUpdate = true;
    canopies.instanceMatrix.needsUpdate = true;
    bushes.instanceMatrix.needsUpdate = true;
    if (canopies.instanceColor) canopies.instanceColor.needsUpdate = true;
    if (bushes.instanceColor) bushes.instanceColor.needsUpdate = true;
  };
  terrain.onTileAssigned(dress);

  return {
    group,
    dispose() {
      scene.remove(group);
      trunk.dispose();
      canopy.dispose();
      bushGeo.dispose();
      trunkMat.dispose();
      canopyMat.dispose();
      bushMat.dispose();
      trunks.dispose();
      canopies.dispose();
      bushes.dispose();
    },
  };
}
