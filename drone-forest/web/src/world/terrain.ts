/**
 * Endless ground: a ring of TILE_COUNT ground tiles laid along -Z, recycled as the drone
 * advances. Each tile is a displaced, vertex-coloured plane; heights and colours come from
 * world-coordinate noise, so a tile that is recycled to row r always looks the same as the
 * last time row r was visible. The bumps are purely visual — they never exceed ±0.6 m in the
 * corridor and the collision / sensor ground is the y = 0 plane.
 *
 * Rows: row r covers world z ∈ [-(r+1)·L, -r·L]; the drone flies toward -Z so rows increase.
 */
import * as THREE from 'three';
import { markScenery } from './layers';
import { fbm } from './noise';

export const TILE_LENGTH = 100;
export const TILE_WIDTH = 280;
export const TILE_COUNT = 6;
/** How many tiles are kept behind the drone (the rest extend ahead). */
const TILES_BEHIND = 1;
const SEG_X = 70;
const SEG_Z = 25;

export type TileListener = (slot: number, row: number) => void;

export interface Terrain {
  readonly group: THREE.Group;
  /** Called whenever a slot is (re)assigned to a row; scenery uses it to re-dress the tile. */
  onTileAssigned(listener: TileListener): void;
  /** World-space z of the centre of a row. */
  rowCenterZ(row: number): number;
  /** Visual ground height at a world x/z (not the collision ground). */
  heightAt(x: number, z: number): number;
  update(droneZ: number): void;
  /** Forget all rows so the next update rebuilds around the (reset) drone. */
  reset(): void;
}

const moss = new THREE.Color(0x36591f);
const grass = new THREE.Color(0x5f8127);
const dirt = new THREE.Color(0x5b4730);
const shade = new THREE.Color(0x1f3316);

export function createTerrain(scene: THREE.Scene, halfWidth: number): Terrain {
  const group = new THREE.Group();
  group.name = 'terrain';
  scene.add(group);

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 1,
    metalness: 0,
  });

  const listeners: TileListener[] = [];
  const slotRow: number[] = new Array<number>(TILE_COUNT).fill(-1);
  const meshes: THREE.Mesh[] = [];

  for (let s = 0; s < TILE_COUNT; s++) {
    const geometry = new THREE.PlaneGeometry(TILE_WIDTH, TILE_LENGTH, SEG_X, SEG_Z);
    geometry.rotateX(-Math.PI / 2);
    geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(new Float32Array(geometry.attributes['position']!.count * 3), 3),
    );
    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    markScenery(mesh);
    group.add(mesh);
    meshes.push(mesh);
  }

  const heightAt = (x: number, z: number): number => {
    const micro = (fbm(x * 0.045, z * 0.045, 3, 11) - 0.5) * 1.2;
    const edge = Math.max(0, Math.abs(x) - halfWidth - 12);
    const hills = (fbm(x * 0.008, z * 0.008, 3, 23) - 0.35) * 9 * Math.min(1, edge / 50);
    return micro + Math.max(0, hills);
  };

  const c = new THREE.Color();
  const buildTile = (slot: number, row: number): void => {
    const mesh = meshes[slot]!;
    const centerZ = -(row + 0.5) * TILE_LENGTH;
    mesh.position.set(0, 0, centerZ);
    const pos = mesh.geometry.attributes['position'] as THREE.BufferAttribute;
    const col = mesh.geometry.attributes['color'] as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i);
      const wz = pos.getZ(i) + centerZ;
      const h = heightAt(wx, wz);
      pos.setY(i, h);
      const patch = fbm(wx * 0.07, wz * 0.07, 3, 5);
      const wear = fbm(wx * 0.21, wz * 0.21, 2, 9);
      c.copy(moss).lerp(grass, THREE.MathUtils.smoothstep(patch, 0.45, 0.7));
      c.lerp(dirt, THREE.MathUtils.smoothstep(wear, 0.62, 0.8) * 0.8);
      c.lerp(shade, THREE.MathUtils.clamp(-h * 0.5, 0, 0.5));
      col.setXYZ(i, c.r, c.g, c.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingSphere();
    slotRow[slot] = row;
    for (const l of listeners) l(slot, row);
  };

  const update = (droneZ: number): void => {
    const droneRow = Math.floor(-droneZ / TILE_LENGTH);
    const first = droneRow - TILES_BEHIND;
    const wanted = new Set<number>();
    for (let k = 0; k < TILE_COUNT; k++) wanted.add(first + k);
    const missing = [...wanted].filter((r) => !slotRow.includes(r));
    for (let s = 0; s < TILE_COUNT && missing.length > 0; s++) {
      if (!wanted.has(slotRow[s]!)) buildTile(s, missing.shift()!);
    }
  };

  return {
    group,
    onTileAssigned: (l) => {
      listeners.push(l);
    },
    rowCenterZ: (row) => -(row + 0.5) * TILE_LENGTH,
    heightAt,
    update,
    reset: () => {
      slotRow.fill(-1);
    },
  };
}
