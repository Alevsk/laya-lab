/**
 * Trees: a conifer (stacked cones) or a broadleaf (jittered icosahedron canopy) on a trunk.
 * All geometry and materials are created ONCE per factory and shared by every instance, so a
 * forest of 150 trees is 150 cheap Groups over ~7 GPU buffers; `dispose()` frees them.
 *
 * Each Group's origin is the canopy centre, which is also the obstacle's bounding-sphere
 * centre (radius = canopy radius × scale). The spawn anchor is the ground contact point, so
 * the factory lifts the centre by the trunk + half canopy height, scaled.
 */
import * as THREE from 'three';
import type { Obstacle, SpawnOptions, WorldContext } from '../../core/types';
import { createRng } from '../../core/rng';
import { BaseObstacle } from '../base';
import { STATIC_BEHAVIOR } from '../behaviors/static';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import type { SpawnProfile, SpawnableFactory } from '../profile';

export interface FactoryOptions {
  corridor?: CorridorFrame;
}

const TRUNK_HEIGHT = 2.4;
/** The lowest cone tier starts slightly below the trunk top so the join is hidden. */
const CONIFER_CANOPY_BASE = TRUNK_HEIGHT - 0.3;
const CONIFER_TIERS: ReadonlyArray<readonly [radius: number, height: number, lift: number]> = [
  [1.5, 2.3, 0],
  [1.15, 2.0, 1.4],
  [0.8, 1.7, 2.6],
];
const CONIFER_CANOPY_HEIGHT = 4.3;
const CONIFER_RADIUS = 1.5;
const BROADLEAF_RADIUS = 1.9;
const CONIFER_SHARE = 0.55;

interface Variant {
  readonly radius: number;
  readonly centreHeight: number;
  readonly build: (materialIndex: number) => THREE.Group;
}

export class TreeFactory implements SpawnableFactory {
  readonly kind = 'tree' as const;
  readonly profile: SpawnProfile = {
    layer: 'ground',
    maxCount: 150,
    density: 2.2,
    baseRadius: BROADLEAF_RADIUS,
    scale: [0.7, 1.6],
    minSpacing: 1.2,
    respectsGap: true,
  };

  private readonly corridor: CorridorFrame;
  private readonly trunkGeometry: THREE.CylinderGeometry;
  private readonly tierGeometries: THREE.ConeGeometry[];
  private readonly canopyGeometry: THREE.IcosahedronGeometry;
  private readonly trunkMaterial: THREE.MeshStandardMaterial;
  private readonly needleMaterials: THREE.MeshStandardMaterial[];
  private readonly leafMaterials: THREE.MeshStandardMaterial[];
  private readonly conifer: Variant;
  private readonly broadleaf: Variant;

  constructor(options: FactoryOptions = {}) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    this.trunkGeometry = new THREE.CylinderGeometry(0.16, 0.3, TRUNK_HEIGHT, 7);
    this.tierGeometries = CONIFER_TIERS.map(([r, h]) => new THREE.ConeGeometry(r, h, 8));
    this.canopyGeometry = new THREE.IcosahedronGeometry(BROADLEAF_RADIUS, 1);
    jitterVertices(this.canopyGeometry, 0.22, createRng(0x7a33));
    this.canopyGeometry.scale(1, 0.85, 1);
    this.canopyGeometry.computeVertexNormals();

    this.trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x5a3f27, roughness: 0.95, flatShading: true });
    this.needleMaterials = [0x1f4a22, 0x2a5d2a, 0x18401c].map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.9, flatShading: true }),
    );
    this.leafMaterials = [0x4a7a2a, 0x6a8f2e, 0x3f6b25].map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.85, flatShading: true }),
    );

    this.conifer = {
      radius: CONIFER_RADIUS,
      centreHeight: CONIFER_CANOPY_BASE + CONIFER_CANOPY_HEIGHT / 2,
      build: (mi) => {
        const group = this.trunkGroup(this.conifer.centreHeight);
        const material = this.needleMaterials[mi % this.needleMaterials.length]!;
        this.tierGeometries.forEach((geometry, i) => {
          const [, height, lift] = CONIFER_TIERS[i]!;
          const tier = new THREE.Mesh(geometry, material);
          tier.position.y = CONIFER_CANOPY_BASE + lift + height / 2 - this.conifer.centreHeight;
          tier.castShadow = true;
          group.add(tier);
        });
        return group;
      },
    };
    this.broadleaf = {
      radius: BROADLEAF_RADIUS,
      centreHeight: TRUNK_HEIGHT + BROADLEAF_RADIUS * 0.8,
      build: (mi) => {
        const group = this.trunkGroup(this.broadleaf.centreHeight);
        const canopy = new THREE.Mesh(this.canopyGeometry, this.leafMaterials[mi % this.leafMaterials.length]!);
        canopy.castShadow = true;
        group.add(canopy);
        return group;
      },
    };
  }

  create(ctx: WorldContext, opts: SpawnOptions): Obstacle {
    const variant = ctx.rng.next() < CONIFER_SHARE ? this.conifer : this.broadleaf;
    const scale = opts.scale ?? ctx.rng.range(this.profile.scale[0], this.profile.scale[1]);
    const group = variant.build(Math.floor(ctx.rng.next() * 3));
    group.scale.setScalar(scale);
    group.rotateOnAxis(this.corridor.up, ctx.rng.range(0, 2 * Math.PI));
    const position = opts.position.clone().addScaledVector(this.corridor.up, variant.centreHeight * scale);
    return new BaseObstacle({
      kind: this.kind,
      object3d: group,
      radius: variant.radius * scale,
      position,
      behavior: STATIC_BEHAVIOR,
      corridor: this.corridor,
      scene: ctx.scene,
    });
  }

  dispose(): void {
    this.trunkGeometry.dispose();
    this.tierGeometries.forEach((g) => g.dispose());
    this.canopyGeometry.dispose();
    this.trunkMaterial.dispose();
    this.needleMaterials.forEach((m) => m.dispose());
    this.leafMaterials.forEach((m) => m.dispose());
  }

  private trunkGroup(centreHeight: number): THREE.Group {
    const group = new THREE.Group();
    const trunk = new THREE.Mesh(this.trunkGeometry, this.trunkMaterial);
    trunk.position.y = TRUNK_HEIGHT / 2 - centreHeight;
    trunk.castShadow = true;
    trunk.receiveShadow = true;
    group.add(trunk);
    return group;
  }
}

/** Displace every vertex along its own direction by ±amount so a primitive stops looking like one. */
export function jitterVertices(geometry: THREE.BufferGeometry, amount: number, rng: { next(): number }): void {
  const attr = geometry.getAttribute('position') as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  for (let i = 0; i < attr.count; i++) {
    v.fromBufferAttribute(attr, i);
    const k = 1 + (rng.next() * 2 - 1) * amount;
    attr.setXYZ(i, v.x * k, v.y * k, v.z * k);
  }
  attr.needsUpdate = true;
}
