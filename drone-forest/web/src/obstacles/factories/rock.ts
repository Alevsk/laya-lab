/**
 * Rocks: low boulders that sit partly buried in the ground. Five jittered dodecahedron
 * variants are built once from a fixed-seed rng (so rock shapes never perturb the world's
 * seeded stream) and shared by every instance.
 */
import * as THREE from 'three';
import type { Obstacle, SpawnOptions, WorldContext } from '../../core/types';
import { createRng } from '../../core/rng';
import { BaseObstacle } from '../base';
import { STATIC_BEHAVIOR } from '../behaviors/static';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import type { SpawnProfile, SpawnableFactory } from '../profile';
import { jitterVertices, type FactoryOptions } from './tree';

const VARIANTS = 5;
const BASE_RADIUS = 1.0;
/** Fraction of the radius that stays above ground. */
const EXPOSED = 0.55;

export class RockFactory implements SpawnableFactory {
  readonly kind = 'rock' as const;
  readonly profile: SpawnProfile = {
    layer: 'ground',
    maxCount: 40,
    density: 0.6,
    baseRadius: BASE_RADIUS,
    scale: [0.6, 1.4],
    minSpacing: 0.8,
    respectsGap: true,
  };

  private readonly corridor: CorridorFrame;
  private readonly geometries: THREE.DodecahedronGeometry[];
  private readonly materials: THREE.MeshStandardMaterial[];

  constructor(options: FactoryOptions = {}) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    const shapeRng = createRng(0x0c4);
    this.geometries = Array.from({ length: VARIANTS }, () => {
      const geometry = new THREE.DodecahedronGeometry(BASE_RADIUS, 0);
      jitterVertices(geometry, 0.28, shapeRng);
      geometry.scale(1.15, 0.7, 1);
      geometry.computeVertexNormals();
      return geometry;
    });
    this.materials = [0x6f6a62, 0x7d7770, 0x5c5852].map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.95, metalness: 0.02, flatShading: true }),
    );
  }

  create(ctx: WorldContext, opts: SpawnOptions): Obstacle {
    const scale = opts.scale ?? ctx.rng.range(this.profile.scale[0], this.profile.scale[1]);
    const mesh = new THREE.Mesh(ctx.rng.pick(this.geometries), ctx.rng.pick(this.materials));
    mesh.scale.setScalar(scale);
    mesh.rotateOnAxis(this.corridor.up, ctx.rng.range(0, 2 * Math.PI));
    mesh.rotateOnAxis(this.corridor.right, ctx.rng.range(-0.25, 0.25));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const radius = BASE_RADIUS * scale;
    const position = opts.position.clone().addScaledVector(this.corridor.up, radius * EXPOSED);
    return new BaseObstacle({
      kind: this.kind,
      object3d: mesh,
      radius,
      position,
      behavior: STATIC_BEHAVIOR,
      corridor: this.corridor,
      scene: ctx.scene,
    });
  }

  dispose(): void {
    this.geometries.forEach((g) => g.dispose());
    this.materials.forEach((m) => m.dispose());
  }
}
