/**
 * Thrown rocks. Not spawned by the spawner - an ogre's behaviour asks this factory for one at
 * release time and hands it to `ctx.emit`. Ballistic flight, tumbling, a short rest on the
 * ground, then it expires. Shares one small jittered dodecahedron and one material.
 */
import * as THREE from 'three';
import type { Obstacle, WorldContext } from '../../core/types';
import { createRng } from '../../core/rng';
import { BaseObstacle, type BaseObstacleInit } from '../base';
import { BallisticBehavior } from '../behaviors/ballistic';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import { jitterVertices, type FactoryOptions } from './tree';

export const PROJECTILE_RADIUS = 0.45;
/** Seconds a landed rock stays before it is recycled. */
const REST_S = 1.5;
/** A rock that has somehow flown this long is gone regardless. */
const MAX_AGE_S = 12;

class ProjectileObstacle extends BaseObstacle {
  constructor(
    init: BaseObstacleInit,
    launchVelocity: THREE.Vector3,
    private readonly ballistic: BallisticBehavior,
    private readonly born: number,
    private readonly spinAxis: THREE.Vector3,
    private readonly spinRate: number,
  ) {
    super(init);
    this.velocity.copy(launchVelocity);
  }

  protected override animate(ctx: WorldContext): void {
    if (!this.ballistic.landed) this.object3d.rotateOnAxis(this.spinAxis, this.spinRate * ctx.dt);
  }

  override isExpired(ctx: WorldContext): boolean {
    if (super.isExpired(ctx)) return true;
    if (this.ballistic.landed && ctx.time - this.ballistic.landedAt > REST_S) return true;
    return ctx.time - this.born > MAX_AGE_S;
  }
}

export class ProjectileFactory {
  readonly kind = 'projectile' as const;
  private readonly corridor: CorridorFrame;
  private readonly geometry: THREE.DodecahedronGeometry;
  private readonly material: THREE.MeshStandardMaterial;

  constructor(options: FactoryOptions = {}) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    this.geometry = new THREE.DodecahedronGeometry(PROJECTILE_RADIUS, 0);
    jitterVertices(this.geometry, 0.22, createRng(0x0d1));
    this.geometry.computeVertexNormals();
    this.material = new THREE.MeshStandardMaterial({ color: 0x5a544c, roughness: 0.95, flatShading: true });
  }

  /** A rock leaving `origin` at `velocity` (m/s), now. */
  throw(ctx: WorldContext, origin: THREE.Vector3, velocity: THREE.Vector3): Obstacle {
    const mesh = new THREE.Mesh(this.geometry, this.material);
    mesh.castShadow = true;
    const ballistic = new BallisticBehavior({ corridor: this.corridor });
    const axis = new THREE.Vector3(ctx.rng.next() - 0.5, ctx.rng.next() - 0.5, ctx.rng.next() - 0.5).normalize();
    return new ProjectileObstacle(
      {
        kind: this.kind,
        object3d: mesh,
        radius: PROJECTILE_RADIUS,
        position: origin,
        behavior: ballistic,
        corridor: this.corridor,
        scene: ctx.scene,
      },
      velocity,
      ballistic,
      ctx.time,
      axis,
      ctx.rng.range(6, 14),
    );
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
