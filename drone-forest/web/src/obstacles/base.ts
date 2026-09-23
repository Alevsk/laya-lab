/**
 * The one concrete `Obstacle` implementation. Factories build an Object3D and hand it here
 * together with a behaviour; the base class owns the bookkeeping every kind shares:
 *
 *   - `position` IS the bounding-sphere centre, and the Object3D origin is placed there too
 *     (factories build their meshes around the sphere centre) so `object3d.position` is a
 *     straight copy each frame and collision / raycast code needs no per-kind offsets;
 *   - `update` runs the behaviour (the policy that decides where the obstacle goes), syncs the
 *     transform, then calls the `animate` hook (how the obstacle looks while doing it);
 *   - expiry is measured along the corridor so recycling is independent of the drone's heading;
 *   - `dispose` only detaches from the scene: geometry and materials are shared per factory
 *     and freed by the factory, never by an instance.
 */
import * as THREE from 'three';
import type { Obstacle, ObstacleBehavior, ObstacleKind, WorldContext } from '../core/types';
import type { CorridorFrame } from './corridor';

/** How far behind the drone (metres along the corridor) an obstacle is considered gone. */
export const EXPIRE_BEHIND_M = 25;

let nextObstacleId = 1;

export interface BaseObstacleInit {
  kind: ObstacleKind;
  object3d: THREE.Object3D;
  radius: number;
  /** bounding-sphere centre in world space */
  position: THREE.Vector3;
  behavior: ObstacleBehavior;
  corridor: CorridorFrame;
  scene: THREE.Scene;
}

export class BaseObstacle implements Obstacle {
  readonly id: number;
  readonly kind: ObstacleKind;
  readonly object3d: THREE.Object3D;
  readonly radius: number;
  readonly position: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  behavior: ObstacleBehavior;
  protected readonly corridor: CorridorFrame;

  constructor(init: BaseObstacleInit) {
    this.id = nextObstacleId++;
    this.kind = init.kind;
    this.object3d = init.object3d;
    this.radius = init.radius;
    this.position = init.position.clone();
    this.velocity = new THREE.Vector3();
    this.behavior = init.behavior;
    this.corridor = init.corridor;
    this.object3d.position.copy(this.position);
    this.object3d.userData['obstacleId'] = this.id;
    this.object3d.userData['obstacleKind'] = this.kind;
    init.scene.add(this.object3d);
  }

  update(ctx: WorldContext): void {
    this.behavior.update(this, ctx);
    this.object3d.position.copy(this.position);
    this.animate(ctx);
  }

  /** Presentation hook run after the behaviour moved the obstacle (flapping, swaying, ...). */
  protected animate(_ctx: WorldContext): void {}

  isExpired(ctx: WorldContext): boolean {
    return this.corridor.along(this.position) < this.corridor.along(ctx.drone.position) - EXPIRE_BEHIND_M;
  }

  dispose(): void {
    this.object3d.removeFromParent();
  }
}
