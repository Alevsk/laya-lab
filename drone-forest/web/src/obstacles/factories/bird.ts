/**
 * Birds: the moving hazard. A small body with two wing planes that flap at a rate tied to
 * airspeed, oriented along the velocity the behaviour sets. The behaviour is injected — the
 * default is `RandomWanderBehavior` inside a 3–25 m band — so an AI policy replaces random
 * flight by passing a different `behavior` provider, with no change to this file.
 */
import * as THREE from 'three';
import type { Obstacle, ObstacleBehavior, SpawnOptions, WorldContext } from '../../core/types';
import { BaseObstacle, type BaseObstacleInit } from '../base';
import { RandomWanderBehavior } from '../behaviors/random-wander';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import type { SpawnProfile, SpawnableFactory } from '../profile';
import type { FactoryOptions } from './tree';

export interface BirdFactoryOptions extends FactoryOptions {
  /** Policy constructor; one behaviour instance per bird. */
  behavior?: (corridor: CorridorFrame) => ObstacleBehavior;
}

const RADIUS = 0.5;
const ALTITUDE_BAND: readonly [number, number] = [3, 25];
const FLAP_AMPLITUDE = 0.75;

class BirdObstacle extends BaseObstacle {
  private flapPhase = 0;
  private static readonly lookTarget = new THREE.Vector3();

  constructor(
    init: BaseObstacleInit,
    private readonly leftWing: THREE.Object3D,
    private readonly rightWing: THREE.Object3D,
  ) {
    super(init);
  }

  protected override animate(ctx: WorldContext): void {
    const speed = this.velocity.length();
    if (speed > 0.05) {
      this.object3d.lookAt(BirdObstacle.lookTarget.copy(this.position).add(this.velocity));
    }
    this.flapPhase += ctx.dt * 2 * Math.PI * (3 + speed * 0.9);
    const flap = Math.sin(this.flapPhase) * FLAP_AMPLITUDE;
    // The left wing is mirrored with rotation.y = π, so the same signed angle lifts both tips.
    this.leftWing.rotation.z = flap;
    this.rightWing.rotation.z = flap;
  }
}

export class BirdFactory implements SpawnableFactory {
  readonly kind = 'bird' as const;
  readonly profile: SpawnProfile = {
    layer: 'air',
    maxCount: 12,
    density: 0.16,
    baseRadius: RADIUS,
    scale: [0.8, 1.2],
    minSpacing: 2,
    respectsGap: false,
    altitude: ALTITUDE_BAND,
  };

  private readonly corridor: CorridorFrame;
  private readonly makeBehavior: (corridor: CorridorFrame) => ObstacleBehavior;
  private readonly bodyGeometry: THREE.ConeGeometry;
  private readonly headGeometry: THREE.SphereGeometry;
  private readonly wingGeometry: THREE.PlaneGeometry;
  private readonly bodyMaterial: THREE.MeshStandardMaterial;
  private readonly wingMaterial: THREE.MeshStandardMaterial;

  constructor(options: BirdFactoryOptions = {}) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    this.makeBehavior =
      options.behavior ??
      ((corridor) =>
        new RandomWanderBehavior({
          corridor,
          speed: [3, 7],
          verticalSpeed: [-1.2, 1.2],
          altitudeBand: ALTITUDE_BAND,
          wallMargin: 1.5,
          bobAmplitude: 0.25,
          bobHz: 1.4,
        }));
    // Body points along +Z so Object3D.lookAt aims it down the velocity vector.
    this.bodyGeometry = new THREE.ConeGeometry(0.14, 0.7, 6);
    this.bodyGeometry.rotateX(Math.PI / 2);
    this.headGeometry = new THREE.SphereGeometry(0.11, 6, 5);
    // Wing plane lies flat (XZ), hinged along its inner edge at the body.
    this.wingGeometry = new THREE.PlaneGeometry(0.75, 0.3);
    this.wingGeometry.rotateX(-Math.PI / 2);
    this.wingGeometry.translate(0.375, 0, 0);
    this.bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x2b2b30, roughness: 0.8, flatShading: true });
    this.wingMaterial = new THREE.MeshStandardMaterial({
      color: 0x3a3a42,
      roughness: 0.9,
      side: THREE.DoubleSide,
      flatShading: true,
    });
  }

  create(ctx: WorldContext, opts: SpawnOptions): Obstacle {
    const scale = opts.scale ?? ctx.rng.range(this.profile.scale[0], this.profile.scale[1]);
    const group = new THREE.Group();
    const body = new THREE.Mesh(this.bodyGeometry, this.bodyMaterial);
    body.castShadow = true;
    const head = new THREE.Mesh(this.headGeometry, this.bodyMaterial);
    head.position.set(0, 0.05, 0.38);
    const leftWing = new THREE.Mesh(this.wingGeometry, this.wingMaterial);
    leftWing.rotation.y = Math.PI;
    leftWing.castShadow = true;
    const rightWing = new THREE.Mesh(this.wingGeometry, this.wingMaterial);
    rightWing.castShadow = true;
    group.add(body, head, leftWing, rightWing);
    group.scale.setScalar(scale);
    return new BirdObstacle(
      {
        kind: this.kind,
        object3d: group,
        radius: RADIUS * scale,
        position: opts.position,
        behavior: this.makeBehavior(this.corridor),
        corridor: this.corridor,
        scene: ctx.scene,
      },
      leftWing,
      rightWing,
    );
  }

  dispose(): void {
    this.bodyGeometry.dispose();
    this.headGeometry.dispose();
    this.wingGeometry.dispose();
    this.bodyMaterial.dispose();
    this.wingMaterial.dispose();
  }
}
