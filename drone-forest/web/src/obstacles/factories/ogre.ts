/**
 * Ogres: brutes on the ground that throw rocks at the drone. Built from primitives around a
 * shoulder-pivoted throwing arm so the wind-up reads from a distance (the telegraph the drone's
 * sensors and a human can both see). The model turns to face the drone; the behaviour decides
 * when and how well to throw, scaled by the world's difficulty.
 *
 * Rocks are produced by an internal `ProjectileFactory` that shares its geometry across every
 * ogre this factory makes; the behaviour receives it as a `throwRock` closure and never learns
 * what a mesh is.
 */
import * as THREE from 'three';
import type { Obstacle, SpawnOptions, WorldContext } from '../../core/types';
import { BaseObstacle, type BaseObstacleInit } from '../base';
import { OgreThrowerBehavior } from '../behaviors/ogre-thrower';
import { DEFAULT_CORRIDOR, type CorridorFrame } from '../corridor';
import type { SpawnProfile, SpawnableFactory } from '../profile';
import { ProjectileFactory } from './projectile';
import type { FactoryOptions } from './tree';

const RADIUS = 1.4;
/** Sphere centre above the ground contact point, metres (unscaled). */
const CENTRE_HEIGHT = 1.65;
const ARM_REST = 0.35;
const ARM_RAISED = -2.4;

class OgreObstacle extends BaseObstacle {
  private armAngle = ARM_REST;
  private readonly tmp = new THREE.Vector3();

  constructor(
    init: BaseObstacleInit,
    private readonly thrower: OgreThrowerBehavior,
    private readonly throwingArm: THREE.Object3D,
    private readonly torso: THREE.Object3D,
  ) {
    super(init);
  }

  protected override animate(ctx: WorldContext): void {
    // face the drone (yaw only; the model's front is +Z)
    const f = this.thrower.facing;
    this.object3d.rotation.y = Math.atan2(f.x, f.z);
    // wind-up: arm swings back over the shoulder, then whips forward on release
    const target = ARM_REST + (ARM_RAISED - ARM_REST) * this.thrower.armRaise;
    this.armAngle += (target - this.armAngle) * Math.min(1, ctx.dt * 14);
    this.throwingArm.rotation.x = this.armAngle;
    // idle breathing
    this.torso.position.y = 1.75 + Math.sin(ctx.time * 1.7 + this.id) * 0.04;
    this.tmp.set(0, 0, 0);
  }
}

export class OgreFactory implements SpawnableFactory {
  readonly kind = 'ogre' as const;
  readonly profile: SpawnProfile = {
    layer: 'ground',
    maxCount: 4,
    density: 0.03,
    baseRadius: RADIUS,
    scale: [0.9, 1.25],
    minSpacing: 3,
    respectsGap: true,
  };

  private readonly corridor: CorridorFrame;
  private readonly projectiles: ProjectileFactory;
  private readonly legGeometry: THREE.CylinderGeometry;
  private readonly torsoGeometry: THREE.CapsuleGeometry;
  private readonly headGeometry: THREE.DodecahedronGeometry;
  private readonly armGeometry: THREE.CylinderGeometry;
  private readonly fistGeometry: THREE.SphereGeometry;
  private readonly clothGeometry: THREE.BoxGeometry;
  private readonly eyeGeometry: THREE.SphereGeometry;
  private readonly skin: THREE.MeshStandardMaterial;
  private readonly cloth: THREE.MeshStandardMaterial;
  private readonly eye: THREE.MeshStandardMaterial;

  constructor(options: FactoryOptions = {}) {
    this.corridor = options.corridor ?? DEFAULT_CORRIDOR;
    this.projectiles = new ProjectileFactory(options);
    this.legGeometry = new THREE.CylinderGeometry(0.26, 0.34, 1.1, 7);
    this.torsoGeometry = new THREE.CapsuleGeometry(0.78, 1.0, 4, 10);
    this.headGeometry = new THREE.DodecahedronGeometry(0.46, 0);
    // arm origin at the shoulder so rotation.x swings it like a joint
    this.armGeometry = new THREE.CylinderGeometry(0.2, 0.26, 1.25, 7);
    this.armGeometry.translate(0, -0.62, 0);
    this.fistGeometry = new THREE.SphereGeometry(0.28, 7, 6);
    this.fistGeometry.translate(0, -1.3, 0);
    this.clothGeometry = new THREE.BoxGeometry(1.5, 0.6, 1.2);
    this.eyeGeometry = new THREE.SphereGeometry(0.07, 6, 5);
    this.skin = new THREE.MeshStandardMaterial({ color: 0x6d7a48, roughness: 0.9, flatShading: true });
    this.cloth = new THREE.MeshStandardMaterial({ color: 0x5a3c28, roughness: 1, flatShading: true });
    this.eye = new THREE.MeshStandardMaterial({ color: 0xffd36b, emissive: 0xffb347, emissiveIntensity: 1.8 });
  }

  create(ctx: WorldContext, opts: SpawnOptions): Obstacle {
    const scale = opts.scale ?? ctx.rng.range(this.profile.scale[0], this.profile.scale[1]);
    const group = new THREE.Group();
    const mesh = (g: THREE.BufferGeometry, m: THREE.Material): THREE.Mesh => {
      const x = new THREE.Mesh(g, m);
      x.castShadow = true;
      x.receiveShadow = true;
      return x;
    };
    const legL = mesh(this.legGeometry, this.skin);
    legL.position.set(-0.38, 0.55, 0);
    const legR = mesh(this.legGeometry, this.skin);
    legR.position.set(0.38, 0.55, 0);
    const cloth = mesh(this.clothGeometry, this.cloth);
    cloth.position.set(0, 1.05, 0);
    const torso = mesh(this.torsoGeometry, this.skin);
    torso.position.set(0, 1.75, 0);
    const head = mesh(this.headGeometry, this.skin);
    head.position.set(0, 2.78, 0.1);
    const eyeL = new THREE.Mesh(this.eyeGeometry, this.eye);
    eyeL.position.set(-0.17, 2.85, 0.5);
    const eyeR = new THREE.Mesh(this.eyeGeometry, this.eye);
    eyeR.position.set(0.17, 2.85, 0.5);
    const armL = mesh(this.armGeometry, this.skin);
    armL.position.set(-0.98, 2.25, 0);
    armL.rotation.x = ARM_REST;
    armL.add(mesh(this.fistGeometry, this.skin));
    const armR = mesh(this.armGeometry, this.skin); // the throwing arm
    armR.position.set(0.98, 2.25, 0);
    armR.rotation.x = ARM_REST;
    armR.add(mesh(this.fistGeometry, this.skin));
    group.add(legL, legR, cloth, torso, head, eyeL, eyeR, armL, armR);
    // the sphere centre sits mid-body; place the model's feet on the ground relative to it
    group.children.forEach((child) => child.position.y -= CENTRE_HEIGHT);
    group.scale.setScalar(scale);

    const position = opts.position.clone().addScaledVector(this.corridor.up, CENTRE_HEIGHT * scale);
    const thrower = new OgreThrowerBehavior({
      corridor: this.corridor,
      handHeight: (2.6 - CENTRE_HEIGHT) * scale,
      throwRock: (c, origin, velocity) => this.projectiles.throw(c, origin, velocity),
    });
    return new OgreObstacle(
      {
        kind: this.kind,
        object3d: group,
        radius: RADIUS * scale,
        position,
        behavior: thrower,
        corridor: this.corridor,
        scene: ctx.scene,
      },
      thrower,
      armR,
      torso,
    );
  }

  dispose(): void {
    [this.legGeometry, this.torsoGeometry, this.headGeometry, this.armGeometry, this.fistGeometry,
      this.clothGeometry, this.eyeGeometry].forEach((g) => g.dispose());
    [this.skin, this.cloth, this.eye].forEach((m) => m.dispose());
    this.projectiles.dispose();
  }
}
