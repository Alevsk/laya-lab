/**
 * The world: renderer, camera, sky, lighting, endless terrain, scenery, particles, post-fx and
 * the obstacle pool. It depends on `Spawner` and `Obstacle` from core/types only — never on a
 * concrete tree or bird — and on a `FollowTarget` (the drone) it is told about after creation.
 *
 * Conventions (shared with drone/, sensors/ and the server framing):
 *   - the corridor runs along -Z; heading 0° looks down -Z; bearing + is to the right (+X)
 *   - altitude is world y; the sensor/collision ground is the y = 0 plane
 *   - one World owns exactly one WebGLRenderer attached to `opts.root`
 *
 * Per frame `update(dt)` advances time, recycles terrain around the drone, asks the spawner for
 * new obstacles, updates every obstacle through its own behaviour and disposes the expired.
 */
import * as THREE from 'three';
import { createRng } from '../core/rng';
import type { Bounds, Obstacle, Rng, Spawner, WorldContext } from '../core/types';
import { SCENERY_LAYER } from './layers';
import { createLighting } from './lighting';
import { createParticles } from './particles';
import { createPostFx } from './postfx';
import { createScenery } from './scenery';
import { createSky, FOG_COLOR } from './sky';
import { createTerrain } from './terrain';

export { OBSTACLE_LAYER, SCENERY_LAYER, markScenery } from './layers';

export type WorldBounds = Bounds & { half_width: number; corridor_length: number };

export const DEFAULT_BOUNDS: WorldBounds = {
  altitude_min: 2,
  altitude_max: 40,
  speed_max: 18,
  half_width: 24,
  corridor_length: 180,
};

/** What the world needs to know about the thing it follows (the drone). */
export interface FollowTarget {
  readonly position: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  readonly headingDeg: number;
}

export interface WorldOptions {
  root: HTMLElement;
  seed: number;
  /** Where obstacles come from. Omit for an empty corridor (useful when testing the scene alone). */
  spawner?: Spawner;
  bounds?: Partial<WorldBounds>;
}

export interface ResetOptions {
  /** Re-seed the world's own rng (particles, flash jitter) — pass the run seed for replayable episodes. */
  seed?: number;
  /** Swap the spawner (main.ts rebuilds one with a fresh rng so each episode sees the same forest). */
  spawner?: Spawner;
}

export interface World {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly ctx: WorldContext;
  readonly obstacles: Obstacle[];
  readonly rng: Rng;
  readonly sunDirection: THREE.Vector3;
  /** Attach the object the terrain, lights, particles and spawner follow. */
  follow(target: FollowTarget): void;
  /** Put an obstacle into the scene and the pool (idempotent). */
  add(o: Obstacle): void;
  /** Take an obstacle out of the scene and the pool and dispose it. */
  remove(o: Obstacle): void;
  update(dt: number): void;
  render(): void;
  resize(): void;
  /** Brief red exposure pulse — collision feedback. */
  flash(): void;
  /** Clear every obstacle and rebuild the terrain around the followed target. */
  reset(opts?: ResetOptions): void;
  dispose(): void;
}

const FLASH_SECONDS = 0.45;
const BASE_EXPOSURE = 0.58;
const FORWARD = new THREE.Vector3(0, 0, -1);

export function createWorld(opts: WorldOptions): World {
  const bounds: WorldBounds = { ...DEFAULT_BOUNDS, ...opts.bounds };
  let rng = createRng(opts.seed);
  let spawner: Spawner | null = opts.spawner ?? null;

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = BASE_EXPOSURE;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  opts.root.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(62, 1, 0.1, 1500);
  camera.layers.enable(SCENERY_LAYER);
  camera.position.set(0, 10, 12);
  scene.add(camera);

  const sky = createSky(scene);
  const lighting = createLighting(scene, sky.sunDirection);
  const terrain = createTerrain(scene, bounds.half_width);
  const scenery = createScenery(scene, terrain, bounds.half_width);
  const particles = createParticles(scene, rng);
  const postfx = createPostFx(renderer, scene, camera);

  const obstacles: Obstacle[] = [];
  const ctx: WorldContext = {
    time: 0,
    dt: 0,
    scene,
    rng,
    bounds,
    drone: { position: new THREE.Vector3(0, 8, 0), velocity: new THREE.Vector3(), heading_deg: 0 },
  };

  let target: FollowTarget | null = null;
  let flashT = 0;
  const fogBase = new THREE.Color(FOG_COLOR);
  const fogFlash = new THREE.Color(0xd8402a);

  const add = (o: Obstacle): void => {
    if (!obstacles.includes(o)) obstacles.push(o);
    o.object3d.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
    if (o.object3d.parent !== scene) scene.add(o.object3d);
  };

  const remove = (o: Obstacle): void => {
    const i = obstacles.indexOf(o);
    if (i >= 0) obstacles.splice(i, 1);
    scene.remove(o.object3d);
    o.dispose();
  };

  const resize = (): void => {
    const w = opts.root.clientWidth || window.innerWidth;
    const h = opts.root.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    postfx.setSize(w, h);
  };
  window.addEventListener('resize', resize);
  resize();

  const syncDrone = (): void => {
    if (!target) return;
    ctx.drone.position.copy(target.position);
    ctx.drone.velocity.copy(target.velocity);
    ctx.drone.heading_deg = target.headingDeg;
  };

  const update = (dt: number): void => {
    ctx.time += dt;
    ctx.dt = dt;
    syncDrone();
    const anchor = ctx.drone.position;

    terrain.update(anchor.z);
    lighting.update(anchor, FORWARD);
    particles.update(ctx.time, anchor);

    if (spawner) for (const o of spawner.update(ctx, obstacles)) add(o);
    for (let i = obstacles.length - 1; i >= 0; i--) {
      const o = obstacles[i]!;
      o.update(ctx);
      if (o.isExpired(ctx)) remove(o);
    }

    if (flashT > 0) {
      flashT = Math.max(0, flashT - dt / FLASH_SECONDS);
      const k = flashT * flashT;
      renderer.toneMappingExposure = BASE_EXPOSURE * (1 + 0.55 * k);
      sky.fog.color.copy(fogBase).lerp(fogFlash, 0.4 * k);
    }
  };

  const render = (): void => {
    sky.update(camera.position);
    postfx.render();
  };

  const reset = (r?: ResetOptions): void => {
    for (let i = obstacles.length - 1; i >= 0; i--) remove(obstacles[i]!);
    if (r?.seed !== undefined) {
      rng = createRng(r.seed);
      ctx.rng = rng;
    }
    if (r?.spawner) spawner = r.spawner;
    ctx.time = 0;
    flashT = 0;
    renderer.toneMappingExposure = BASE_EXPOSURE;
    sky.fog.color.copy(fogBase);
    terrain.reset();
    syncDrone();
    terrain.update(ctx.drone.position.z);
  };

  terrain.update(ctx.drone.position.z);

  return {
    scene,
    camera,
    renderer,
    ctx,
    obstacles,
    get rng() {
      return rng;
    },
    sunDirection: sky.sunDirection,
    follow(t) {
      target = t;
      syncDrone();
      terrain.reset();
      terrain.update(ctx.drone.position.z);
    },
    add,
    remove,
    update,
    render,
    resize,
    flash() {
      flashT = 1;
    },
    reset,
    dispose() {
      window.removeEventListener('resize', resize);
      for (let i = obstacles.length - 1; i >= 0; i--) remove(obstacles[i]!);
      particles.dispose();
      scenery.dispose();
      postfx.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
