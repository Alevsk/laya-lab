/**
 * Headless behaviour check for ogres: `node --experimental-strip-types src/obstacles/ogre.check.ts`
 * An ogre stands 60 m ahead of a drone flying straight at it at 8 m/s. We step the world at
 * 60 Hz for 8 s at difficulty 1 and 5 and measure what the behaviour did. (Throws stop inside
 * 14 m, so the engagement window is range-14 m long: ~2.6 s at level 1, ~5.8 s at level 5.)
 */
import * as THREE from 'three';
import type { Obstacle, WorldContext } from '../core/types.ts';
import { createRng } from '../core/rng.ts';
import { OgreFactory } from './factories/ogre.ts';
import { ogreTuningFor } from './behaviors/index.ts';
import { positionAt } from './behaviors/ballistics.ts';

const assert = (c: boolean, m: string): void => { if (!c) throw new Error(m); };
const DT = 1 / 60;

function run(level: number, seed = 11) {
  const rng = createRng(seed);
  const emitted: Obstacle[] = [];
  const ctx: WorldContext = {
    time: 0, dt: DT, scene: new THREE.Scene(), rng,
    bounds: { altitude_min: 2, altitude_max: 40, speed_max: 18, half_width: 24, corridor_length: 200 },
    drone: { position: new THREE.Vector3(0, 8, 0), velocity: new THREE.Vector3(0, 0, -8), heading_deg: 0 },
    difficulty: level,
    emit: (o) => { emitted.push(o); },
  };
  const factory = new OgreFactory();
  const ogre = factory.create(ctx, { position: new THREE.Vector3(1.5, 0, -60), scale: 1 });
  const rocks = new Map<Obstacle, { closest: number; launchSpeed: number; landed: boolean }>();
  for (let step = 0; step < 8 * 60; step++) {
    ctx.time += DT;
    ctx.drone.position.addScaledVector(ctx.drone.velocity, DT);
    const before = emitted.length;
    ogre.update(ctx);
    for (const r of emitted.slice(before)) rocks.set(r, { closest: Infinity, launchSpeed: r.velocity.length(), landed: false });
    for (const r of emitted) {
      r.update(ctx);
      const rec = rocks.get(r)!;
      rec.closest = Math.min(rec.closest, r.position.distanceTo(ctx.drone.position));
      if (r.velocity.lengthSq() === 0 && r.position.y <= r.radius + 1e-6) rec.landed = true;
    }
  }
  const t = ogreTuningFor(level);
  const list = [...rocks.values()];
  return { level, throws: list.length, tuning: t,
    meanClosest: list.length ? list.reduce((a, b) => a + b.closest, 0) / list.length : NaN,
    meanSpeed: list.length ? list.reduce((a, b) => a + b.launchSpeed, 0) / list.length : NaN,
    landed: list.filter((r) => r.landed).length,
    ogreVelocity: ogre.velocity.length(), kind: emitted[0]?.kind, ogreKind: ogre.kind };
}

const easy = run(1), brutal = run(5);
for (const r of [easy, brutal]) console.log(JSON.stringify({ ...r, tuning: undefined, meanClosest: +r.meanClosest.toFixed(2), meanSpeed: +r.meanSpeed.toFixed(2) }));
assert(easy.ogreKind === 'ogre' && easy.kind === 'projectile', 'kinds must be ogre / projectile');
assert(easy.ogreVelocity === 0, 'an ogre does not move');
assert(easy.throws >= 1 && brutal.throws >= 1, 'both levels must throw at least once in 8 s');
assert(brutal.throws > easy.throws, `level 5 must throw more often (${brutal.throws} vs ${easy.throws})`);
assert(brutal.meanSpeed > easy.meanSpeed + 6, `level 5 rocks must be much faster (${brutal.meanSpeed} vs ${easy.meanSpeed})`);
assert(brutal.meanClosest < easy.meanClosest, `level 5 must be more accurate (closest ${brutal.meanClosest} vs ${easy.meanClosest})`);
assert(brutal.meanClosest < 3, `level 5 rocks should pass within 3 m of a drone flying straight (got ${brutal.meanClosest})`);
assert(easy.landed + brutal.landed >= 1, 'rocks must land under gravity within the run');
// gravity sanity on the ballistics itself: a rock launched upward comes back down
const p = positionAt(new THREE.Vector3(0, 2, 0), new THREE.Vector3(0, 10, -10), 3);
assert(p.y < 2, 'after 3 s a 10 m/s upward launch is below its origin');
console.log('ogre check ok');
