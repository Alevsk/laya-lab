/** Throwaway numeric check: `node --experimental-strip-types src/obstacles/behaviors/ballistics.check.ts` */
import * as THREE from 'three';
import { positionAt, predictAimPoint, solveLaunch } from './ballistics.ts';

const assert = (cond: boolean, msg: string): void => { if (!cond) throw new Error(msg); };
const origin = new THREE.Vector3(2, 2.5, 0);
// 1) a reachable stationary target is hit on the low arc
const target = new THREE.Vector3(-3, 6, -30);
const sol = solveLaunch(origin, target, 24);
assert(sol.reachable, 'target should be reachable at 24 m/s');
const hit = positionAt(origin, sol.velocity, sol.flightTime);
assert(hit.distanceTo(target) < 1e-6, `low arc misses by ${hit.distanceTo(target)} m`);
assert(sol.velocity.y > 0, 'low arc still launches upward toward a higher target');
// 2) out of range -> 45° max-range shot, flagged
const far = new THREE.Vector3(0, 6, -200);
const s2 = solveLaunch(origin, far, 14);
assert(!s2.reachable, 'far target must be flagged unreachable');
assert(Math.abs(Math.atan2(s2.velocity.y, Math.hypot(s2.velocity.x, s2.velocity.z)) - Math.PI / 4) < 1e-9, 'unreachable -> 45 deg');
// 3) lead prediction: a target moving at 12 m/s along -Z is hit by aiming at the predicted point
const moving = new THREE.Vector3(0, 8, 30);       // drone 30 m short of the ogre, flying toward it
const vel = new THREE.Vector3(0, 0, -12);
const aim = predictAimPoint(origin, moving, vel, 22, 1);
const sol3 = solveLaunch(origin, aim, 22);
const whereTarget = moving.clone().addScaledVector(vel, sol3.flightTime);
const whereRock = positionAt(origin, sol3.velocity, sol3.flightTime);
assert(whereRock.distanceTo(whereTarget) < 0.05, `lead intercept misses by ${whereRock.distanceTo(whereTarget).toFixed(3)} m`);
// 3b) a target receding at 12 m/s from a 22 m/s rock: converges (damped) to a consistent intercept or a flagged max-range shot
const receding = new THREE.Vector3(0, 8, -25);
const aimR = predictAimPoint(origin, receding, vel, 22, 1);
const solR = solveLaunch(origin, aimR, 22);
const tgtR = receding.clone().addScaledVector(vel, solR.flightTime);
assert(!solR.reachable || positionAt(origin, solR.velocity, solR.flightTime).distanceTo(tgtR) < 0.5, `receding intercept misses by ${positionAt(origin, solR.velocity, solR.flightTime).distanceTo(tgtR).toFixed(2)} m while reachable`);
// 4) lead=0 aims at the current position (the miss grows with target speed)
const aim0 = predictAimPoint(origin, moving, vel, 22, 0);
assert(aim0.distanceTo(moving) < 1e-9, 'lead 0 must aim at the current position');
console.log(`ballistics ok: low-arc exact hit, unreachable flagged at 45°, lead intercept within ${whereRock.distanceTo(whereTarget).toFixed(4)} m over ${sol3.flightTime.toFixed(2)} s`);
