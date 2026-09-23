/**
 * The in-browser flight policy: a geometric heuristic over the sensor fan, pure and
 * deterministic so it can be unit-checked in node. Same spirit as the server's heuristic
 * engine (sector clearance scoring, forward preference, altitude bounds, anti-oscillation) —
 * it exists so the game is playable with no server, and as a strong, honest baseline that no
 * other engine is allowed to peek at.
 *
 * Scoring in one paragraph: every ray gives a clearance in [0, 1] — distance over a
 * speed-aware horizon (~2.5 s of flight, at least 20 m), so a tree 45 m out does not trigger a
 * swerve while one 15 m out does. Rays are grouped into sectors (ahead, left, right, up, down);
 * a sector's clearance is a blend of its worst ray and its mean so one blocked ray drags the
 * sector down without zeroing it.
 * Each action gets the clearance of the sector it moves into, forward gets a bias so the drone
 * keeps making progress, climb/descend are capped by the altitude room left inside the bounds,
 * and brake only wins when nothing else is clear. Hard rules run first: altitude limits, and an
 * emergency stop when the forward ray is inside the stopping distance and no escape is clear.
 * Hysteresis keeps a bank going for a few ticks and refuses to flip to the opposite bank unless
 * it is clearly better, which is what kills left-right oscillation between two trees.
 */
import { ACTIONS } from '../core/types';
import type { ActionName, Ray, SensorFrame } from '../core/types';

export interface HeuristicConfig {
  forwardBias: number;
  altitudeMargin: number;
  climbCost: number;
  descendCost: number;
  /** clearance below which the sector counts as blocked for the emergency rule */
  blockedClearance: number;
  reactionS: number;
  decelMps2: number;
  minStopDistance: number;
  /** ticks a fresh bank is held before another action may replace it (unless ahead clears) */
  holdTicks: number;
  /** score margin the opposite bank must win by to reverse a turn */
  reverseMargin: number;
  /** clearance horizon in seconds of flight at the current speed … */
  horizonS: number;
  /** … but never shorter than this many metres */
  minHorizon: number;
}

export const DEFAULT_HEURISTIC_CONFIG: HeuristicConfig = {
  forwardBias: 0.15,
  altitudeMargin: 1.5,
  climbCost: 0.1,
  descendCost: 0.15,
  blockedClearance: 0.3,
  reactionS: 0.4,
  decelMps2: 8,
  minStopDistance: 4,
  holdTicks: 3,
  reverseMargin: 0.15,
  horizonS: 2.5,
  minHorizon: 20,
};

export interface HeuristicState {
  lastAction: ActionName | null;
  held: number;
}

export interface HeuristicVerdict {
  action: ActionName;
  reason: string;
  /** per-action scores, normalised to sum to 1 — a preference profile, not calibrated probabilities */
  scores: Record<ActionName, number>;
  sectors: Sectors;
}

export interface Sectors { ahead: number; left: number; right: number; up: number; down: number }

export const initialHeuristicState = (): HeuristicState => ({ lastAction: null, held: 0 });

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);

function clearance(ray: Ray | undefined, horizon: number): number {
  if (!ray) return 1;
  return clamp01(ray.distance / Math.min(horizon, ray.max_range));
}

export function clearanceHorizon(frame: SensorFrame, cfg: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG): number {
  return Math.max(cfg.minHorizon, frame.drone.speed * cfg.horizonS);
}

/** worst-ray-dominated blend: a single blocked ray pulls the sector down hard */
function blend(values: number[]): number {
  if (values.length === 0) return 1;
  let min = Infinity;
  let sum = 0;
  for (const v of values) {
    if (v < min) min = v;
    sum += v;
  }
  return 0.65 * min + 0.35 * (sum / values.length);
}

export function sectorClearances(frame: SensorFrame, cfg: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG): Sectors {
  const horizon = clearanceHorizon(frame, cfg);
  const byName = new Map<string, Ray>();
  for (const r of frame.rays) byName.set(r.name, r);
  const c = (name: string): number => clearance(byName.get(name), horizon);
  return {
    ahead: blend([c('forward'), c('left_15'), c('right_15')]),
    left: blend([c('left_30'), c('left_45'), c('left_60')]),
    right: blend([c('right_30'), c('right_45'), c('right_60')]),
    up: c('up'),
    down: c('down'),
  };
}

function normalise(raw: Record<ActionName, number>): Record<ActionName, number> {
  let sum = 0;
  for (const a of ACTIONS) sum += Math.max(0, raw[a]);
  const out = {} as Record<ActionName, number>;
  for (const a of ACTIONS) out[a] = sum > 0 ? Math.max(0, raw[a]) / sum : 1 / ACTIONS.length;
  return out;
}

const fmt = (x: number): string => x.toFixed(2);

/**
 * One decision. `state` is mutated (last action, hold counter) so the caller keeps it per
 * episode; `reset` it on a new run.
 */
export function heuristicPolicy(
  frame: SensorFrame,
  state: HeuristicState,
  cfg: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
): HeuristicVerdict {
  const s = sectorClearances(frame, cfg);
  const alt = frame.drone.altitude;
  const { altitude_min: altMin, altitude_max: altMax } = frame.bounds;
  const forwardRay = frame.rays.find((r) => r.name === 'forward');
  const forwardDist = forwardRay ? forwardRay.distance : Infinity;
  const speed = frame.drone.speed;
  const stopDist = Math.max(cfg.minStopDistance, speed * cfg.reactionS + (speed * speed) / (2 * cfg.decelMps2));

  // room left inside the altitude band, as a 0..1 factor over a few metres
  const headroom = clamp01((altMax - cfg.altitudeMargin - alt) / 8);
  const floorroom = clamp01((alt - altMin - cfg.altitudeMargin) / 5);

  const raw: Record<ActionName, number> = {
    forward: s.ahead + cfg.forwardBias,
    bank_left: s.left,
    bank_right: s.right,
    climb: Math.min(s.up, headroom) - cfg.climbCost,
    descend: Math.min(s.down, floorroom) - cfg.descendCost,
    brake: 0,
  };
  const bestEscape = Math.max(s.ahead, s.left, s.right, Math.min(s.up, headroom), Math.min(s.down, floorroom));
  raw.brake = clamp01(1 - bestEscape) * 0.9;

  const finish = (action: ActionName, why: string): HeuristicVerdict => {
    if (action === state.lastAction) state.held++;
    else state.held = 1;
    state.lastAction = action;
    const tag = `A ${fmt(s.ahead)} L ${fmt(s.left)} R ${fmt(s.right)} U ${fmt(s.up)} D ${fmt(s.down)}`;
    return { action, reason: `${why} | ${tag}`, scores: normalise(raw), sectors: s };
  };

  // 1. altitude limits are not negotiable
  if (alt < altMin + cfg.altitudeMargin && s.up > cfg.blockedClearance) {
    return finish('climb', `below floor ${fmt(alt)}m < ${fmt(altMin + cfg.altitudeMargin)}m`);
  }
  if (alt > altMax - cfg.altitudeMargin && s.down > cfg.blockedClearance) {
    return finish('descend', `above ceiling ${fmt(alt)}m > ${fmt(altMax - cfg.altitudeMargin)}m`);
  }

  // 2. emergency: something inside the stopping distance and no clear way around it
  if (forwardDist < stopDist) {
    const escapes: Array<[ActionName, number]> = [
      ['bank_left', s.left],
      ['bank_right', s.right],
      ['climb', Math.min(s.up, headroom)],
      ['descend', Math.min(s.down, floorroom)],
    ];
    escapes.sort((a, b) => b[1] - a[1]);
    const [escape, clear] = escapes[0]!;
    if (clear < cfg.blockedClearance) {
      return finish('brake', `obstacle ${fmt(forwardDist)}m < stop ${fmt(stopDist)}m, no escape`);
    }
    // keep an in-progress bank rather than swapping sides in the middle of an evasion
    if ((state.lastAction === 'bank_left' || state.lastAction === 'bank_right') && raw[state.lastAction] > cfg.blockedClearance) {
      return finish(state.lastAction, `evading ${fmt(forwardDist)}m ahead, holding turn`);
    }
    return finish(escape, `evading ${fmt(forwardDist)}m ahead via clearest sector`);
  }

  // 3. hysteresis on banks
  let best: ActionName = 'forward';
  for (const a of ACTIONS) if (raw[a] > raw[best]) best = a;

  const last = state.lastAction;
  if (last === 'bank_left' || last === 'bank_right') {
    const opposite: ActionName = last === 'bank_left' ? 'bank_right' : 'bank_left';
    if (best === opposite && raw[best] < raw[last] + cfg.reverseMargin) {
      return finish(last, `holding turn, ${opposite} not clearly better`);
    }
    if (best !== last && state.held < cfg.holdTicks && s.ahead < 0.6 && raw[last] > cfg.blockedClearance) {
      return finish(last, `committing to turn (${state.held}/${cfg.holdTicks})`);
    }
  }

  const why =
    best === 'forward'
      ? 'ahead clear'
      : best === 'brake'
        ? 'boxed in'
        : `${best} is the clearest sector`;
  return finish(best, why);
}


/** Speed policy, mirrored from server/engines/heuristic_engine.py::_target_speed. */
export const SPEED_CLEAR_M = 30;
export const SPEED_NEAR_M = 8;
export const TTC_TARGET_S = 2.5;
const SPEED_FLOOR = 4;

/** Max speed whenever the way ahead is open, held down by whatever is closest in the front cone. */
export function targetSpeed(frame: SensorFrame, action: ActionName): number {
  const smax = frame.bounds.speed_max;
  let ahead = Infinity;
  for (const r of frame.rays) {
    if ((r.name === 'forward' || r.name === 'left_15' || r.name === 'right_15') && r.hit !== null) ahead = Math.min(ahead, r.distance);
  }
  const frac = clamp01((ahead - SPEED_NEAR_M) / (SPEED_CLEAR_M - SPEED_NEAR_M));
  let target = SPEED_FLOOR + frac * (smax - SPEED_FLOOR);
  const fwd = frame.rays.find((r) => r.name === 'forward');
  if (fwd && fwd.hit !== null) target = Math.min(target, Math.max(SPEED_FLOOR, fwd.distance / TTC_TARGET_S));
  const n = frame.nearest;
  if (n && n.closing_speed > 0 && Math.abs(n.bearing_deg) <= 30) target = Math.min(target, Math.max(SPEED_FLOOR, n.distance / TTC_TARGET_S));
  if (action === 'brake') target = SPEED_FLOOR;
  return Math.round(Math.min(smax, Math.max(SPEED_FLOOR, target)) * 100) / 100;
}
