/**
 * Entry point: wires the modules together and runs the loop.
 *
 * Every cross-module call goes through `core/types.ts` interfaces and each module's `index.ts`
 * factory — main.ts is the ONLY place that knows which concrete world, spawner, sensor,
 * decision source and HUD are in play. In play mode the loop is:
 *
 *   rAF ──► fixed sub-steps (dt capped, h ≤ FIXED_DT) ──► every step:
 *             drone.applyAction(loop.current()) → drone.update → world.update → collide
 *             at DECISION_HZ: sensors.frame() → decisionLoop.tick()   (async, never blocks)
 *         ──► camera rig, render, HUD
 *
 * In arena mode (`?arena=quality|realtime`) the same world, drone, sensors and source are
 * handed to `runArena`, which owns the simulation for a fixed number of simulated seconds and
 * prints one `ARENA_RESULT {json}` line — the fair-comparison path between engines.
 *
 * URL parameters:
 *   engine=heuristic|laya|random|local   remote engine name, or the in-browser heuristic
 *   seed=<int>       world seed (identical forest for every engine → fair comparison)
 *   hz=<n>           decision rate (default 10)
 *   corridor=forest|canyon|open   corridor preset (half-width 24 / 12 / 40 m)
 *   seconds=<n>      play: episode length, 0 (default) = fly until a collision; arena: run length,
 *                    0 (default) = endless, until the first collision or Escape
 *   arena=quality|realtime   run the benchmark instead of playing (see decision/arena.ts)
 *   server=<ws url>  decision microservice (default ws://127.0.0.1:8765/ws)
 * If the service is unreachable the game falls back to the local heuristic so it always runs.
 * Keys: C camera (chase / nose), R sensor rays on/off, Space reset (arena: restart), P pause,
 * H hide/show every panel, A hide/show the arena panel.
 *
 * `window.__droneForest` exposes live state and stats for automated verification (Playwright).
 */
import { createRng } from './core/rng';
import type { ActionName, Decision, DecisionSource, DecisionStats, GameEvent, SensorFrame } from './core/types';
import {
  createDecisionLoop,
  createLocalHeuristicSource,
  createRemoteSource,
  parseArenaParams,
  runArena,
  type ArenaParams,
  type ArenaResult,
  type LoopStats,
} from './decision';
import { createChaseRig, createDrone, type CameraMode } from './drone';
import { createHud } from './hud';
import { createSpawner, defaultFactories } from './obstacles';
import { createRayDebug, createSensors } from './sensors';
import { createWorld, DEFAULT_BOUNDS, type WorldBounds } from './world';

// ---------------------------------------------------------------- configuration

type EngineName = 'heuristic' | 'laya' | 'random' | 'local';
type CorridorName = 'forest' | 'canyon' | 'open';

interface RunConfig {
  engine: EngineName;
  seed: number;
  hz: number;
  corridor: CorridorName;
  seconds: number;
  server: string;
  /** obstacle difficulty preset 1..5 */
  difficulty: number;
  arena: ArenaParams | null;
  /** last seed of a `seeds=A-B` chain, or null when not chaining */
  seedsEnd: number | null;
}

const CORRIDORS: Record<CorridorName, Partial<WorldBounds>> = {
  forest: { half_width: 24 },
  canyon: { half_width: 12 },
  open: { half_width: 40 },
};

const ENGINES: readonly EngineName[] = ['heuristic', 'laya', 'random', 'local'];
const FIXED_DT = 1 / 120;
const MAX_FRAME_DT = 1 / 20;
const RESET_DELAY_MS = 1200;
/**
 * How long to wait for the service's `info` (engine loaded and warm) before flying on it. A cold
 * Laya load is ~3 s on an idle M4 Max but was measured at 60-85 s on the same machine under a
 * load average of 465, so the budget is generous; a warm engine answers in milliseconds.
 */
const ENGINE_READY_MS = 120_000;

function readConfig(): RunConfig {
  const q = new URLSearchParams(window.location.search);
  const num = (key: string, fallback: number): number => {
    const v = Number(q.get(key));
    return q.has(key) && Number.isFinite(v) ? v : fallback;
  };
  const engine = q.get('engine') as EngineName | null;
  const corridor = q.get('corridor') as CorridorName | null;
  const arena = parseArenaParams(window.location.search);
  const difficulty = Math.min(5, Math.max(1, Math.round(num('difficulty', 3))));
  return {
    difficulty,
    engine: engine && ENGINES.includes(engine) ? engine : 'heuristic',
    seed: num('seed', 1) >>> 0,
    hz: Math.min(60, Math.max(1, num('hz', 10))),
    corridor: corridor && corridor in CORRIDORS ? corridor : 'forest',
    seconds: Math.max(0, num('seconds', 0)),
    server: q.get('server') ?? 'ws://127.0.0.1:8765/ws',
    arena: arena ? { ...arena, engine: engine && ENGINES.includes(engine) ? engine : 'heuristic', seed: num('seed', 1) >>> 0, difficulty } : null,
    seedsEnd: parseSeedsEnd(q.get('seeds')),
  };
}

/**
 * `seeds=100-115` chains arena runs: when a run finishes and `seed` is below the end of the
 * range, the page reloads itself with `seed+1`. One navigation therefore collects a whole
 * batch of runs — and, through the remote engine, a whole batch of teacher-labelled telemetry.
 */
function parseSeedsEnd(raw: string | null): number | null {
  if (!raw) return null;
  const m = /^(\d+)-(\d+)$/.exec(raw.trim());
  if (!m) return null;
  const end = Number(m[2]);
  return Number.isFinite(end) ? end >>> 0 : null;
}

// ---------------------------------------------------------------- run state (what the HUD shows)

export interface RunState {
  engine: string;
  requestedEngine: EngineName;
  connected: boolean;
  action: ActionName;
  lastDecision: Decision | null;
  stats: DecisionStats;
  loop: LoopStats;
  fps: number;
  simTime: number;
  episode: number;
  episodeTime: number;
  distance: number;
  speed: number;
  altitude: number;
  lateral: number;
  collisions: number;
  nearMisses: number;
  bestDistance: number;
  frame: SensorFrame | null;
  raysVisible: boolean;
  obstacles: number;
  paused: boolean;
  resetting: boolean;
  camera: CameraMode;
  seed: number;
  hz: number;
  arena: string;
  seconds: number;  /** obstacle difficulty preset 1..5 */
  difficulty: number;
  /** engine-recommended forward speed, m/s; null = cruise */
  targetSpeed: number | null;
}

/** Debug handle for automated verification; see the module header. */
export interface DroneForestDebug {
  config: RunConfig;
  state(): RunState;
  stats(): { source: DecisionStats; loop: LoopStats };
  arenaResult: ArenaResult | null;
  /** Resolves with the arena result in arena mode; resolves null immediately in play mode. */
  arenaDone: Promise<ArenaResult | null>;
}

declare global {
  interface Window {
    __droneForest?: DroneForestDebug;
  }
}

// ---------------------------------------------------------------- decision source selection

/**
 * Open the requested source. A remote source only counts once the socket is open AND the
 * service has sent `info` — the service warms the engine before that message, so waiting for
 * it means a 2-3 s Laya load never turns into blind flight (which would be unfair to Laya).
 */
async function openSource(cfg: RunConfig, engine: EngineName): Promise<DecisionSource> {
  if (engine !== 'local') {
    let ready: () => void = () => {};
    const info = new Promise<void>((resolve) => {
      ready = resolve;
    });
    const remote = createRemoteSource(cfg.server, engine, {
      onInfo: () => ready(),
      onError: (message) => console.warn(`[drone-forest] service: ${message}`),
    });
    await remote.connect();
    if (remote.stats().connected) {
      const timeout = new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), ENGINE_READY_MS));
      if ((await Promise.race([info.then(() => 'ready' as const), timeout])) === 'ready') return remote;
      console.warn(`[drone-forest] engine ${engine} did not become ready in ${ENGINE_READY_MS} ms; using the local heuristic`);
    } else {
      console.warn(`[drone-forest] ${cfg.server} unreachable; using the local heuristic`);
    }
    remote.dispose();
  }
  const local = createLocalHeuristicSource();
  await local.connect();
  return local;
}

// ---------------------------------------------------------------- main

async function main(): Promise<void> {
  const cfg = readConfig();
  const appRoot = document.getElementById('app');
  const hudRoot = document.getElementById('hud');
  if (!appRoot || !hudRoot) throw new Error('index.html must provide #app and #hud');

  const bounds: WorldBounds = { ...DEFAULT_BOUNDS, ...CORRIDORS[cfg.corridor] };
  // One set of factories (shared GPU resources) for the whole session; a fresh rng per episode
  // so every episode — and every engine — flies the same forest.
  const factories = defaultFactories();
  let difficulty = cfg.difficulty;
  const newSpawner = () => createSpawner(createRng(cfg.seed), factories, { level: difficulty });

  const world = createWorld({ root: appRoot, seed: cfg.seed, spawner: newSpawner(), bounds });
  const drone = createDrone(world.scene);
  world.follow(drone);
  const rig = createChaseRig(world.camera, drone);
  const sensors = createSensors(world, drone);
  const rays = createRayDebug(world.scene);
  const hud = createHud(hudRoot, {
    engines: ENGINES,
    keys: [
      ['C', 'camera'],
      ['R', 'rays'],
      ['Space', 'reset'],
      ['P', 'pause'],
      ['H', 'panels'],
      ['A', 'arena'],
      ['Esc', 'stop'],
    ],
  });

  let source = await openSource(cfg, cfg.engine);
  hud.setEngine(cfg.engine);
  hud.setDifficulty(difficulty);
  const loop = createDecisionLoop(source, 1000 / cfg.hz, { mode: cfg.arena?.mode ?? 'realtime' });

  let frameId = 0;
  let simTime = 0;
  let episode = 1;
  let episodeStart = 0;
  let startZ = drone.position.z;
  let collisions = 0;
  let nearMisses = 0;
  let bestDistance = 0;
  let paused = false;
  let stopRequested = false;
  let resetAt: number | null = null;
  let decisionClock = 0;
  let lastAction: ActionName | null = null;
  let fps = 60;
  let arenaResult: ArenaResult | null = null;
  const nearMissed = new Set<number>();
  const decisionInterval = 1 / cfg.hz;

  const event = (type: GameEvent['type'], extra: Partial<GameEvent> = {}): void => {
    source.report({ type, frame_id: frameId, t: simTime, ...extra });
  };

  const distance = (): number => Math.max(0, startZ - drone.position.z);

  const scoreAndReset = (why: 'collision' | 'timeout' | 'manual'): void => {
    event('score', {
      details: {
        reason: why,
        episode,
        distance_m: distance(),
        duration_s: simTime - episodeStart,
        near_misses: nearMisses,
        engine: source.name,
      },
    });
    drone.reset();
    world.reset({ seed: cfg.seed, spawner: newSpawner() });
    rig.snap();
    loop.reset();
    nearMissed.clear();
    episode += 1;
    episodeStart = simTime;
    startZ = drone.position.z;
    decisionClock = 0;
    lastAction = null;
    event('reset', { details: { episode } });
  };

  const onCollision = (kind: GameEvent['obstacle_kind']): void => {
    collisions += 1;
    world.flash();
    hud.flash('collision');
    event('collision', { obstacle_kind: kind ?? null, details: { distance_m: distance(), episode } });
    resetAt = performance.now() + RESET_DELAY_MS;
  };

  const switchEngine = async (name: string): Promise<void> => {
    if (!ENGINES.includes(name as EngineName)) return;
    const previous = source;
    event('engine_switch', { details: { from: previous.name, to: name } });
    source = await openSource(cfg, name as EngineName);
    hud.setEngine(name);
    loop.setSource(source);
    previous.dispose();
    scoreAndReset('manual');
  };

  const state = (): RunState => ({
    engine: source.name,
    requestedEngine: cfg.engine,
    connected: source.stats().connected,
    action: loop.current(),
    lastDecision: loop.last(),
    stats: source.stats(),
    loop: loop.stats(),
    fps,
    simTime,
    episode,
    episodeTime: simTime - episodeStart,
    distance: distance(),
    speed: drone.speed,
    altitude: drone.position.y,
    lateral: drone.position.x,
    collisions,
    nearMisses,
    bestDistance,
    frame: sensors.lastFrame(),
    raysVisible: rays.isVisible(),
    obstacles: world.obstacles.length,
    paused,
    resetting: resetAt !== null,
    camera: rig.mode,
    seed: cfg.seed,
    hz: cfg.hz,
    arena: cfg.arena ? `arena:${cfg.arena.mode}` : cfg.corridor,
    seconds: cfg.arena?.seconds ?? cfg.seconds,
    difficulty,
    targetSpeed: drone.targetSpeed,
  });

  let resolveArena: (r: ArenaResult | null) => void = () => {};
  const arenaDone = new Promise<ArenaResult | null>((resolve) => {
    resolveArena = resolve;
  });
  window.__droneForest = {
    config: cfg,
    state,
    stats: () => ({ source: source.stats(), loop: loop.stats() }),
    get arenaResult() {
      return arenaResult;
    },
    arenaDone,
  };

  window.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    switch (e.code) {
      case 'KeyC':
        rig.toggle();
        break;
      case 'KeyR':
        rays.toggle();
        break;
      case 'Space':
        e.preventDefault();
        if (cfg.arena) {
          window.location.reload(); // an arena run is defined by its URL; restarting it is reloading it
          return;
        }
        scoreAndReset('manual');
        break;
      case 'KeyP':
        paused = !paused;
        break;
      case 'KeyH':
        hud.togglePanels();
        break;
      case 'KeyA':
        hud.toggleArena();
        break;
      case 'Escape':
        if (cfg.arena) stopRequested = true;   // end an arena run now and show what it has
        break;
    }
  });
  window.addEventListener('beforeunload', () => source.dispose());

  let last = performance.now();
  document.addEventListener('visibilitychange', () => {
    last = performance.now();
  });

  // ---- arena mode: the runner owns the simulation; we only render what it does
  if (cfg.arena) {
    const params = cfg.arena;
    hud.setArenaProgress({ t: 0, seconds: params.seconds, collisions: 0, near_misses: 0, distance: 0, action: 'forward' });
    const render = (): void => {
      const now = performance.now();
      const raw = Math.min((now - last) / 1000, MAX_FRAME_DT);
      last = now;
      rig.update(raw);
      world.render();
      hud.update(state());
    };
    const result = await runArena(params, {
      world,
      drone,
      sensors,
      source,
      loop,
      intervalMs: 1000 / cfg.hz,
      render,
      isPaused: () => paused,
      shouldStop: () => stopRequested,
      onFrame: (f) => {
        frameId = f.frame_id;
        simTime = f.t;
        rays.update(f);
      },
      onProgress: (p) => {
        simTime = p.t;
        collisions = p.collisions;
        nearMisses = p.near_misses;
        bestDistance = Math.max(bestDistance, distance());
        hud.setArenaProgress(p);
      },
    });
    arenaResult = result;
    hud.showArena(result);
    resolveArena(result);
    if (cfg.seedsEnd !== null && cfg.seed < cfg.seedsEnd) {
      const next = new URLSearchParams(window.location.search);
      next.set('seed', String(cfg.seed + 1));
      console.log(`ARENA_CHAIN next seed ${cfg.seed + 1} of ${cfg.seedsEnd}`);
      setTimeout(() => { window.location.search = next.toString(); }, 400);
      return;
    }
    // keep drawing the final state so the run stays inspectable
    const idle = (): void => {
      render();
      requestAnimationFrame(idle);
    };
    requestAnimationFrame(idle);
    return;
  }
  resolveArena(null);

  // ---- play mode
  hud.onEngineChange((name: string) => {
    void switchEngine(name);
  });
  hud.onReset(() => scoreAndReset('manual'));
  hud.onDifficultyChange((level: number) => {
    if (cfg.arena) return; // arena runs are fixed by their URL so results stay comparable
    difficulty = level;
    event('reset', { details: { reason: 'difficulty', difficulty: level } });
    scoreAndReset('manual');
  });

  const step = (h: number): void => {
    simTime += h;
    drone.applyAction(loop.current(), h);
    drone.update(h, world.ctx);
    world.update(h);

    const c = drone.collide(world.obstacles);
    if (c.hit) {
      onCollision(c.hit.kind);
      return;
    }
    if (c.nearMiss && !nearMissed.has(c.nearMiss.id)) {
      nearMissed.add(c.nearMiss.id);
      nearMisses += 1;
      hud.flash('near_miss');
      event('near_miss', { obstacle_kind: c.nearMiss.kind });
    }

    decisionClock += h;
    if (decisionClock >= decisionInterval) {
      decisionClock -= decisionInterval;
      frameId += 1;
      const frame: SensorFrame = sensors.frame(frameId, simTime, lastAction);
      loop.tick(frame);
      rays.update(frame);
      lastAction = loop.current();
    }
    {
      const want = loop.last()?.target_speed ?? null;
      if (want !== drone.targetSpeed) drone.setTargetSpeed(want);
    }

    if (cfg.seconds > 0 && simTime - episodeStart >= cfg.seconds) scoreAndReset('timeout');
  };

  const frame = (now: number): void => {
    const raw = Math.min((now - last) / 1000, MAX_FRAME_DT);
    last = now;
    fps += (1 / Math.max(raw, 1e-3) - fps) * 0.05;

    if (resetAt !== null && now >= resetAt) {
      resetAt = null;
      scoreAndReset('collision');
    }
    if (!paused && resetAt === null) {
      let remaining = raw;
      while (remaining > 1e-6 && resetAt === null) {
        const h = Math.min(FIXED_DT, remaining);
        remaining -= h;
        step(h);
      }
    }

    bestDistance = Math.max(bestDistance, distance());
    rig.update(raw);
    world.render();
    hud.update(state());
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

void main();
