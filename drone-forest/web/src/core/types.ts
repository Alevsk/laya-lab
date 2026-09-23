/**
 * The contracts every module in this game depends on — and the only thing they may depend
 * on from each other. Mirrors `server/schemas.py` field for field; if you change one, change
 * the other and bump PROTOCOL_VERSION in both.
 *
 * Dependency inversion in practice:
 *   - the game loop depends on `DecisionSource`, never on a concrete Laya client
 *   - the world depends on `Obstacle` and `ObstacleFactory`, never on Tree/Rock/Bird
 *   - an obstacle depends on `ObstacleBehavior`, never on a concrete wander/flock/AI policy
 */
import type * as THREE from 'three';

export const PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------- actions & sensing

export const ACTIONS = ['forward', 'bank_left', 'bank_right', 'climb', 'descend', 'brake'] as const;
export type ActionName = (typeof ACTIONS)[number];

export type ObstacleKind = 'tree' | 'rock' | 'bird' | 'ground' | 'wall';

export interface Vec3 { x: number; y: number; z: number }

export interface Ray {
  name: string;
  bearing_deg: number;     // left(-) / right(+) of heading
  elevation_deg: number;   // up(+) / down(-)
  distance: number;        // metres to first hit, or max_range if clear
  hit: ObstacleKind | null;
  max_range: number;
}

export interface DroneState {
  position: Vec3;
  velocity: Vec3;
  heading_deg: number;
  altitude: number;
  speed: number;
}

export interface Nearest {
  kind: ObstacleKind;
  distance: number;
  bearing_deg: number;
  elevation_deg: number;
  closing_speed: number;
}

export interface Bounds { altitude_min: number; altitude_max: number }

/** Everything the drone knows at one instant. Built by `sensors/`, consumed by a `DecisionSource`. */
export interface SensorFrame {
  protocol: number;
  frame_id: number;
  t: number;
  drone: DroneState;
  rays: Ray[];               // exactly RAY_SPEC, in order
  nearest: Nearest | null;
  bounds: Bounds;
  last_action: ActionName | null;
}

export interface Decision {
  action: ActionName;
  engine: string;
  frame_id: number;
  decision_id: number;
  latency_ms: number;        // the engine's own think time
  confidence?: number | null;
  probabilities?: Record<string, number> | null;
  collision_imminent?: number | null;
  urgency?: number | null;
  reason?: string;
  option_index?: number | null;
  option_order?: string[] | null;
}

export interface GameEvent {
  type: 'collision' | 'near_miss' | 'reset' | 'engine_switch' | 'score';
  frame_id: number;
  t: number;
  obstacle_kind?: ObstacleKind | null;
  details?: Record<string, unknown>;
}

// ---------------------------------------------------------------- decision source (DIP)

export interface DecisionStats {
  engine: string;
  decisions: number;
  p50_ms: number;            // round-trip as seen by the game (network + think)
  p95_ms: number;
  think_p50_ms: number;      // engine-reported latency
  inflight: boolean;
  connected: boolean;
}

/**
 * Where decisions come from. The loop calls `decide` at most once per `intervalMs` and keeps
 * flying on the last action until the next answer arrives — a slow source loses ticks, it does
 * not stall the game. Implementations: `RemoteEngineSource` (WebSocket to the service) and
 * `LocalHeuristicSource` (in-browser fallback so the game runs with no server at all).
 */
export interface DecisionSource {
  readonly name: string;
  connect(): Promise<void>;
  decide(frame: SensorFrame): Promise<Decision>;
  report(event: GameEvent): void;
  stats(): DecisionStats;
  dispose(): void;
}

// ---------------------------------------------------------------- world & obstacles (DIP)

export interface Rng { next(): number; range(lo: number, hi: number): number; pick<T>(xs: readonly T[]): T }

export interface WorldContext {
  time: number;
  dt: number;
  scene: THREE.Scene;
  rng: Rng;
  bounds: Bounds & { half_width: number; corridor_length: number };
  drone: { position: THREE.Vector3; velocity: THREE.Vector3; heading_deg: number };
}

/** A policy that moves an obstacle. Static trees use `StaticBehavior`; birds use `RandomWander`; a future AI policy plugs in here. */
export interface ObstacleBehavior {
  readonly name: string;
  update(obstacle: Obstacle, ctx: WorldContext): void;
}

export interface Obstacle {
  readonly id: number;
  readonly kind: ObstacleKind;
  readonly object3d: THREE.Object3D;
  /** bounding-sphere radius used for collision and ray hits */
  readonly radius: number;
  readonly position: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  behavior: ObstacleBehavior;
  update(ctx: WorldContext): void;
  /** true when the obstacle has fallen behind the drone and can be recycled */
  isExpired(ctx: WorldContext): boolean;
  dispose(): void;
}

export interface SpawnOptions { position: THREE.Vector3; scale?: number }

export interface ObstacleFactory {
  readonly kind: ObstacleKind;
  create(ctx: WorldContext, opts: SpawnOptions): Obstacle;
}

/** Decides what to spawn, where and when. The world depends on this, not on any factory. */
export interface Spawner {
  update(ctx: WorldContext, obstacles: Obstacle[]): Obstacle[];   // returns newly spawned
}

// ---------------------------------------------------------------- collision

export interface CollisionResult {
  hit: Obstacle | null;
  nearMiss: Obstacle | null;   // within nearMissRadius but not touching
}
