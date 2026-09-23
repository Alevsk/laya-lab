/**
 * Where decisions come from. The game depends on `DecisionSource` (core/types) and on the
 * factories here — never on a concrete engine. Adding a source is one new module exporting a
 * `DecisionSource` and one line in this file.
 */
export { createRemoteSource, RemoteEngineSource } from './remote';
export type { RemoteOptions } from './remote';
export { createLocalHeuristicSource, LocalHeuristicSource, LOCAL_ENGINE_NAME } from './local';
export { heuristicPolicy, sectorClearances, clearanceHorizon, initialHeuristicState, DEFAULT_HEURISTIC_CONFIG } from './heuristic';
export type { HeuristicConfig, HeuristicState, HeuristicVerdict, Sectors } from './heuristic';
export { createDecisionLoop } from './loop';
export type { DecisionLoop, DecisionLoopOptions, DecisionMeta, LoopMode, LoopStats } from './loop';
export { runArena, parseArenaParams, ARENA_RESULT_PREFIX } from './arena';
export type { ArenaDeps, ArenaDrone, ArenaMode, ArenaParams, ArenaProgress, ArenaResult, ArenaWorld } from './arena';
export { LatencyWindow, RateMeter, nowMs } from './stats';
export type { ExtendedDecisionStats } from './stats';
