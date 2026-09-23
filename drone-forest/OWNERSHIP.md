# Who owns what (build phase)

Files are partitioned so parallel agents never edit the same file.

| path | owner | notes |
|---|---|---|
| `server/schemas.py`, `server/engines/base.py`, `server/fastload.py`, `web/src/core/*` | **contracts — frozen** | Edit only with a note in your report; both sides must stay in sync |
| `server/app.py`, `server/telemetry.py`, `server/engines/heuristic_engine.py`, `server/engines/random_engine.py`, `server/config.py`, `tests/conftest.py`, `tests/test_server*.py`, `tests/test_heuristic*.py` | **server-core** | FastAPI app, WS protocol, telemetry, baseline engines |
| `server/engines/laya_engine.py`, `server/engines/framing.py`, `tests/test_laya_engine.py` | **server-laya** | the Laya engine and its semantic framing |
| `web/src/world/*`, `web/src/drone/*`, `web/src/main.ts`, `web/public/*` | **web-world** | scene, terrain, lighting, post-fx, drone model + physics + camera, loop wiring |
| `web/src/obstacles/*` | **web-obstacles** | Obstacle/Behavior/Factory/Spawner implementations + registry |
| `web/src/sensors/*`, `web/src/decision/*`, `web/src/hud/*` | **web-decision** | raycast sensors → SensorFrame, DecisionSource impls, HUD |
| `Makefile`, `README.md`, `docs/*`, `tests/test_telemetry_cli.py`, `pyproject.toml`, `web/vite.config.ts`, `web/index.html` | **integration** | after everything builds; may touch any non-contract file to make the whole run |

Cross-module calls go through `web/src/core/types.ts` interfaces and the factory functions
each module exports from its `index.ts`. The exports as built (a superset of the minimum that was
specified before the build):

- `world/index.ts`   → `createWorld(opts): World` with `{ scene, camera, renderer, ctx, obstacles, rng, sunDirection, follow(target), add(o), remove(o), update(dt), render(), resize(), flash(), reset({seed?, spawner?}), dispose() }`; `DEFAULT_BOUNDS`
- `drone/index.ts`   → `createDrone(scene, opts?): Drone` with `{ object3d, body, position, velocity, headingDeg, speed, action, radius, nearMissRadius, applyAction(a, dt), update(dt, ctx), collide(obstacles): CollisionResult, reset(), dispose() }`; `createChaseRig(camera, drone)`
- `obstacles/index.ts` → `createSpawner(rng, factories?, options?): ForestSpawner`, `defaultFactories(options?): ObstacleFactory[]`, `registerFactory(provider)`, `disposeFactories(fs)`
- `sensors/index.ts` → `createSensors(world, drone, opts?): { frame(frameId, t, lastAction): SensorFrame; lastFrame(); basis() }`, `createRayDebug(scene)`
- `decision/index.ts` → `createRemoteSource(url, engine, opts?): RemoteEngineSource` (adds `setEngine`, `connected`, `onInfo`/`onError` options), `createLocalHeuristicSource(): DecisionSource`, `createDecisionLoop(source, intervalMs, opts?): DecisionLoop` with `{ tick, due, current, last, awaitDecision, stats(): LoopStats, mode, setMode, setSource, source, reset }`, `runArena(params, deps)`, `parseArenaParams(search)`
- `hud/index.ts`     → `createHud(root, opts?): Hud` with `{ update(state), onEngineChange(cb), onReset(cb), setEngine, setEngines, setArenaProgress, showArena, flash, dispose }`

Wire-protocol note (server-core → web-decision): the service sends `info` on connect only after
the requested engine is loaded and warm, again after a successful `set_engine`, and an `error`
followed by `info` when `?engine=` names an unknown engine. The client treats `info` as "ready"
and accepts it at any time.
