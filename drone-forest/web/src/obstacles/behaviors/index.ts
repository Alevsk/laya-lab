/**
 * Behaviours are policies that move an obstacle; an obstacle depends on `ObstacleBehavior`
 * from the contract and never on a concrete policy, so swapping how birds fly touches nothing
 * but the factory line that constructs the behaviour.
 *
 * Built-ins:
 *   StaticBehavior        scenery that never moves (trees, rocks)
 *   RandomWanderBehavior  piecewise-constant random flight inside the corridor (birds)
 *
 * Extension point — an AI-driven policy
 * ------------------------------------
 * A behaviour that consults an external policy (the Laya microservice, a learned flocking
 * model, another engine) plugs in through the same `ObstacleBehavior` contract. The only
 * added consideration is that a remote call is asynchronous while `update` must be
 * synchronous and run every frame, so the policy shape is: request occasionally, keep flying
 * on the last answer meanwhile. The interface such a behaviour would implement:
 *
 *   interface AiBehavior extends ObstacleBehavior {
 *     /** Serialise what the policy may see (own state + drone state + neighbours). *\/
 *     observe(obstacle: Obstacle, ctx: WorldContext): AiObservation;
 *     /** Fire-and-forget request; the answer lands in `apply` when it arrives. *\/
 *     request(observation: AiObservation): void;
 *     /** Turn the policy's answer (e.g. a discrete manoeuvre label) into a target velocity. *\/
 *     apply(answer: AiAnswer, obstacle: Obstacle): void;
 *   }
 *
 * `update` then becomes: if `ctx.time >= nextRequestAt` → `request(observe(...))`; always
 * integrate the current velocity and apply the same wall / altitude reflection that
 * `RandomWanderBehavior` uses (extract it if two policies share it). The transport is injected
 * into the constructor (a `DecisionSource`-like object with `decide(observation): Promise`),
 * so unit tests hand it a stub and the game hands it the WebSocket client from `decision/`.
 * Registering it is one line in the factory that should use it:
 *   `new BirdFactory({ behavior: () => new FlockAiBehavior(transport) })`.
 */
export { StaticBehavior, STATIC_BEHAVIOR } from './static';
export { RandomWanderBehavior, type RandomWanderOptions } from './random-wander';
