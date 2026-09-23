/**
 * The factory registry: the ONE place that knows which obstacle kinds exist. The world asks
 * for `defaultFactories()` and gets fresh instances (each world owns and disposes its own GPU
 * resources); nothing outside this file names TreeFactory, RockFactory or BirdFactory.
 *
 * Adding a kind: write `factories/<kind>.ts` and add one `registerFactory` line here (or call
 * `registerFactory` from anywhere before the world is created).
 */
import type { ObstacleFactory } from '../core/types';
import { DEFAULT_CORRIDOR, type CorridorFrame } from './corridor';
import { BirdFactory, RockFactory, TreeFactory, type FactoryOptions } from './factories';

export type FactoryProvider = (options: Required<FactoryOptions>) => ObstacleFactory;

const providers: FactoryProvider[] = [];

/** Register a kind. A later provider for the same `kind` replaces the earlier one. */
export function registerFactory(provider: FactoryProvider): void {
  providers.push(provider);
}

/** Fresh instances of every registered factory, in registration order, de-duplicated by kind. */
export function defaultFactories(options: { corridor?: CorridorFrame } = {}): ObstacleFactory[] {
  const resolved: Required<FactoryOptions> = { corridor: options.corridor ?? DEFAULT_CORRIDOR };
  const byKind = new Map<ObstacleFactory['kind'], ObstacleFactory>();
  for (const provide of providers) {
    const factory = provide(resolved);
    const replaced = byKind.get(factory.kind);
    if (replaced !== undefined) disposeFactories([replaced]);
    byKind.set(factory.kind, factory);
  }
  return [...byKind.values()];
}

/** Calls `dispose` on every factory that has one. */
export function disposeFactories(factories: readonly ObstacleFactory[]): void {
  for (const factory of factories) {
    if ('dispose' in factory && typeof factory.dispose === 'function') factory.dispose();
  }
}

// ---- built-in kinds: one line each -------------------------------------------------------
registerFactory((o) => new TreeFactory(o));
registerFactory((o) => new RockFactory(o));
registerFactory((o) => new BirdFactory(o));
