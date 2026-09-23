/**
 * `LocalHeuristicSource`: a DecisionSource that runs `heuristicPolicy` in the page. Zero
 * network, sub-millisecond, always "connected" — the fallback that keeps the game playable and
 * the baseline every remote engine is measured against.
 */
import type { Decision, DecisionSource, GameEvent, SensorFrame } from '../core/types';
import { DEFAULT_HEURISTIC_CONFIG, heuristicPolicy, initialHeuristicState } from './heuristic';
import type { HeuristicConfig, HeuristicState } from './heuristic';
import { LatencyWindow, RateMeter, nowMs } from './stats';
import type { ExtendedDecisionStats } from './stats';

export const LOCAL_ENGINE_NAME = 'local-heuristic';

export class LocalHeuristicSource implements DecisionSource {
  readonly name = LOCAL_ENGINE_NAME;
  private state: HeuristicState = initialHeuristicState();
  private decisionId = 0;
  private readonly think = new LatencyWindow();
  private readonly rate = new RateMeter();

  constructor(private readonly cfg: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG) {}

  connect(): Promise<void> {
    return Promise.resolve();
  }

  decide(frame: SensorFrame): Promise<Decision> {
    const t0 = nowMs();
    const verdict = heuristicPolicy(frame, this.state, this.cfg);
    const latency = nowMs() - t0;
    this.think.push(latency);
    this.rate.mark(t0);
    return Promise.resolve({
      action: verdict.action,
      engine: this.name,
      frame_id: frame.frame_id,
      decision_id: ++this.decisionId,
      latency_ms: latency,
      confidence: null,
      probabilities: verdict.scores,
      collision_imminent: null,
      urgency: null,
      reason: verdict.reason,
      option_index: null,
      option_order: null,
    });
  }

  report(event: GameEvent): void {
    if (event.type === 'reset') this.state = initialHeuristicState();
  }

  stats(): ExtendedDecisionStats {
    return {
      engine: this.name,
      decisions: this.think.count,
      p50_ms: this.think.percentile(0.5),
      p95_ms: this.think.percentile(0.95),
      think_p50_ms: this.think.percentile(0.5),
      think_p95_ms: this.think.percentile(0.95),
      decisions_per_s: this.rate.perSecond(nowMs()),
      inflight: false,
      connected: true,
      skipped: 0,
      errors: 0,
      timeouts: 0,
      stale: 0,
      reconnects: 0,
      engines: [this.name],
      last_error: null,
    };
  }

  dispose(): void {
    this.think.reset();
    this.rate.reset();
  }
}

export function createLocalHeuristicSource(cfg?: HeuristicConfig): DecisionSource {
  return new LocalHeuristicSource(cfg);
}
