/**
 * HUD: a DOM overlay showing what the engine is doing and how well it is keeping up. It is a
 * pure view — it renders whatever `HudState` it is handed and raises callbacks for the engine
 * selector and the reset button; it never touches the world, the drone or a source.
 *
 * DOM nodes are built once; `update` writes into them at most every RENDER_EVERY_MS so calling
 * it every frame is fine.
 */
import './hud.css';

import { ACTIONS } from '../core/types';
import type { ActionName, Decision, DecisionStats, SensorFrame } from '../core/types';
import type { ArenaProgress, ArenaResult, LoopStats } from '../decision';
import type { ExtendedDecisionStats } from '../decision';

/**
 * What the HUD renders. Required fields are the minimum for a meaningful overlay; the optional
 * ones light up extra rows when present. Shaped so the game's own run-state object can be
 * passed straight in.
 */
export interface HudState {
  /** name of the source in effect (e.g. 'remote:laya', 'local-heuristic') */
  engine: string;
  /** selector value (local | heuristic | laya | random); falls back to `engine` */
  requestedEngine?: string;
  connected?: boolean;
  /** a status line about the engine source: loading, fallback in effect, etc. */
  notice?: string | null;
  action: ActionName;
  lastDecision: Decision | null;
  stats: DecisionStats | ExtendedDecisionStats;
  loop: LoopStats;
  distance: number;
  collisions: number;
  nearMisses: number;
  bestDistance?: number;
  /** the latest sensor frame, for altitude / speed / nearest; `altitude` and `speed` may be given directly instead */
  frame?: SensorFrame | null;
  altitude?: number;
  speed?: number;
  fps?: number;
  camera?: string;
  paused?: boolean;
  resetting?: boolean;
  raysVisible?: boolean;
  episode?: number;
  episodeTime?: number;
  simTime?: number;
  obstacles?: number;
  seed?: number;
  hz?: number;
  arena?: string;
  seconds?: number;
  lateral?: number;
  difficulty?: number;
  /** engine-recommended speed, m/s; null = cruise */
  targetSpeed?: number | null;
}

export interface Hud {
  update(state: HudState): void;
  onEngineChange(cb: (engine: string) => void): void;
  onReset(cb: () => void): void;
  /** Reflect an engine change that did not come from the selector (URL param, server info). */
  setEngine(engine: string): void;
  setEngines(engines: readonly string[]): void;
  onDifficultyChange(cb: (level: number) => void): void;
  setDifficulty(level: number): void;
  /** Hide/show every panel (H). Returns the new visibility. */
  togglePanels(): boolean;
  /** Hide/show just the arena panel (A). Returns the new visibility. */
  toggleArena(): boolean;
  setArenaProgress(progress: ArenaProgress): void;
  showArena(result: ArenaResult | null): void;
  flash(kind: 'collision' | 'near_miss'): void;
  dispose(): void;
}

export interface HudOptions {
  engines?: readonly string[];
  /** key → what it does, for the help panel */
  keys?: ReadonlyArray<readonly [key: string, label: string]>;
  renderEveryMs?: number;
}

export const DEFAULT_ENGINE_OPTIONS = ['local', 'heuristic', 'laya', 'laya-ft', 'random'] as const;
export const DEFAULT_KEYS: ReadonlyArray<readonly [string, string]> = [
  ['R', 'rays'],
  ['C', 'camera'],
  ['Space', 'reset'],
  ['P', 'pause'],
  ['H', 'panels'],
  ['A', 'arena'],
  ['Esc', 'stop'],
];
const RENDER_EVERY_MS = 100;

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const ms = (x: number): string => (x >= 100 ? x.toFixed(0) : x.toFixed(1));
const pct = (n: number, d: number): string => (d > 0 ? `${((100 * n) / d).toFixed(0)}%` : '–');

interface Row { root: HTMLElement; v: HTMLElement }
function row(parent: HTMLElement, key: string): Row {
  const root = el('div', 'row');
  root.append(el('span', 'k', key));
  const v = el('span', 'v', '–');
  root.append(v);
  parent.append(root);
  return { root, v };
}

interface Bar { root: HTMLElement; fill: HTMLElement; label: Row }
function bar(parent: HTMLElement, key: string, cls = ''): Bar {
  const label = row(parent, key);
  const root = el('div', `bar ${cls}`.trim());
  const fill = el('div', 'fill');
  root.append(fill);
  parent.append(root);
  return { root, fill, label };
}

interface HistRow { root: HTMLElement; fill: HTMLElement; n: HTMLElement }
function histRow(parent: HTMLElement, label: string): HistRow {
  const r = el('div', 'hrow');
  r.append(el('span', 'k', label));
  const hb = el('div', 'hbar');
  const fill = el('div', 'fill');
  hb.append(fill);
  r.append(hb);
  const n = el('span', 'n', '0');
  r.append(n);
  parent.append(r);
  return { root: r, fill, n };
}

export function createHud(root: HTMLElement, opts: HudOptions = {}): Hud {
  const renderEvery = opts.renderEveryMs ?? RENDER_EVERY_MS;
  const container = el('div', 'hud');
  root.append(container);

  const engineCbs: Array<(engine: string) => void> = [];
  const resetCbs: Array<() => void> = [];

  // ---- engine panel
  const engine = el('div', 'panel engine');
  engine.append(el('h3', undefined, 'Decision engine'));
  const select = el('select');
  engine.append(select);
  const setEngines = (names: readonly string[]): void => {
    const current = select.value;
    select.replaceChildren(...names.map((n) => Object.assign(el('option', undefined, n), { value: n })));
    if (names.includes(current)) select.value = current;
  };
  setEngines(opts.engines ?? DEFAULT_ENGINE_OPTIONS);
  select.addEventListener('change', () => engineCbs.forEach((cb) => cb(select.value)));

  const diffCbs: Array<(level: number) => void> = [];
  const diffWrap = el('div', 'row');
  diffWrap.append(el('span', 'k', 'difficulty'));
  const diffSelect = el('select');
  diffSelect.replaceChildren(...[1, 2, 3, 4, 5].map((n) => Object.assign(el('option', undefined, `${n} - ${['sparse', 'easy', 'normal', 'hard', 'brutal'][n - 1]}`), { value: String(n) })));
  diffSelect.value = '3';
  diffWrap.append(diffSelect);
  engine.append(diffWrap);
  diffSelect.addEventListener('change', () => diffCbs.forEach((cb) => cb(Number(diffSelect.value))));

  const conn = el('div', 'row');
  const dot = el('span', 'dot');
  const connText = el('span', 'v', 'connecting');
  conn.append(dot, connText);
  engine.append(conn);
  const rSource = row(engine, 'source');
  const rThink = row(engine, 'think ms');
  const rRtt = row(engine, 'round-trip p50/p95');
  const rRate = row(engine, 'decisions/s');
  const rTicks = row(engine, 'ticks met/missed');
  const rSkipped = row(engine, 'slots skipped / timeouts');
  const rErr = row(engine, 'last error');
  rErr.root.classList.add('bad');
  rErr.root.hidden = true;
  const rNotice = row(engine, 'status');
  rNotice.root.classList.add('warn');
  rNotice.root.hidden = true;
  const runLine = el('div', 'muted', '');
  engine.append(runLine);

  // ---- flight panel
  const flight = el('div', 'panel flight');
  flight.append(el('h3', undefined, 'Flight'));
  const action = el('div', 'action', 'forward');
  const reason = el('div', 'reason', '');
  flight.append(action, reason);
  const rAlt = row(flight, 'altitude');
  const rSpeed = row(flight, 'speed');
  const rTarget = row(flight, 'target speed');
  const rNearest = row(flight, 'nearest');
  const rDist = row(flight, 'distance');
  const rBest = row(flight, 'best');
  const rColl = row(flight, 'collisions');
  const rNear = row(flight, 'near-misses');
  const rConf = row(flight, 'confidence');
  const bImminent = bar(flight, 'collision imminent', 'danger');
  const bUrgency = bar(flight, 'urgency');
  const rEpisode = row(flight, 'episode');
  const rFps = row(flight, 'fps');
  const rMode = row(flight, 'camera');
  const status = el('div', 'warn', '');
  flight.append(status);
  const resetBtn = el('button', undefined, 'reset [space]');
  resetBtn.style.cssText = 'margin-top:6px;font:inherit;color:inherit;background:#121924;border:1px solid rgba(255,255,255,.12);border-radius:4px;padding:2px 8px;cursor:pointer';
  resetBtn.addEventListener('click', () => resetCbs.forEach((cb) => cb()));
  flight.append(resetBtn);

  // ---- histogram panel
  const hist = el('div', 'panel hist');
  hist.append(el('h3', undefined, 'Actions'));
  const histRows = {} as Record<ActionName, HistRow>;
  for (const a of ACTIONS) histRows[a] = histRow(hist, a);
  const bias = el('div', 'bias');
  bias.hidden = true;
  bias.append(el('h3', undefined, 'Option index (positional bias)'));
  hist.append(bias);
  const biasRows = new Map<number, HistRow>();

  // ---- keys panel
  const keys = el('div', 'panel keys');
  for (const [key, label] of opts.keys ?? DEFAULT_KEYS) {
    keys.append(el('kbd', undefined, key), el('span', undefined, `${label}\u00a0\u00a0 `));
  }

  // ---- arena panel
  const arena = el('div', 'panel arena');
  arena.hidden = true;
  arena.append(el('h3', undefined, 'Arena'));
  const arenaStatus = el('div', 'status', '');
  arena.append(arenaStatus);
  const arenaBar = el('div', 'bar progress');
  const arenaFill = el('div', 'fill');
  arenaBar.append(arenaFill);
  arena.append(arenaBar);
  const arenaTable = el('table');
  arena.append(arenaTable);
  arena.append(el('div', 'hint', 'Esc stop · Space restart · P pause · A hide this panel · H hide all panels · C camera · R rays'));

  const flashEl = el('div', 'flash');
  // left column: the engine panel with the arena panel stacked directly beneath it, so the
  // arena never overlaps the side panels and never covers the drone in the centre of the view
  const colLeft = el('div', 'col left');
  colLeft.append(engine, arena);
  container.append(flashEl, colLeft, flight, hist, keys);

  // ---- rendering
  let lastRender = -Infinity;
  let pendingState: HudState | null = null;
  let flashTimer: ReturnType<typeof setTimeout> | null = null;

  const render = (s: HudState): void => {
    const wanted = s.requestedEngine ?? s.engine;
    if (select.value !== wanted && Array.from(select.options).some((o) => o.value === wanted)) select.value = wanted;

    const src = s.stats;
    const ext = src as Partial<ExtendedDecisionStats>;
    const connected = s.connected ?? src.connected;
    dot.className = `dot ${connected ? 'on' : 'off'}`;
    connText.textContent = `${connected ? 'connected' : 'disconnected'} · ${s.engine}`;
    rSource.v.textContent = `${src.decisions} decisions${src.inflight ? ' · thinking' : ''}`;
    rThink.v.textContent = ext.think_p95_ms !== undefined ? `${ms(src.think_p50_ms)} / ${ms(ext.think_p95_ms)}` : ms(src.think_p50_ms);
    rRtt.v.textContent = `${ms(src.p50_ms)} / ${ms(src.p95_ms)}`;
    rRate.v.textContent = ext.decisions_per_s !== undefined ? ext.decisions_per_s.toFixed(1) : '–';
    const met = s.loop.ticks_met;
    const missed = s.loop.ticks_missed;
    rTicks.v.textContent = `${met}/${missed} (${pct(met, met + missed)} met)`;
    rTicks.v.className = `v ${missed > met ? 'bad' : missed > 0 ? 'warn' : 'good'}`;
    rSkipped.v.textContent = `${s.loop.slots_skipped} / ${ext.timeouts ?? 0}`;
    rErr.root.hidden = !ext.last_error;
    rErr.v.textContent = ext.last_error ?? '';
    rNotice.root.hidden = !s.notice;
    rNotice.v.textContent = s.notice ?? '';
    const runBits: string[] = [];
    if (s.seed !== undefined) runBits.push(`seed ${s.seed}`);
    if (s.hz !== undefined) runBits.push(`${s.hz} Hz`);
    if (s.arena !== undefined) runBits.push(s.arena);
    if (s.obstacles !== undefined) runBits.push(`${s.obstacles} obstacles`);
    runLine.textContent = runBits.join(' · ');
    runLine.hidden = runBits.length === 0;

    const d = s.lastDecision;
    const a: ActionName = s.action;
    action.textContent = a;
    action.className = `action ${a}`;
    reason.textContent = d?.reason ?? '';
    rConf.v.textContent = d?.confidence !== null && d?.confidence !== undefined ? d.confidence.toFixed(2) : '–';

    const f = s.frame ?? null;
    const altitude = s.altitude ?? f?.drone.altitude;
    const speed = s.speed ?? f?.drone.speed;
    rAlt.v.textContent = altitude !== undefined ? `${altitude.toFixed(1)} m` : '–';
    rSpeed.v.textContent = speed !== undefined ? `${speed.toFixed(1)} m/s` : '–';
    rTarget.v.textContent = s.targetSpeed != null ? `${s.targetSpeed.toFixed(1)} m/s` : 'cruise';
    rNearest.root.hidden = !f;
    rNearest.v.textContent = f?.nearest
      ? `${f.nearest.kind} ${f.nearest.distance.toFixed(1)}m @${f.nearest.bearing_deg.toFixed(0)}°${f.nearest.closing_speed > 0.5 ? ' closing' : ''}`
      : '–';
    rDist.v.textContent = `${s.distance.toFixed(0)} m`;
    rBest.root.hidden = s.bestDistance === undefined;
    rBest.v.textContent = s.bestDistance !== undefined ? `${s.bestDistance.toFixed(0)} m` : '–';
    rColl.v.textContent = String(s.collisions);
    rNear.v.textContent = String(s.nearMisses);
    rEpisode.root.hidden = s.episode === undefined;
    rEpisode.v.textContent = s.episode !== undefined ? `#${s.episode}${s.episodeTime !== undefined ? ` · ${s.episodeTime.toFixed(0)} s` : ''}` : '–';
    rFps.v.textContent = s.fps !== undefined ? s.fps.toFixed(0) : '–';
    rFps.root.hidden = s.fps === undefined;
    rMode.root.hidden = s.camera === undefined && s.raysVisible === undefined;
    rMode.v.textContent = [s.camera, s.raysVisible ? 'rays on' : null].filter(Boolean).join(' · ');
    status.textContent = s.paused ? 'PAUSED' : s.resetting ? 'resetting…' : '';
    status.hidden = !s.paused && !s.resetting;

    const imminent = d?.collision_imminent;
    bImminent.root.hidden = bImminent.label.root.hidden = imminent === null || imminent === undefined;
    if (imminent !== null && imminent !== undefined) {
      bImminent.fill.style.width = `${Math.round(100 * Math.min(1, Math.max(0, imminent)))}%`;
      bImminent.label.v.textContent = imminent.toFixed(2);
    }
    const urgency = d?.urgency;
    bUrgency.root.hidden = bUrgency.label.root.hidden = urgency === null || urgency === undefined;
    if (urgency !== null && urgency !== undefined) {
      bUrgency.fill.style.width = `${Math.round((100 * Math.min(3, Math.max(0, urgency))) / 3)}%`;
      bUrgency.label.v.textContent = `${urgency.toFixed(2)} / 3`;
    }

    const h = s.loop.action_histogram;
    const total = ACTIONS.reduce((acc, k) => acc + (h[k] ?? 0), 0);
    for (const k of ACTIONS) {
      const n = h[k] ?? 0;
      histRows[k].fill.style.width = total > 0 ? `${Math.round((100 * n) / total)}%` : '0';
      histRows[k].n.textContent = String(n);
    }
    const oh = s.loop.option_index_hist;
    bias.hidden = !oh;
    if (oh) {
      const idx = Object.keys(oh).map(Number).sort((x, y) => x - y);
      const oTotal = idx.reduce((acc, i) => acc + (oh[i] ?? 0), 0);
      let added = false;
      for (const i of idx) {
        let r = biasRows.get(i);
        if (!r) {
          r = histRow(bias, `#${i}`);
          biasRows.set(i, r);
          added = true;
        }
        const n = oh[i] ?? 0;
        r.fill.style.width = oTotal > 0 ? `${Math.round((100 * n) / oTotal)}%` : '0';
        r.n.textContent = String(n);
      }
      // slots appear in the order the model first picks them; keep the rows in slot order
      if (added) for (const i of idx) bias.append(biasRows.get(i)!.root);
    }
  };

  const update = (s: HudState): void => {
    pendingState = s;
    const now = performance.now();
    if (now - lastRender < renderEvery) return;
    lastRender = now;
    render(pendingState);
  };

  const arenaRows = (pairs: Array<[string, string]>): void => {
    arenaTable.replaceChildren(
      ...pairs.map(([k, v]) => {
        const tr = el('tr');
        tr.append(el('td', undefined, k), el('td', undefined, v));
        return tr;
      }),
    );
  };

  const histText = (h: Record<string, number>): string =>
    Object.entries(h)
      .filter(([, n]) => n > 0)
      .map(([k, n]) => `${k}:${n}`)
      .join(' ');

  return {
    update,
    onEngineChange: (cb) => {
      engineCbs.push(cb);
    },
    onReset: (cb) => {
      resetCbs.push(cb);
    },
    setEngine: (name) => {
      if (Array.from(select.options).some((o) => o.value === name)) select.value = name;
    },
    setEngines,
    togglePanels: () => {
      container.classList.toggle('panels-hidden');
      return !container.classList.contains('panels-hidden');
    },
    toggleArena: () => {
      arena.hidden = !arena.hidden;
      return !arena.hidden;
    },
    onDifficultyChange: (cb) => {
      diffCbs.push(cb);
    },
    setDifficulty: (level) => {
      diffSelect.value = String(level);
    },
    setArenaProgress: (p) => {
      arena.hidden = false;
      arenaStatus.className = 'status';
      const endless = p.seconds <= 0;
      arenaStatus.textContent = endless
        ? `running · ${p.t.toFixed(0)} s · until a crash or Esc · ${p.action}`
        : `running · ${p.t.toFixed(0)} / ${p.seconds} s · ${p.action}`;
      arenaFill.style.width = endless ? '100%' : `${Math.round((100 * Math.min(p.t, p.seconds)) / p.seconds)}%`;
      arenaRows([
        ['distance', `${p.distance.toFixed(0)} m`],
        ['collisions', String(p.collisions)],
        ['near-misses', String(p.near_misses)],
      ]);
    },
    showArena: (r) => {
      if (!r) {
        arena.hidden = true;
        return;
      }
      arena.hidden = false;
      arenaStatus.className = 'status done';
      arenaStatus.textContent = `done · ${r.engine} · ${r.mode} · seed ${r.seed} · ${r.ended_by === 'collision' ? 'crashed' : r.ended_by === 'stop' ? 'stopped (Esc)' : 'time up'} after ${r.simulated_s.toFixed(0)} s`;
      arenaFill.style.width = '100%';
      const rows: Array<[string, string]> = [
        ['simulated', `${r.seconds} s (${r.sim_steps} steps, ${(r.wall_ms / 1000).toFixed(1)} s wall)`],
        ['distance', `${r.distance.toFixed(0)} m (displacement ${r.displacement.toFixed(0)} m)`],
        ['collisions', String(r.collisions)],
        ['near-misses', String(r.near_misses)],
        ['ticks met / missed', `${r.ticks_met} / ${r.ticks_missed} (${pct(r.ticks_met, r.ticks_met + r.ticks_missed)} met)`],
        ['slots skipped', String(r.slots_skipped)],
        ['decisions', String(r.decisions)],
        ['round-trip p50 / p95', `${ms(r.decision_p50_ms)} / ${ms(r.decision_p95_ms)} ms`],
        ['think p50', `${ms(r.think_p50_ms)} ms`],
        ['actions', histText(r.action_histogram)],
      ];
      if (r.option_index_hist) rows.push(['option index', histText(r.option_index_hist)]);
      arenaRows(rows);
    },
    flash: (kind) => {
      flashEl.className = `flash ${kind} show`;
      if (flashTimer) clearTimeout(flashTimer);
      flashTimer = setTimeout(() => {
        flashEl.className = `flash ${kind}`;
      }, 60);
    },
    dispose: () => {
      if (flashTimer) clearTimeout(flashTimer);
      container.remove();
      engineCbs.length = 0;
      resetCbs.length = 0;
    },
  };
}
