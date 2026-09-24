/**
 * The evaluation matrix, driven by the URL. Given
 *
 *   ?arena=quality&engines=heuristic,laya-ft&difficulties=3,5&seeds=7-9&modes=quality,realtime
 *
 * the page runs one arena, then reloads itself with the NEXT combination until every one has
 * run: seed is the innermost dimension, then difficulty, then engine, then mode. Each result is
 * reported to the service as a `score` event, so `make scoreboard` turns the whole matrix into
 * one table. A dimension whose parameter is absent simply does not iterate, which is how the
 * older `seeds=A-B` chaining keeps working unchanged.
 */

export type ArenaModeName = 'quality' | 'realtime';

interface Dim {
  key: string;
  values: string[] | null;
  current: string;
}

function list(q: URLSearchParams, key: string): string[] | null {
  const raw = q.get(key);
  if (!raw) return null;
  const xs = raw.split(',').map((s) => s.trim()).filter(Boolean);
  return xs.length ? xs : null;
}

function seedRange(q: URLSearchParams): string[] | null {
  const raw = q.get('seeds');
  if (!raw) return null;
  const m = /^(\d+)-(\d+)$/.exec(raw.trim());
  if (!m) return null;
  const a = Number(m[1]);
  const b = Number(m[2]);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b < a || b - a > 10_000) return null;
  return Array.from({ length: b - a + 1 }, (_, i) => String(a + i));
}

/** The dimensions in iteration order, innermost first, with their current values. */
export function matrixDims(search: string): Dim[] {
  const q = new URLSearchParams(search);
  return [
    { key: 'seed', values: seedRange(q), current: q.get('seed') ?? '7' },
    { key: 'difficulty', values: list(q, 'difficulties'), current: q.get('difficulty') ?? '3' },
    { key: 'engine', values: list(q, 'engines'), current: q.get('engine') ?? 'heuristic' },
    { key: 'arena', values: list(q, 'modes'), current: q.get('arena') ?? 'quality' },
  ];
}

/** The search string for the next combination, or null when the matrix is complete. */
export function nextMatrixUrl(search: string): string | null {
  const q = new URLSearchParams(search);
  const dims = matrixDims(search);
  for (let i = 0; i < dims.length; i++) {
    const dim = dims[i]!;
    if (!dim.values) continue;
    const idx = dim.values.indexOf(dim.current);
    if (idx >= 0 && idx < dim.values.length - 1) {
      q.set(dim.key, dim.values[idx + 1]!);
      for (let j = 0; j < i; j++) {
        const inner = dims[j]!;
        if (inner.values) q.set(inner.key, inner.values[0]!);
      }
      return q.toString();
    }
  }
  return null;
}

/** How many combinations the matrix has, and which one this is (1-based), for the HUD/console. */
export function matrixProgress(search: string): { index: number; total: number } {
  const dims = matrixDims(search).filter((d) => d.values);
  let total = 1;
  let index = 0;
  let stride = 1;
  for (const d of dims) {
    const n = d.values!.length;
    const at = Math.max(0, d.values!.indexOf(d.current));
    index += at * stride;
    stride *= n;
    total *= n;
  }
  return { index: index + 1, total };
}
