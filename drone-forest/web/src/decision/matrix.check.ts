/** `make check-web` runs this: the matrix iterates every combination exactly once, innermost seed first. */
import { matrixProgress, nextMatrixUrl } from './matrix.ts';

const assert = (c: boolean, m: string): void => { if (!c) throw new Error(m); };
let search = 'arena=quality&engines=heuristic,laya-ft&difficulties=3,5&seeds=7-8&modes=quality,realtime&seconds=45&engine=heuristic&difficulty=3&seed=7';
const seen: string[] = [];
for (let i = 0; i < 100; i++) {
  const q = new URLSearchParams(search);
  seen.push(`${q.get('arena')}/${q.get('engine')}/${q.get('difficulty')}/${q.get('seed')}`);
  const p = matrixProgress(search);
  assert(p.index === i + 1 && p.total === 16, `progress ${p.index}/${p.total} at step ${i}`);
  const next = nextMatrixUrl(search);
  if (next === null) break;
  search = next;
}
assert(seen.length === 16, `expected 16 combinations, ran ${seen.length}`);
assert(new Set(seen).size === 16, 'every combination must be distinct');
assert(seen[0] === 'quality/heuristic/3/7' && seen[1] === 'quality/heuristic/3/8', 'seed is the innermost dimension');
assert(seen[2] === 'quality/heuristic/5/7', 'difficulty advances after seeds are exhausted');
assert(seen[4] === 'quality/laya-ft/3/7', 'engine advances after difficulties');
assert(seen[8] === 'realtime/heuristic/3/7', 'mode advances last');
assert(nextMatrixUrl('arena=quality&seeds=7-9&seed=9') === null, 'plain seed chain ends at the last seed');
assert(nextMatrixUrl('arena=quality&seeds=7-9&seed=7')!.includes('seed=8'), 'plain seed chain still works');
assert(nextMatrixUrl('arena=quality&seed=7') === null, 'no matrix params: no chaining');
console.log(`matrix ok: ${seen.length} combinations, order ${seen.slice(0, 5).join(' > ')} ...`);
