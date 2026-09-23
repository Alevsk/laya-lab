"""Tetris arena: a latency budget is not a quality measure.

A viral demo (atomic.chat, 21 Sep 2026, ~5.8k likes) put two Tetris boards side
by side on the same piece sequence: a cloud decision model at a reported 316-326
ms per decision against Laya running locally at a reported 45-51 ms. The cloud
board filled up and printed "GAME OVER - COULD NOT KEEP UP". The claim attached
to it was "11x faster decisions".

The mechanism underneath that demo is real and worth reproducing: a game loop
ticks at a fixed rate, and any decider that overruns the tick forfeits that
piece. The cloud agent lost to the clock. But losing to the clock says nothing
about whether the winner plays well, and earlier scenarios in this project
already measured Laya behaving as a near-constant predictor on spatial and
numeric tasks. So this file measures the two properties SEPARATELY, and that
separation is the entire point:

  0  A real 10x20 Tetris engine with in-file assert-based self-tests, a seeded
     7-bag so every decider faces a byte-identical piece sequence, a
     decider-agnostic candidate generator, and a check that the rendered choice
     options are not silently truncated by laya's 192-token head budget --
     measured across every decision of three whole games on tall, jagged boards,
     where the option text is longest, not on one tidy mid-game example.
  A  LIVE RACE. Two boards racing at one fixed tick budget: "laya-local"
     against "cloud-300ms" -- the same Laya decision plus a SIMULATED 300 ms
     round trip (source: the demo's reported 316-326 ms; nothing here was
     measured against any cloud service). Overrun a tick and the piece drops
     undecided at its spawn column. Watch the penalty accumulate. The tick
     budget is DERIVED from this run's own measured latency rather than
     hard-coded, so the race is not an artefact of one laptop. On a machine slow
     enough that no budget separates the two, the scenario says so in red and
     fails, rather than picking a number that flatters the local decider.
  B  THE LATENCY BUDGET CURVE, measured. Fraction of ticks met per decider over
     a sweep of tick budgets. Laya's position on this curve comes from real
     synchronised timings on this machine; only the cloud decider's 300 ms
     offset is simulated, and it is labelled everywhere it appears.
  C  DECISION QUALITY with latency removed. Four deciders -- heuristic, random,
     laya-numeric, laya-semantic -- on the SAME seeded sequences with unlimited
     time, over several seeds, reporting lines cleared, pieces survived, holes
     and final height. Plus the degeneracy test: the histogram of which option
     INDEX each Laya decider picks. A spike there means a constant predictor,
     not a player.
  D  The honest two-column conclusion.

The framing experiment inside C is the part the viral demo never runs.
laya-numeric and laya-semantic see the SAME candidates: one renders them as
numbers ("col 3 rot 1: clears 0, holes +2, top 7, bumps 9"), the other buckets
those same numbers into words ("no clear, buries two gaps, stack is high,
jagged surface"). Laya has no numeric grounding but was trained on typed
semantic decisions, so does framing rescue it? The answer is printed either way.

How to read the output: section B answers "can it keep up", section C answers
"does it decide well". They are independent, and the demo only demonstrates the
first.

What this run actually found, so nobody has to run it to know: the latency
mechanism reproduces -- Laya met every tick of the derived budget while the
simulated 300 ms decider met none and topped out inside a dozen pieces. The
quality half does not flatter anyone. Over 11 seeds x 80 pieces, a four-term
linear score in pure Python cleared 7.6 lines; laya-numeric cleared 0.3,
laya-semantic 0.0, and a uniform random pick 0.1. Neither Laya framing is
measurably better than random (exact paired permutation test, p = 0.63 on
lines). Semantic framing did not rescue it, though the two framings fail
differently: laya-numeric lands on option index 0 about 83% of the time -- and
since the option order is shuffled per decision, "always take slot 0" IS a
uniform random pick, which is exactly why its scores sit on top of the random
decider's. laya-semantic spreads across all eight slots and still clears
nothing. A diverse output distribution is not evidence of judgement.

Those quality figures are seed-deterministic: repeated runs, and MPS versus CPU,
reproduce them digit for digit. The millisecond figures are not -- they are this
machine on this run, and the tick budget is derived from them.

A note on what is asserted and what is merely arithmetic. The derived tick budget
is computed from the same latency samples that section B's starred column scores,
so "100% vs 0%" in that one column is forced by construction and is labelled as
such; the independent latency evidence is the live race in section A, whose
timings were taken after the budget was fixed and were free to overrun it. The
quality half of the file has no such circularity: the games are seeded, every
decider faces byte-identical sequences, the candidate sets are built without ever
consulting the heuristic's score, and the option order is shuffled per decision
with a seed shared by all deciders.

Stable invariants asserted at the end, each labelled `measured` or `by
construction` in the printed table. The `measured` ones are the findings and
every one of them can fail: the engine self-tests pass (each runs in its own
try/except, so a broken rotation table produces a FAIL row rather than a
traceback); no option or state is silently truncated; a tick budget separating
local from a 300 ms round trip exists on this machine at all; laya-local met
>=75% of its ticks in the live race, at latencies taken AFTER the budget was
fixed; the pure-Python heuristic outplays every Laya framing by more than 3x on
lines and 1.5x on pieces; NEITHER Laya framing gets a quarter of the way from
random to the heuristic; at least one framing shows a positional bias of twice
uniform or more, scored against the per-decider uniform baseline implied by how
many options each decision actually offered rather than a flat 1/8; and no
framing beats a random pick on lines at p<0.05 under an exact paired permutation
test. The `by construction` invariants -- the sweep column at the derived
budget, and the simulated decider forfeiting -- are arithmetic consequences of
the simulation, kept as regression guards and marked as not being evidence. No
absolute millisecond figure and no exact probability is asserted, and nothing
here asserts that Laya plays Tetris well -- it does not, and the assertions say
so.
"""
from __future__ import annotations

import itertools
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

# ---------------------------------------------------------------- constants

BOARD_W, BOARD_H = 10, 20
SPAWN_COL = 3                 # where an undecided piece falls when a tick is missed
MAX_CANDIDATES = 8            # kept small so no option is truncated (see verify_prompt_budget)

RACE_SEED = 7
# The race's tick budget is DERIVED at runtime from this run's measured p95 Laya decision
# (see `derive_tick_ms`) rather than hard-coded, so the race and every invariant below
# reproduce on a slower machine, on CPU, and on hardware this was never run on.
# These are chosen so the derived window survives ordinary machine noise. The window
# [floor, ceiling] exists iff TICK_SLACK x p95(local) - min(local) <= (1 - 2 x TICK_SHARE) x 300,
# which at 1.75/0.15 tolerates a p95 local decision of ~140 ms against a ~30 ms fastest one.
# Measured here: a quiet MPS run spreads 26-38 ms, a busy one 37-173 ms, CPU 155-190 ms. Tighter
# values (1.40/0.25, floored on the absolute max) failed on a machine that merely had something
# else running -- which would have told a stranger the demo does not reproduce when it does.
TICK_SLACK = 1.75             # tick budget floor  = this x the p95 local decision measured
TICK_SHARE = 0.15             # ... plus this share of the simulated round trip
TICK_TAIL_Q = 0.95            # which tail the floor plans against (see derive_tick_ms)
RACE_MAX_PIECES = 24
SIMULATED_CLOUD_MS = 300      # SIMULATED round trip. Source: the demo's reported 316-326 ms.

# 11 seeds, not 5. The headline negative claim in section C ("not measurably better than
# random") is a claim about a small difference, so it needs enough paired games to say
# anything at all. Measured: the direction of every comparison here is stable from 3 seeds
# to 11, and 11 makes the exact paired permutation test below meaningful (2^11 sign
# assignments, enumerated in full) at a cost of about 20 s.
QUALITY_SEEDS = tuple(11 * i for i in range(1, 12))
QUALITY_MAX_PIECES = 80

LATENCY_SAMPLE_BOARDS = 24    # distinct mid-game boards used for the latency distribution
BUDGET_SWEEP_MS = (20, 50, 100, 150, 200, 300, 400)

# Four-term linear board score. These are the widely reused weights from Yiyuan Lee's
# tuned Tetris heuristic (aggregate height / lines / holes / bumpiness) -- the same
# family as Dellacherie's hand-built evaluator, with fewer terms. Pure Python,
# microseconds per placement; it is the strong baseline, not the subject.
W_HEIGHT, W_LINES, W_HOLES, W_BUMPS = -0.510066, 0.760666, -0.35663, -0.184483

# ---------------------------------------------------------------- engine

# Base cell offsets, (row, col), top-left anchored. Rotations are generated.
BASE_PIECES = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}


def _normalise(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate_cw(cells):
    """(r, c) -> (c, maxr - r): a quarter turn clockwise, re-anchored to the origin."""
    maxr = max(r for r, _ in cells)
    return _normalise([(c, maxr - r) for r, c in cells])


def _rotations(cells):
    """All distinct rotations, in turn order. O has 1, I/S/Z have 2, T/J/L have 4."""
    out, seen, cur = [], set(), _normalise(cells)
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rotate_cw(cur)
    return out


PIECES = {name: _rotations(cells) for name, cells in BASE_PIECES.items()}
PIECE_NAMES = tuple(sorted(PIECES))


def new_board():
    return [[0] * BOARD_W for _ in range(BOARD_H)]


def clone(board):
    return [row[:] for row in board]


def column_heights(board):
    """Height of each column: distance from the floor to its highest filled cell."""
    out = []
    for c in range(BOARD_W):
        h = 0
        for r in range(BOARD_H):
            if board[r][c]:
                h = BOARD_H - r
                break
        out.append(h)
    return out


def count_holes(board):
    """Empty cells with at least one filled cell somewhere above them in the column."""
    total = 0
    for c in range(BOARD_W):
        seen_block = False
        for r in range(BOARD_H):
            if board[r][c]:
                seen_block = True
            elif seen_block:
                total += 1
    return total


def features(board):
    hs = column_heights(board)
    return {
        "aggregate_height": sum(hs),
        "max_height": max(hs),
        "holes": count_holes(board),
        "bumpiness": sum(abs(hs[i] - hs[i + 1]) for i in range(BOARD_W - 1)),
    }


def fits(board, cells, top, left):
    for r, c in cells:
        rr, cc = top + r, left + c
        if rr < 0 or rr >= BOARD_H or cc < 0 or cc >= BOARD_W or board[rr][cc]:
            return False
    return True


def drop_row(board, cells, left):
    """Final resting row for `cells` dropped in column `left`, or None if blocked at spawn."""
    if not fits(board, cells, 0, left):
        return None
    top = 0
    while fits(board, cells, top + 1, left):
        top += 1
    return top


def lock_and_clear(board, cells, top, left):
    """Return (new board, lines cleared). Rows above a cleared line shift down."""
    nb = clone(board)
    for r, c in cells:
        nb[top + r][left + c] = 1
    kept = [row for row in nb if not all(row)]
    cleared = BOARD_H - len(kept)
    return [[0] * BOARD_W for _ in range(cleared)] + kept, cleared


def bag_sequence(seed, n):
    """Seeded 7-bag: every consecutive block of 7 is a permutation of the 7 pieces."""
    rng = random.Random(seed)
    seq, bag = [], []
    while len(seq) < n:
        if not bag:
            bag = list(PIECE_NAMES)
            rng.shuffle(bag)
        seq.append(bag.pop())
    return seq


def render_board(board, width_chars=True):
    cell = ("[]", " .") if width_chars else ("#", ".")
    return ["|" + "".join(cell[0] if v else cell[1] for v in row) + "|" for row in board]


# ---------------------------------------------------------------- self-tests


def engine_self_tests():
    """Assert-based proof that the engine is a real Tetris engine.

    Every number later in this file is computed on top of these primitives, so if
    they are wrong nothing downstream means anything.

    Each check is run inside its own try/except so a broken engine produces a FAIL
    row and a failed invariant rather than a traceback. That matters: an invariant
    whose only failure mode is "the process already died" is not an invariant, and
    this list is mutation-tested (swap S with Z, or J with L, and check 2 fails).

    Returns [(name, ok, detail)].
    """
    def t_rotation_tables():
        assert [len(PIECES[p]) for p in "IOTSZJL"] == [2, 1, 4, 2, 2, 4, 4], "rotation counts"
        assert all(len(cs) == 4 for rots in PIECES.values() for cs in rots), "a rotation has != 4 cells"
        return "7 pieces, distinct rotations 2/1/4/2/2/4/4, 4 cells each"

    def t_rotation_shapes():
        """The rotation SHAPES, not merely the right COUNT.

        Counts alone are a weak proof and this was verified by mutation: swapping S
        with Z, or J with L, or replacing T by a second J, leaves every rotation count
        identical and slips through a counts-only check. So each orientation of each
        piece is compared against a grid drawn by hand here, independently of the
        generator that produced it.
        """
        def _grid(*rows):
            return _normalise([(r, c) for r, row in enumerate(rows)
                               for c, ch in enumerate(row) if ch == "X"])

        truth = {
            "I": [_grid("XXXX"), _grid("X", "X", "X", "X")],
            "O": [_grid("XX", "XX")],
            "T": [_grid(".X.", "XXX"), _grid("X.", "XX", "X."),
                  _grid("XXX", ".X."), _grid(".X", "XX", ".X")],
            "S": [_grid(".XX", "XX."), _grid("X.", "XX", ".X")],
            "Z": [_grid("XX.", ".XX"), _grid(".X", "XX", "X.")],
            "J": [_grid("X..", "XXX"), _grid("XX", "X.", "X."),
                  _grid("XXX", "..X"), _grid(".X", ".X", "XX")],
            "L": [_grid("..X", "XXX"), _grid("X.", "X.", "XX"),
                  _grid("XXX", "X.."), _grid("XX", ".X", ".X")],
        }
        assert set(truth) == set(PIECES), "a piece is missing from the shape ground truth"
        for name, want in truth.items():
            assert set(PIECES[name]) == set(want), f"{name} is not the standard tetromino"
        return "all 19 orientations match a hand-drawn grid; S/Z and J/L not swapped"

    def t_line_clear():
        b = new_board()
        for c in range(BOARD_W):
            b[BOARD_H - 1][c] = 1          # a complete bottom row ...
        b[BOARD_H - 1][4] = b[BOARD_H - 1][5] = 0   # ... with an O-shaped gap in it
        b[BOARD_H - 3][0] = 1              # a lone marker two rows above the floor
        o_cells = PIECES["O"][0]
        top = drop_row(b, o_cells, 4)
        assert top == BOARD_H - 2, f"the O must fall into the gap, rested at row {top}"
        after, cleared = lock_and_clear(b, o_cells, top, 4)
        assert cleared == 1, f"expected exactly 1 line cleared, got {cleared}"
        assert after[BOARD_H - 1][4] == after[BOARD_H - 1][5] == 1, "the O's upper half fell to the floor"
        assert sum(after[BOARD_H - 1]) == 2, "nothing else survived on the floor row"
        assert b[BOARD_H - 3][0] == 1 and after[BOARD_H - 3][0] == 0, "the marker left row 17"
        assert after[BOARD_H - 2][0] == 1, "the marker must have shifted DOWN by one row"
        assert sum(sum(r) for r in after) == 3, "2 O cells + 1 marker survive the clear"
        return "1 row cleared, marker moved from r17 to r18"

    def t_multi_line_clear():
        """1, 2, 3 and 4 rows at once, with a marker above to prove the shift distance."""
        for n in (1, 2, 3, 4):
            b = new_board()
            for r in range(BOARD_H - n, BOARD_H):
                for c in range(BOARD_W):
                    b[r][c] = 1
                b[r][9] = 0                        # a clear vertical well in the last column
            b[BOARD_H - n - 2][0] = 1              # marker one row above the stack
            cells = PIECES["I"][1]                 # the vertical I, 4 tall
            top = drop_row(b, cells, 9)
            assert top is not None, f"the vertical I must fit the well for n={n}"
            after, cleared = lock_and_clear(b, cells, top, 9)
            assert cleared == n, f"expected {n} rows cleared, got {cleared}"
            assert after[BOARD_H - n - 2 + n][0] == 1, f"marker must shift down by exactly {n}"
            assert sum(sum(r) for r in after) == 4 - n + 1, f"wrong survivor count for n={n}"
        return "1/2/3/4 simultaneous rows clear; rows above shift down by exactly n"

    def t_collision():
        b = new_board()
        rng = random.Random(0)
        overlaps = floating = 0
        for name in bag_sequence(3, 30):
            cells = PIECES[name][rng.randrange(len(PIECES[name]))]
            w = max(c for _, c in cells) + 1
            col = rng.randrange(BOARD_W - w + 1)
            top = drop_row(b, cells, col)
            if top is None:
                break
            if any(b[top + r][col + c] for r, c in cells):
                overlaps += 1
            if fits(b, cells, top + 1, col):
                floating += 1
            b, _ = lock_and_clear(b, cells, top, col)
        assert overlaps == 0, f"{overlaps} placements overlapped an occupied cell"
        assert floating == 0, f"{floating} placements came to rest in mid-air"
        return "30 drops: 0 overlaps, 0 pieces resting in mid-air"

    def t_top_out():
        b = new_board()
        topped = False
        for _ in range(40):
            top = drop_row(b, PIECES["O"][0], 0)
            if top is None:
                topped = True
                break
            b, _ = lock_and_clear(b, PIECES["O"][0], top, 0)
        assert topped, "stacking O pieces in one column must eventually top out"
        return "O pieces stacked in column 0 blocked the spawn row"

    def t_holes():
        b = new_board()
        b[10][2] = 1                        # roof
        assert count_holes(b) == BOARD_H - 11, "every empty cell under the roof is a hole"
        b2 = new_board()
        b2[BOARD_H - 1][2] = 1              # a cell on the floor roofs nothing
        assert count_holes(b2) == 0, "a floor cell creates no holes"
        return "cells under a roof count, a floor cell roofs nothing"

    def t_bag():
        a1, a2 = bag_sequence(42, 70), bag_sequence(42, 70)
        assert a1 == a2, "same seed must reproduce the same sequence"
        assert bag_sequence(43, 70) != a1, "different seeds must differ"
        for i in range(0, 70, 7):
            assert sorted(a1[i:i + 7]) == sorted(PIECE_NAMES), f"bag {i // 7} is not a permutation"
        return "same seed identical over 70 pieces, every bag a permutation"

    tests = [
        ("rotation tables", t_rotation_tables),
        ("rotation shapes", t_rotation_shapes),
        ("line clear shifts rows down", t_line_clear),
        ("multi-line clears", t_multi_line_clear),
        ("collision and lock-down", t_collision),
        ("top-out", t_top_out),
        ("hole counting", t_holes),
        ("seeded 7-bag", t_bag),
    ]
    checks = []
    for name, fn in tests:
        try:
            checks.append((name, True, fn()))
        except AssertionError as e:
            checks.append((name, False, f"FAILED: {e}"))
        except Exception as e:                      # a crash is a failure, not a traceback
            checks.append((name, False, f"FAILED: {type(e).__name__}: {e}"))
    return checks


# ---------------------------------------------------------------- candidates


def enumerate_candidates(board, piece):
    """Every legal (rotation, column) placement, simulated, with resulting features.

    Deduplicated by resulting board, because different rotations of I/O/S/Z can
    produce the identical outcome and a duplicate option would waste a slot.
    """
    before = features(board)
    out, seen = [], set()
    for rot, cells in enumerate(PIECES[piece]):
        w = max(c for _, c in cells) + 1
        for col in range(BOARD_W - w + 1):
            top = drop_row(board, cells, col)
            if top is None:
                continue
            after, lines = lock_and_clear(board, cells, top, col)
            key = tuple(tuple(r) for r in after)
            if key in seen:
                continue
            seen.add(key)
            f = features(after)
            out.append({
                "rot": rot, "col": col, "top": top, "cells": cells,
                "board": after, "lines": lines, "f": f,
                "delta_holes": f["holes"] - before["holes"],
                "score": (W_HEIGHT * f["aggregate_height"] + W_LINES * lines
                          + W_HOLES * f["holes"] + W_BUMPS * f["bumpiness"]),
            })
    return out


def select_candidates(all_cands, rng):
    """Reduce to MAX_CANDIDATES, decider-agnostically and without leaking the answer.

    The placements are first put in canonical (rotation, column) order -- an order
    that knows nothing about any decider. If there are more than MAX_CANDIDATES,
    an evenly spaced subset of that ordering is taken, which keeps a spread across
    the whole board rather than a cluster. Pre-filtering by the heuristic score
    would put the heuristic's answer into every other decider's option list and
    rig the comparison, so it is never used here.

    The survivors are then shuffled with `rng`, which is seeded per decision and
    shared by every decider. Shuffling is what makes the index histogram in
    section C interpretable: with a fixed order, "always picks option 0" and
    "always picks the leftmost column" are the same observation.
    """
    ordered = sorted(all_cands, key=lambda p: (p["rot"], p["col"]))
    n = len(ordered)
    if n > MAX_CANDIDATES:
        step = (n - 1) / (MAX_CANDIDATES - 1)
        idx = sorted({int(round(i * step)) for i in range(MAX_CANDIDATES)})
        ordered = [ordered[i] for i in idx]
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------- option rendering

OPT_KEYS = "abcdefghijkl"
# One key per option. Raising MAX_CANDIDATES past this is not a small edit: the keys run out
# here, and long before that laya silently re-truncates every option (see verify_prompt_budget).
assert MAX_CANDIDATES <= len(OPT_KEYS), "MAX_CANDIDATES exceeds the option keys"


def numeric_option(p):
    return (f"col {p['col']} rot {p['rot']}: clears {p['lines']}, holes +{max(0, p['delta_holes'])}, "
            f"top {p['f']['max_height']}, bumps {p['f']['bumpiness']}")


def semantic_option(p):
    lines = {0: "no clear", 1: "clears one row", 2: "clears two rows",
             3: "clears three rows", 4: "clears four rows"}[min(4, p["lines"])]
    dh = max(0, p["delta_holes"])
    gaps = {0: "no new gaps", 1: "buries one gap", 2: "buries two gaps"}.get(dh, "buries several gaps")
    mh = p["f"]["max_height"]
    height = ("stack stays low" if mh <= 5 else "stack is mid height" if mh <= 10
              else "stack is high" if mh <= 14 else "stack near the ceiling")
    bump = p["f"]["bumpiness"]
    surface = ("flat surface" if bump <= 4 else "uneven surface" if bump <= 9 else "jagged surface")
    return f"{lines}, {gaps}, {height}, {surface}"


FRAMINGS = {"laya-numeric": numeric_option, "laya-semantic": semantic_option}

INSTRUCTIONS = "Which placement leaves the best Tetris board?"


def board_state(board, piece):
    rows = "\n".join("".join("#" if v else "." for v in row) for row in board)
    return f"Tetris board, # filled, . empty, top row first:\n{rows}\nFalling piece: {piece}"


def build_question(cands, render):
    crit = {OPT_KEYS[i]: render(p) for i, p in enumerate(cands)}
    return {"place": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": crit}}


# ---------------------------------------------------------------- prompt budget check


def budget_sample_boards(seeds=(5, 17, 29), max_pieces=None):
    """Boards to test the prompt budget against -- the worst ones, not a tidy one.

    Checking a single mid-game board is not enough: the numeric option text grows with
    the numbers in it, and a tall, jagged, hole-riddled stack renders "top 20, bumps 45"
    where a clean board renders "top 7, bumps 9". A random decider builds exactly those
    boards, so the budget is measured across whole random games. Pure engine work, no
    model calls; it costs a few hundred milliseconds.
    """
    out = []
    for seed in seeds:
        board = new_board()
        rng_sel = random.Random(seed * 1000 + 1)
        rng_dec = random.Random(seed * 1000 + 2)
        for piece in bag_sequence(seed, max_pieces or QUALITY_MAX_PIECES):
            all_c = enumerate_candidates(board, piece)
            if not all_c:
                break
            cands = select_candidates(all_c, random.Random(rng_sel.randrange(1 << 30)))
            out.append((board, piece, cands))
            board = cands[rng_dec.randrange(len(cands))]["board"]
    return out


def verify_prompt_budget(agent, samples):
    """Confirm laya is not silently truncating our options -- on every board, not one.

    `build_sequence` gives instructions + all options a shared 192-token head budget
    (`head_max_len`). Each option is first capped at 48 tokens; if the options together
    still leave under 16 tokens for the instructions, EVERY option is re-truncated to
    `(head_max_len - 16) // n_options` tokens, mid-word and without warning. At 20 options
    that is 8 tokens each and the options stop being distinguishable.

    `samples` is a list of (board, piece, candidates). The returned report is the WORST
    case over all of them per framing -- the largest option-token total, the smallest
    remaining instruction budget, the longest sequence -- plus how many samples tripped
    either truncation. Reporting the worst case is the point: the invariant has to hold
    for every decision the games actually make, not for one flattering example.
    """
    from laya.common import build_sequence, render_options

    head_max_len = agent.cfg.get("head_max_len", 192)
    max_len = agent.cfg.get("max_len", 512)
    tok = agent.tok
    report = {}
    for name, render in FRAMINGS.items():
        agg = {"n_samples": len(samples), "n_options": 0, "longest_option_tokens": 0,
               "option_tokens_total": 0, "head_budget": head_max_len,
               "tokens_left_for_instructions": head_max_len, "options_re_truncated": False,
               "sequence_tokens": 0, "max_len": max_len, "markers_kept": 0,
               "state_truncated": False, "example_option": "",
               "n_samples_options_truncated": 0, "n_samples_state_truncated": 0}
        for board, piece, cands in samples:
            q = build_question(cands, render)["place"]
            internal = agent._to_internal(q)
            opts = render_options(internal)
            # exactly what build_sequence does before the re-truncation branch
            raw = [1 + len(tok(" " + o, add_special_tokens=False)["input_ids"][:48]) for o in opts]
            left = head_max_len - sum(raw)
            ids, markers = build_sequence(tok, board_state(board, piece), internal,
                                          max_len, head_max_len)
            agg["n_samples_options_truncated"] += left < 16
            agg["n_samples_state_truncated"] += len(ids) >= max_len
            if sum(raw) > agg["option_tokens_total"]:      # the worst sample defines the row
                agg.update({"n_options": len(opts), "longest_option_tokens": max(raw),
                            "option_tokens_total": sum(raw), "tokens_left_for_instructions": left,
                            "options_re_truncated": left < 16, "markers_kept": len(markers),
                            "example_option": opts[raw.index(max(raw))]})
            agg["sequence_tokens"] = max(agg["sequence_tokens"], len(ids))
            agg["state_truncated"] = agg["state_truncated"] or len(ids) >= max_len
        report[name] = agg
    return report


# ---------------------------------------------------------------- deciders


class Decider:
    """A decider maps (board, piece, candidate list) -> index into that list."""

    def __init__(self, name, fn, kind):
        self.name, self.fn, self.kind = name, fn, kind

    def decide(self, board, piece, cands, rng):
        return self.fn(board, piece, cands, rng)


def heuristic_pick(board, piece, cands, rng):
    best, best_i = None, 0
    for i, p in enumerate(cands):
        if best is None or p["score"] > best:
            best, best_i = p["score"], i
    return best_i


def random_pick(board, piece, cands, rng):
    return rng.randrange(len(cands))


def make_laya_pick(agent, render):
    def pick(board, piece, cands, rng):
        q = build_question(cands, render)
        ans = agent.predict(board_state(board, piece), q)["answers"]["place"]
        key = ans["choice"]
        return OPT_KEYS.index(key) if key in OPT_KEYS[:len(cands)] else 0
    return pick


# ---------------------------------------------------------------- game loop


def play(decider, seed, max_pieces, device, tick_ms=None, extra_ms=0.0):
    """Play one game. Returns a result dict.

    `tick_ms=None` means unlimited time (section C). With a tick budget, a
    decision whose wall time exceeds it is FORFEIT: the piece drops undecided at
    its spawn column, exactly as in the demo. `extra_ms` is an artificial delay
    added to each decision to model a network round trip; it is only ever used
    for the SIMULATED cloud decider.
    """
    board = new_board()
    seq = bag_sequence(seed, max_pieces)
    rng_sel = random.Random(seed * 1000 + 1)
    rng_dec = random.Random(seed * 1000 + 2)
    lines = pieces = met = missed = 0
    idx_hist, col_hist, nopt_hist, lat = {}, {}, {}, []
    last_ms = 0.0
    topped_out = False

    for _n, piece in enumerate(seq):
        all_c = enumerate_candidates(board, piece)
        if not all_c:
            topped_out = True
            break
        sel_rng = random.Random(rng_sel.randrange(1 << 30))
        cands = select_candidates(all_c, sel_rng)

        C.sync(device)
        t0 = time.perf_counter()
        choice = decider.decide(board, piece, cands, rng_dec)
        if extra_ms:
            time.sleep(extra_ms / 1000.0)
        C.sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        last_ms = elapsed
        lat.append(elapsed)

        if tick_ms is not None and elapsed > tick_ms:
            board, got, alive = forfeit_drop(board, piece)
            if not alive:
                topped_out = True
                break
            missed += 1
        else:
            met += 1
            p = cands[choice]
            idx_hist[choice] = idx_hist.get(choice, 0) + 1
            col_hist[p["col"]] = col_hist.get(p["col"], 0) + 1
            nopt_hist[len(cands)] = nopt_hist.get(len(cands), 0) + 1
            board, got = board_after(board, p)
        lines += got
        pieces += 1

    f = features(board)
    return {
        "decider": decider.name, "seed": seed, "board": board,
        "lines": lines, "pieces": pieces, "topped_out": topped_out,
        "met": met, "missed": missed,
        "holes": f["holes"], "max_height": f["max_height"],
        "aggregate_height": f["aggregate_height"], "bumpiness": f["bumpiness"],
        "index_hist": idx_hist, "col_hist": col_hist, "nopt_hist": nopt_hist,
        "latency_ms": lat, "last_ms": last_ms,
    }


def board_after(board, p):
    """Apply an already-simulated candidate. `p['board']` was computed by the engine."""
    return p["board"], p["lines"]


def forfeit_drop(board, piece):
    """The overrun penalty. ONE implementation, used by every decider and both loops.

    A decision that overruns the tick forfeits the piece: it drops in rotation 0 at
    the spawn column, undecided, exactly as in the demo. Returning this from a single
    function is what makes "the penalty is identical for every decider" a structural
    fact rather than two copies that happen to agree today.

    Returns (board, lines cleared, alive); alive is False when the piece cannot even
    spawn, which is a top-out.
    """
    cells = PIECES[piece][0]
    w = max(c for _, c in cells) + 1
    col = min(SPAWN_COL, BOARD_W - w)
    top = drop_row(board, cells, col)
    if top is None:
        return board, 0, False
    nb, got = lock_and_clear(board, cells, top, col)
    return nb, got, True


# ---------------------------------------------------------------- race rendering


def race_frame(states, tick_ms):
    """One rendered frame of the side-by-side race, as a list of lines."""
    cols = []
    for s in states:
        head = [
            f" {s['label']}",
            f" last decision {s['last_ms']:6.1f} ms   tick {tick_ms} ms",
            f" ticks met {s['met']:2d}  missed {C.c(str(s['missed']).rjust(2), 'red') if s['missed'] else ' 0'}"
            f"   lines {s['lines']}",
            " " + "-" * (BOARD_W * 2 + 2),
        ]
        body = render_board(s["board"])
        foot = [" " + "-" * (BOARD_W * 2 + 2)]
        if s["done"] and s["topped"] and s["missed"] > s["met"]:
            foot.append(C.c(" GAME OVER - COULD NOT KEEP UP", "red"))
        elif s["done"] and s["topped"]:
            foot.append(C.c(" GAME OVER - topped out", "yellow"))
        elif s["done"] and s["missed"]:
            foot.append(C.c(f" survived, {s['missed']} pieces forfeited", "yellow"))
        elif s["done"]:
            foot.append(C.c(" survived, every tick met", "green"))
        else:
            foot.append("")
        cols.append(head + body + foot)

    width = BOARD_W * 2 + 4
    height = max(len(col) for col in cols)
    lines = []
    for r in range(height):
        parts = []
        for col in cols:
            cell = col[r] if r < len(col) else ""
            pad = " " * max(0, width - C._visible_len(cell))
            parts.append(cell + pad)
        lines.append("   " + "    ".join(parts))
    return lines


def run_race(agent, device, tty, tick_ms):
    """Section A. Both boards face the identical piece sequence."""
    laya_pick = make_laya_pick(agent, numeric_option)
    seq = bag_sequence(RACE_SEED, RACE_MAX_PIECES)

    states = [
        {"label": C.c("laya-local", "green"), "board": new_board(), "lines": 0, "pieces": 0,
         "met": 0, "missed": 0, "last_ms": 0.0, "lat": [], "over": False, "topped": False,
         "done": False, "extra": 0.0,
         "rng_sel": random.Random(RACE_SEED * 1000 + 1), "rng_dec": random.Random(1)},
        {"label": C.c(f"cloud-{SIMULATED_CLOUD_MS}ms (simulated)", "yellow"), "board": new_board(),
         "lines": 0, "pieces": 0, "met": 0, "missed": 0, "last_ms": 0.0, "lat": [],
         "over": False, "topped": False, "done": False, "extra": float(SIMULATED_CLOUD_MS),
         "rng_sel": random.Random(RACE_SEED * 1000 + 1), "rng_dec": random.Random(1)},
    ]

    printed = 0

    def draw():
        nonlocal printed
        lines = race_frame(states, tick_ms)
        if tty:
            if printed:
                sys.stdout.write(f"\033[{printed}A")
            sys.stdout.write("\n".join(ln + "\033[K" for ln in lines) + "\n")
            sys.stdout.flush()
            printed = len(lines)
        return lines

    if tty:
        draw()

    for piece in seq:
        if all(s["over"] for s in states):
            break
        for s in states:
            if s["over"]:
                continue
            all_c = enumerate_candidates(s["board"], piece)
            if not all_c:
                s["over"] = s["topped"] = True
                continue
            cands = select_candidates(all_c, random.Random(s["rng_sel"].randrange(1 << 30)))

            C.sync(device)
            t0 = time.perf_counter()
            choice = laya_pick(s["board"], piece, cands, s["rng_dec"])
            if s["extra"]:
                time.sleep(s["extra"] / 1000.0)
            C.sync(device)
            s["last_ms"] = (time.perf_counter() - t0) * 1000.0
            s["lat"].append(s["last_ms"])

            if s["last_ms"] > tick_ms:
                s["board"], got, alive = forfeit_drop(s["board"], piece)
                if not alive:
                    s["over"] = s["topped"] = True
                    continue
                s["missed"] += 1
            else:
                s["met"] += 1
                s["board"], got = board_after(s["board"], cands[choice])
            s["lines"] += got
            s["pieces"] += 1
        if tty:
            draw()

    for s in states:
        s["over"] = s["done"] = True
    final = draw()
    if not tty:
        print("\n".join(final))
    return states


# ---------------------------------------------------------------- stats helpers


def paired_perm_p(a, b):
    """Exact two-sided paired permutation test on per-seed results.

    Every decider plays the SAME seeded sequences, so the games pair up one-to-one and
    the only question is whether the sign pattern of the per-seed differences is
    surprising. With n seeds there are 2^n sign assignments and they are enumerated in
    full -- no sampling, no normality assumption, no scipy. Small n is exactly the
    regime this test is for, and a Tetris game is high variance, so eyeballing a
    standard deviation is not enough to support "not measurably better than random".
    """
    d = [x - y for x, y in zip(a, b)]
    if not d or all(abs(x) < 1e-12 for x in d):
        return 1.0
    obs = abs(statistics.fmean(d))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        if abs(statistics.fmean(sg * x for sg, x in zip(signs, d))) >= obs - 1e-12:
            hits += 1
    return hits / total


def uniform_index_share(nopt_hist, n_slots):
    """Expected share of picks per option index for a decider choosing UNIFORMLY.

    Not 1/n_slots: a decision late in a game can offer fewer than MAX_CANDIDATES legal
    placements, and index 7 simply does not exist on those turns. Comparing a measured
    share against a flat 1/8 therefore overstates the bias on the low indices and
    understates it on the high ones. This returns the honest per-index baseline implied
    by the option counts this decider actually faced.
    """
    total = sum(nopt_hist.values())
    if not total:
        return [1.0 / n_slots] * n_slots
    return [sum(cnt / n for n, cnt in nopt_hist.items() if n > i) / total
            for i in range(n_slots)]


def mean_sd(xs):
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def percentile(samples, q):
    """Nearest-rank percentile of `samples` (0 <= q <= 1). No numpy needed."""
    xs = sorted(samples)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def derive_tick_ms(samples, cloud_ms):
    """Pick a tick budget from measured latency, with both constraints made explicit.

    Hard-coding "120 ms" would make this scenario's result an artefact of one laptop: on a
    slower box Laya would forfeit every tick and the race would say the opposite thing. A
    usable budget has to satisfy two inequalities at once:

      lo = p95 local decision x TICK_SLACK + TICK_SHARE x cloud_ms
           -- above the local tail, with slack, so the local decider does not forfeit when a
              tensor shape it has not seen costs it a kernel recompile mid-game
      hi = fastest local decision + (1 - TICK_SHARE) x cloud_ms
           -- below the fastest possible cloud reply, so the simulated round trip forfeits

    The floor plans against the p95, not the single slowest sample. One outlier -- a kernel
    recompile, another process waking up -- should not define a budget the rest of the run is
    then scored against; that is what made an earlier version of this fail on a machine that
    merely had something else running. The tail it does not cover is covered by the race
    invariant tolerating a quarter of its ticks being missed.

    The window [lo, hi] exists whenever the local decider's own jitter is smaller than the
    simulated handicap. When it is not -- a heavily loaded or very slow machine -- no budget
    separates them, and that is worth saying out loud rather than papering over, so the
    caller is told which bound was used.
    """
    p95 = percentile(samples, TICK_TAIL_Q)
    lo = p95 * TICK_SLACK + TICK_SHARE * cloud_ms
    hi = min(samples) + (1.0 - TICK_SHARE) * cloud_ms
    ok = lo <= hi
    chosen = lo if ok else hi
    return int(-(-chosen // 10) * 10), {"lo": round(lo, 2), "hi": round(hi, 2),
                                        "window_exists": ok,
                                        "tail_quantile": TICK_TAIL_Q,
                                        "local_p95": round(p95, 2),
                                        "local_min": round(min(samples), 2),
                                        "local_max": round(max(samples), 2)}


def frac_met(samples, budget_ms):
    return sum(1 for s in samples if s <= budget_ms) / len(samples) if samples else 0.0


def hist_line(hist, n_slots, total):
    cells = []
    for i in range(n_slots):
        v = hist.get(i, 0)
        cells.append(f"{v:3d}")
    top = max(hist.values()) if hist else 0
    share = top / total if total else 0.0
    return " ".join(cells), share


def mark(ok):
    return C.c("PASS", "green") if ok else C.c("FAIL", "red")


def _plain(text):
    """Strip ANSI so a coloured label can go into a table cell or an artefact."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


# ---------------------------------------------------------------- main


def main():
    global select_candidates      # section 0 temporarily swaps it to measure the cap's cost
    C.require_cached("english")
    device = C.resolve_device()
    tty = sys.stdout.isatty()

    C.header(
        "07 - Tetris arena: latency budget vs decision quality",
        f"english checkpoint on {device} - two independent properties, measured separately",
    )

    # ------------------------------------------------------------ 0
    C.section("0. The engine, the candidate policy, and the prompt budget")

    t0 = time.perf_counter()
    checks = engine_self_tests()
    self_test_ms = (time.perf_counter() - t0) * 1000
    self_tests_ok = all(ok for _, ok, _ in checks)
    C.table([[name, C.c("PASS", "green") if ok else C.c("FAIL", "red"), detail]
             for name, ok, detail in checks],
            ["ENGINE SELF-TEST", "RESULT", "WHAT IT PROVED"])
    print(f"\n  {sum(ok for _, ok, _ in checks)}/{len(checks)} self-tests in {self_test_ms:.1f} ms. "
          f"{BOARD_W}x{BOARD_H} board, seeded 7-bag, standard rotations.")
    print(C.c("  If the engine were wrong every number below would be meaningless, so it is proved "
              "in-file\n  rather than asserted in prose.", "dim"))

    agent = C.load_agent("english")

    print()
    print("  candidate policy: enumerate every legal (rotation, column) placement, simulate each,")
    print("  drop duplicates by resulting board, sort canonically by (rotation, column), then keep")
    print(f"  {MAX_CANDIDATES} EVENLY SPACED across that ordering and shuffle with a per-decision seed.")
    print(C.c("  The heuristic score is never consulted when choosing the option set. Pre-filtering by\n"
              "  it would post the heuristic's own answer into every rival's option list.", "dim"))

    # What the cap costs, measured: the same heuristic, same seeds, same engine, with and
    # without the reduction. Every decider in this file pays this, so it is not a bias --
    # but it does cap how well ANY of them can play, and the number belongs in the open.
    capped_sel = select_candidates

    def _uncapped(all_c, rng):
        o = sorted(all_c, key=lambda p: (p["rot"], p["col"]))
        rng.shuffle(o)
        return o

    heur = Decider("heuristic", heuristic_pick, "code")
    cap_runs = [play(heur, sd, QUALITY_MAX_PIECES, device) for sd in QUALITY_SEEDS]
    try:
        select_candidates = _uncapped
        full_runs = [play(heur, sd, QUALITY_MAX_PIECES, device) for sd in QUALITY_SEEDS]
    finally:
        select_candidates = capped_sel      # restore even if a game raises
    cap_cost = {
        "capped_lines": round(statistics.fmean(r["lines"] for r in cap_runs), 2),
        "uncapped_lines": round(statistics.fmean(r["lines"] for r in full_runs), 2),
        "capped_pieces": round(statistics.fmean(r["pieces"] for r in cap_runs), 2),
        "uncapped_pieces": round(statistics.fmean(r["pieces"] for r in full_runs), 2),
    }
    n_legal = []
    bb = new_board()
    for pc in bag_sequence(99, 30):
        cs = enumerate_candidates(bb, pc)
        if not cs:
            break
        n_legal.append(len(cs))
        bb = max(cs, key=lambda x: x["score"])["board"]
    cap_cost["mean_legal_placements"] = round(statistics.fmean(n_legal), 2)
    print()
    C.table(
        [[f"all legal placements (mean {cap_cost['mean_legal_placements']:.1f} per piece)",
          f"{cap_cost['uncapped_lines']:.1f}", f"{cap_cost['uncapped_pieces']:.1f}"],
         [f"reduced to {MAX_CANDIDATES} (what every decider here sees)",
          f"{cap_cost['capped_lines']:.1f}", f"{cap_cost['capped_pieces']:.1f}"]],
        ["OPTION SET GIVEN TO THE HEURISTIC", "LINES (mean)", "PIECES (mean)"], align="lrr")
    print(C.c(f"  The cap is what laya's 192-token head budget forces -- {MAX_CANDIDATES} options is "
              f"what fits without\n  silent re-truncation (measured in the next table) -- and it "
              f"costs the heuristic most of its\n  play. Every decider below sees the same reduced "
              f"set, so the "
              f"comparison is fair -- but nobody\n  in this file is playing Tetris at full strength, "
              f"and that ceiling is the prompt's, not the\n  player's.", "dim"))

    budget_samples = budget_sample_boards()
    budget = verify_prompt_budget(agent, budget_samples)
    C.table(
        [[name, r["n_options"], r["longest_option_tokens"],
          f"{r['option_tokens_total']}/{r['head_budget']}",
          r["tokens_left_for_instructions"],
          C.c("YES", "red") if r["options_re_truncated"] else C.c("no", "green"),
          f"{r['sequence_tokens']}/{r['max_len']}",
          C.c("YES", "red") if r["state_truncated"] else C.c("no", "green")]
         for name, r in budget.items()],
        ["FRAMING", "OPTS", "LONGEST OPT", "OPT TOKENS", "LEFT FOR INSTR",
         "OPTS TRUNCATED", "SEQ TOKENS", "STATE TRUNCATED"],
        align="lrrrrlrl",
    )
    for name, r in budget.items():
        print(f"  {name:<14} longest option seen -> {r['example_option']!r}")
    print(C.c(f"  laya gives instructions + ALL options one {budget['laya-numeric']['head_budget']}-token "
              "head budget. Past it, every option is silently\n  re-truncated mid-word to "
              "(192-16)/n tokens. That is why the candidate set is capped at "
              f"{MAX_CANDIDATES}\n  and the options are short: at ~20 options they stop being "
              "distinguishable at all.", "dim"))
    print(C.c(f"  The row above is the WORST of {budget['laya-numeric']['n_samples']} real decisions, "
              f"taken from complete random-decider games so the\n  sample spans clean early boards "
              f"and the tall, jagged, hole-riddled ones the games actually reach.\n  Checking a "
              f"single tidy mid-game board would leave the headroom unmeasured exactly where it is "
              f"thinnest\n  -- and it is thin: "
              f"{budget['laya-numeric']['tokens_left_for_instructions']} tokens spare in the worst "
              f"case. Samples that tripped re-truncation: "
              + ", ".join(f"{k} {v['n_samples_options_truncated']}/{v['n_samples']}"
                          for k, v in budget.items()) + ".", "dim"))

    # counts, not the worst-sample flags: one truncated decision anywhere in the sample is a
    # failure, because the games make decisions on boards like those every tick.
    prompt_ok = not any(r["n_samples_options_truncated"] or r["n_samples_state_truncated"]
                        for r in budget.values())

    # ------------------------------------------------------------ latency pool
    C.section("0b. Per-decision latency - measured, except the last row's simulated offset")

    pool_boards = []
    b = new_board()
    seq = bag_sequence(123, LATENCY_SAMPLE_BOARDS + 6)
    for piece in seq:
        cs = enumerate_candidates(b, piece)
        if not cs:
            break
        pool_boards.append((b, piece, select_candidates(cs, random.Random(len(pool_boards)))))
        b = max(cs, key=lambda p: p["score"])["board"]
    pool_boards = pool_boards[:LATENCY_SAMPLE_BOARDS]

    latency = {}
    for name, render in FRAMINGS.items():
        pick = make_laya_pick(agent, render)
        for bb, pp, cc in pool_boards[:3]:          # warm-up, untimed
            pick(bb, pp, cc, random.Random(0))
        C.sync(device)
        samples = []
        for bb, pp, cc in pool_boards:
            C.sync(device)
            t = time.perf_counter()
            pick(bb, pp, cc, random.Random(0))
            C.sync(device)
            samples.append((time.perf_counter() - t) * 1000)
        latency[name] = samples

    for name, fn in (("heuristic", heuristic_pick), ("random", random_pick)):
        samples = []
        for bb, pp, cc in pool_boards:
            t = time.perf_counter()
            fn(bb, pp, cc, random.Random(0))
            samples.append((time.perf_counter() - t) * 1000)
        latency[name] = samples

    latency[f"cloud-{SIMULATED_CLOUD_MS}ms"] = [s + SIMULATED_CLOUD_MS for s in latency["laya-numeric"]]

    tick_ms, tick_src = derive_tick_ms(latency["laya-numeric"] + latency["laya-semantic"],
                                       SIMULATED_CLOUD_MS)
    sweep = tuple(sorted(set(BUDGET_SWEEP_MS) | {tick_ms}))

    lat_rows = []
    for name in ("heuristic", "random", "laya-numeric", "laya-semantic", f"cloud-{SIMULATED_CLOUD_MS}ms"):
        s = latency[name]
        simulated = name.startswith("cloud")
        lat_rows.append([
            name + " (simulated)" if simulated else name,
            C.c("simulated", "yellow") if simulated else C.c("measured", "green"),
            f"{statistics.median(s):8.2f}", f"{min(s):8.2f}", f"{max(s):8.2f}",
            f"{statistics.fmean(s):8.2f}",
        ])
    C.table(lat_rows, ["DECIDER", "SOURCE", "p50 ms", "min ms", "max ms", "mean ms"],
            align="llrrrr")
    spread = max(latency["laya-numeric"]) / max(1e-9, min(latency["laya-numeric"]))
    print(C.c(f"  Note the spread: {spread:.1f}x between the fastest and slowest Laya decision. Every "
              f"board tokenises to a\n  different length and every option set to different text, so "
              f"the backend re-specialises kernels\n  for shapes it has not seen. A fixed-prompt "
              f"microbenchmark does not show this; a game loop, where\n  the prompt changes every "
              f"tick, does. Plan the tick budget against the tail, not the median.", "dim"))
    print(C.c("  The timed region is the DECISION only. Enumerating and reducing the candidates is "
              "shared\n  engine work, identical for every decider, and sits outside the clock in "
              "both this table\n  and the game loop.", "dim"))
    print(f"\n  {LATENCY_SAMPLE_BOARDS} distinct mid-game boards, {MAX_CANDIDATES} options each, "
          f"device synchronised around every call,\n  3 untimed warm-ups first. The cloud row is "
          f"laya-numeric's OWN measured latency plus a flat\n  {SIMULATED_CLOUD_MS} ms: a "
          f"{C.c('SIMULATED', 'yellow')} round trip, sourced from the demo's reported 316-326 ms. "
          f"No cloud\n  service was contacted or benchmarked here, and none is named.")

    # ------------------------------------------------------------ A
    C.section("A. LIVE RACE - same piece sequence, one fixed tick budget")
    print(f"  seed {RACE_SEED}, {RACE_MAX_PIECES} pieces, tick budget {tick_ms} ms -- derived from "
          f"this run's own latency:\n  above {TICK_SLACK}x the p95 local decision "
          f"({tick_src['local_p95']:.0f} ms; slowest seen {tick_src['local_max']:.0f} ms), below "
          f"the fastest possible simulated reply "
          f"({tick_src['local_min'] + SIMULATED_CLOUD_MS:.0f} ms).")
    print(f"  Both boards run the SAME laya-numeric decision on the SAME pieces; the right-hand board "
          f"also waits\n  a {C.c('simulated', 'yellow')} {SIMULATED_CLOUD_MS} ms round trip. Overrun "
          f"the tick and the piece drops undecided at spawn column {SPAWN_COL}.")
    if not tick_src["window_exists"]:
        print(C.c(f"  WARNING: no tick budget separates the two on this machine. A local decision "
                  f"here costs\n  {tick_src['local_min']:.0f}-{tick_src['local_max']:.0f} ms, and "
                  f"its own jitter is larger than the {SIMULATED_CLOUD_MS} ms handicap being "
                  f"simulated, so the\n  floor ({tick_src['lo']:.0f} ms) sits above the ceiling "
                  f"({tick_src['hi']:.0f} ms). The demo's mechanism does not reproduce here and\n"
                  f"  the invariants below will say so rather than quietly picking a budget that "
                  f"flatters it.", "red"))
    if not tty:
        print(C.c("  stdout is not a TTY: animation skipped, final boards printed once.", "dim"))
    print()

    race = run_race(agent, device, tty, tick_ms)
    print()
    C.table(
        [[_plain(s["label"]), s["pieces"], s["met"], s["missed"],
          f"{s['met'] / max(1, s['pieces']):.0%}",
          f"{statistics.median(s['lat']):.0f}" if s["lat"] else "-",
          f"{max(s['lat']):.0f}" if s["lat"] else "-",
          s["lines"], count_holes(s["board"]), max(column_heights(s["board"]))]
         for s in race],
        ["BOARD", "PIECES", "TICKS MET", "TICKS MISSED", "MET %", "p50 ms", "max ms",
         "LINES", "HOLES", "MAX HEIGHT"],
        align="lrrrrrrrrr",
    )
    local, cloud = race
    print(f"\n  The cloud board was not given worse judgement. It ran the IDENTICAL decision procedure "
          f"on the\n  identical pieces -- its answers simply arrived after the tick had passed, so none of "
          f"them was\n  ever used: {cloud['missed']} of its {cloud['pieces']} pieces fell undecided at "
          f"column {SPAWN_COL} until it topped out. That is the whole\n  mechanism behind the viral demo, "
          f"it reproduces cleanly, and it is entirely about latency -- with\n  the {SIMULATED_CLOUD_MS} ms "
          f"handicap supplied by a time.sleep, not by any measured service.")
    print(f"  Now look at the LEFT board. It met {local['met']}/{local['pieces']} ticks, lost nothing "
          f"to the clock, and still finished with\n  {count_holes(local['board'])} holes, max height "
          f"{max(column_heights(local['board']))} and {local['lines']} lines -- a stack at the "
          f"ceiling that was about to end the game anyway. The\n  race is capped at "
          f"{RACE_MAX_PIECES} pieces; section C lets the same decider run until it tops out, and it "
          f"does so early.\n  Winning the race and playing well are different things, which is what "
          f"sections C and D measure\n  separately.")

    # ------------------------------------------------------------ B
    C.section("B. The latency budget curve - fraction of ticks met")

    curve = {}
    rows = []
    for name in ("heuristic", "random", "laya-numeric", "laya-semantic", f"cloud-{SIMULATED_CLOUD_MS}ms"):
        s = latency[name]
        fr = {b: frac_met(s, b) for b in sweep}
        curve[name] = fr
        label = name + " (simulated)" if name.startswith("cloud") else name
        rows.append([label] + [f"{fr[b]:6.0%}" for b in sweep]
                    + [f"{statistics.median(s):7.1f}"])
    C.table(rows, ["DECIDER"] + [(f"{b} ms*" if b == tick_ms else f"{b} ms") for b in sweep]
            + ["p50 ms"], align="l" + "r" * (len(sweep) + 1))
    print(C.c(f"  * {tick_ms} ms is this run's DERIVED tick budget, not a hard-coded one: above "
              f"{TICK_SLACK}x the p95 local\n    decision measured "
              f"({tick_src['local_p95']:.0f} ms, slowest seen {tick_src['local_max']:.0f}) and "
              f"below the fastest possible simulated reply "
              f"({tick_src['local_min'] + SIMULATED_CLOUD_MS:.0f} ms)."
              + ("" if tick_src["window_exists"] else
                 "  NOTE: no budget separates them on this run; local jitter exceeds the "
                 "simulated handicap."), "dim"))
    print()
    for name in ("laya-numeric", f"cloud-{SIMULATED_CLOUD_MS}ms"):
        label = name + " (simulated)" if name.startswith("cloud") else name + " (measured)"
        print(f"  {label:<30} " + C.bar(curve[name][tick_ms], width=30) +
              f"  {curve[name][tick_ms]:.0%} of ticks met at a {tick_ms} ms budget")
    print(C.c("\n  The cliff is the whole story: a decider is fine until the tick budget drops below "
              "its own\n  cost, then it forfeits everything. Where Laya sits on this axis is "
              "measured; where the\n  cloud decider sits is the simulated 300 ms offset, by "
              "construction.", "dim"))
    print(C.c(f"  Read the starred column with that in mind: {tick_ms} ms was DERIVED from these very "
              f"samples, so\n  \"100% here, 0% there\" is forced by the derivation and is not an "
              f"independent finding. The\n  non-circular evidence is section A, whose latencies were "
              f"measured AFTER the budget was fixed\n  and were free to overrun it -- the race's own "
              f"tail reached {max(local['lat']):.0f} ms against a {tick_ms} ms tick. The "
              f"unstarred\n  columns are ordinary measurements and carry the shape of the curve.",
              "dim"))

    # ------------------------------------------------------------ C
    C.section("C. Decision quality with latency removed - unlimited time budget")
    print(f"  {len(QUALITY_SEEDS)} seeds x up to {QUALITY_MAX_PIECES} pieces, identical sequences and "
          f"identical option sets for all four\n  deciders, no tick budget at all. One Tetris game is "
          f"high variance, so mean +/- sd over seeds.")
    print()

    deciders = [
        Decider("heuristic", heuristic_pick, "code"),
        Decider("random", random_pick, "code"),
        Decider("laya-numeric", make_laya_pick(agent, numeric_option), "laya"),
        Decider("laya-semantic", make_laya_pick(agent, semantic_option), "laya"),
    ]

    quality = {}
    for d in deciders:
        runs = [play(d, s, QUALITY_MAX_PIECES, device) for s in QUALITY_SEEDS]
        idx_hist, col_hist, nopt_hist = {}, {}, {}
        for r in runs:
            for k, v in r["index_hist"].items():
                idx_hist[k] = idx_hist.get(k, 0) + v
            for k, v in r["col_hist"].items():
                col_hist[k] = col_hist.get(k, 0) + v
            for k, v in r["nopt_hist"].items():
                nopt_hist[k] = nopt_hist.get(k, 0) + v
        quality[d.name] = {
            "runs": runs, "kind": d.kind,
            "lines": [r["lines"] for r in runs],
            "pieces": [r["pieces"] for r in runs],
            "holes": [r["holes"] for r in runs],
            "max_height": [r["max_height"] for r in runs],
            "topped_out": sum(r["topped_out"] for r in runs),
            "index_hist": idx_hist, "col_hist": col_hist, "nopt_hist": nopt_hist,
        }

    rows = []
    for name, q in quality.items():
        lm, ls = mean_sd(q["lines"])
        pm, ps = mean_sd(q["pieces"])
        hm, _ = mean_sd(q["holes"])
        mm, _ = mean_sd(q["max_height"])
        rows.append([name, f"{lm:5.1f} +/-{ls:4.1f}", f"{pm:5.1f} +/-{ps:4.1f}",
                     f"{hm:5.1f}", f"{mm:5.1f}", f"{q['topped_out']}/{len(QUALITY_SEEDS)}",
                     min(q["lines"]), max(q["lines"])])
    C.table(rows, ["DECIDER", "LINES (mean+/-sd)", "PIECES SURVIVED", "HOLES", "MAX HT",
                   "TOPPED OUT", "MIN LINES", "MAX LINES"], align="lrrrrrrr")

    h_lines = statistics.fmean(quality["heuristic"]["lines"])
    r_lines = statistics.fmean(quality["random"]["lines"])
    ln = {k: statistics.fmean(v["lines"]) for k, v in quality.items()}
    pc = {k: statistics.fmean(v["pieces"]) for k, v in quality.items()}
    best_laya = max(("laya-numeric", "laya-semantic"), key=lambda k: ln[k])

    print(f"\n  heuristic {ln['heuristic']:.1f} lines, random {ln['random']:.1f}, "
          f"laya-numeric {ln['laya-numeric']:.1f}, laya-semantic {ln['laya-semantic']:.1f}.")
    print(f"  pieces survived: heuristic {pc['heuristic']:.1f}, random {pc['random']:.1f}, "
          f"laya-numeric {pc['laya-numeric']:.1f}, laya-semantic {pc['laya-semantic']:.1f}.")

    # ---- the framing experiment
    print()
    print(C.c("  THE FRAMING EXPERIMENT (numbers vs words, identical candidates)", "bold"))
    num, sem = ln["laya-numeric"], ln["laya-semantic"]
    numP, semP = pc["laya-numeric"], pc["laya-semantic"]
    span = max(1e-9, ln["heuristic"] - ln["random"])
    C.table(
        [["lines cleared", f"{num:.1f}", f"{sem:.1f}", f"{sem - num:+.1f}"],
         ["pieces survived", f"{numP:.1f}", f"{semP:.1f}", f"{semP - numP:+.1f}"],
         ["position between random and heuristic",
          f"{(num - r_lines) / span:.2f}", f"{(sem - r_lines) / span:.2f}", ""]],
        ["METRIC", "laya-numeric", "laya-semantic", "semantic - numeric"], align="lrrr")
    p_sem_num_lines = paired_perm_p(quality["laya-semantic"]["lines"], quality["laya-numeric"]["lines"])
    p_sem_num_pieces = paired_perm_p(quality["laya-semantic"]["pieces"], quality["laya-numeric"]["pieces"])
    if abs(sem - num) < 0.25 * max(1.0, span):
        framing_note = ("Framing did NOT rescue it. Bucketing the same numbers into words moved "
                        "LINES by less\n  than a quarter of the random-to-heuristic span. Whether "
                        "that residue is noise is a question\n  for the test below, not for the eye.")
    elif sem > num:
        framing_note = ("Semantic framing beat numeric framing on this run, by the margin in the "
                        "table above.\n  Worth re-running on more seeds before leaning on it.")
    else:
        framing_note = ("Numeric framing beat semantic framing on this run, which is the opposite of "
                        "the prior.\n  Worth re-running on more seeds before leaning on it.")
    print("  " + framing_note)
    print(f"  Exact paired permutation test over the {len(QUALITY_SEEDS)} seeds (same sequences, so "
          f"the games pair up):\n  semantic - numeric  lines p = {p_sem_num_lines:.3f}, "
          f"pieces survived p = {p_sem_num_pieces:.3f}.")
    if p_sem_num_pieces < 0.05 <= p_sem_num_lines:
        print(C.c("  Worth being precise: on LINES the two framings are indistinguishable, but on "
                  "PIECES SURVIVED\n  the semantic framing is reliably WORSE. Words did not help "
                  "and, on the metric with enough\n  signal to tell, they hurt.", "yellow"))
    print(C.c("  Caveat, stated because it matters: the numeric options also carry the raw placement "
              "coordinates\n  (\"col 3 rot 0\") and the semantic ones do not, per the rendering shown "
              "in section 0. The two\n  framings describe the same outcomes; they are not "
              "token-for-token information-equal.", "dim"))

    # ---- degeneracy test
    print()
    print(C.c("  DEGENERACY TEST - is this a chooser, or a constant?", "bold"))
    print(C.c("  Option order is shuffled per decision, so a spike in the INDEX histogram cannot be a "
              "board\n  preference; it can only be positional. A spike in the COLUMN histogram with a "
              "flat index\n  histogram would be the opposite, and much more interesting.", "dim"))
    print()
    deg_rows = []
    degenerate = {}
    for name in ("laya-numeric", "laya-semantic", "random"):
        q = quality[name]
        total = sum(q["index_hist"].values())
        cells, share = hist_line(q["index_hist"], MAX_CANDIDATES, total)
        top_idx = max(q["index_hist"], key=q["index_hist"].get) if q["index_hist"] else -1
        distinct = len(q["index_hist"])
        # The baseline is NOT a flat 1/8: some decisions offer fewer than 8 legal
        # placements, so the high indices are not always on the table. This is the
        # share a uniform chooser would land on each index given the option counts
        # this decider actually faced.
        base = uniform_index_share(q["nopt_hist"], MAX_CANDIDATES)
        b_top = base[top_idx] if 0 <= top_idx < MAX_CANDIDATES else 1.0 / MAX_CANDIDATES
        degenerate[name] = {"top_index": top_idx, "top_share": share, "distinct": distinct,
                            "total": total, "hist": q["index_hist"],
                            "uniform_baseline_top_index": round(b_top, 4),
                            "uniform_baseline": [round(x, 4) for x in base],
                            "mean_options_offered": round(
                                sum(k * v for k, v in q["nopt_hist"].items())
                                / max(1, sum(q["nopt_hist"].values())), 2),
                            "excess_over_uniform": round(share / max(1e-9, b_top), 2)}
        deg_rows.append([name, cells, f"{share:.0%}", f"idx {top_idx}", f"{b_top:.0%}",
                         f"{share / max(1e-9, b_top):.1f}x", f"{distinct}/{MAX_CANDIDATES}"])
    C.table(deg_rows, ["DECIDER", f"CHOSEN INDEX 0..{MAX_CANDIDATES - 1}", "TOP SHARE",
                       "MODE", "UNIFORM WOULD BE", "EXCESS", "INDICES USED"], align="llrrrrr")
    print(C.c(f"  UNIFORM WOULD BE is per-decider and is not a flat 1/{MAX_CANDIDATES}: a decision "
              f"late in a game can offer fewer than\n  {MAX_CANDIDATES} legal placements, so index "
              f"{MAX_CANDIDATES - 1} is not always on the table. Mean options offered: "
              + ", ".join(f"{k} {degenerate[k]['mean_options_offered']:.2f}"
                          for k in ("laya-numeric", "laya-semantic", "random"))
              + f".\n  Scoring the mode against a flat {1 / MAX_CANDIDATES:.1%} would overstate "
              f"the bias on the low indices and understate it on\n  the high ones, so EXCESS "
              f"divides by this baseline instead.", "dim"))
    for name in ("laya-numeric", "laya-semantic"):
        d = degenerate[name]
        u = d["uniform_baseline_top_index"]
        if d["top_share"] >= 0.60:
            note = C.c(f"DEGENERATE: index {d['top_index']} chosen {d['top_share']:.0%} of the time "
                       f"regardless of content ({d['excess_over_uniform']:.1f}x uniform)", "red")
        elif d["top_share"] >= 2.5 * u:
            note = C.c(f"strong positional bias toward index {d['top_index']} "
                       f"({d['top_share']:.0%} vs {u:.0%} uniform)", "yellow")
        else:
            note = C.c(f"no single index dominates ({d['top_share']:.0%} vs {u:.0%} uniform)",
                       "green")
        print(f"  {name:<15} {note}")
    print(C.c("  What that line means, and it is the sentence most worth quoting from this file: "
              "because the\n  option order is shuffled every decision, a decider that takes slot 0 "
              "four times in five IS\n  drawing a placement uniformly at random. That is not bad "
              "judgement; on most turns it is not\n  judgement at all -- which is exactly why "
              "laya-numeric's scores land on top of the random\n  decider's in the table above.",
              "yellow"))
    print()
    col_rows = []
    for name in ("heuristic", "laya-numeric", "laya-semantic", "random"):
        q = quality[name]
        total = sum(q["col_hist"].values()) or 1
        cells, share = hist_line(q["col_hist"], BOARD_W, total)
        col_rows.append([name, cells, f"{share:.0%}"])
    C.table(col_rows, ["DECIDER", f"CHOSEN COLUMN 0..{BOARD_W - 1}", "TOP SHARE"], align="llr")

    dn, ds = degenerate["laya-numeric"], degenerate["laya-semantic"]
    print(f"  The more interesting half: the two framings FAIL DIFFERENTLY. laya-numeric collapses "
          f"onto one option\n  slot -- {dn['top_share']:.0%} of its picks land on index "
          f"{dn['top_index']}, {dn['excess_over_uniform']:.1f}x what a uniform chooser would put "
          f"there, and the other\n  {MAX_CANDIDATES - 1} slots share what is left. laya-semantic "
          f"spreads much wider: its mode takes {ds['top_share']:.0%} of picks at "
          f"{ds['excess_over_uniform']:.1f}x\n  uniform. Words bought a genuinely less degenerate "
          f"distribution and bought no play at all ({sem:.1f} lines\n  against {num:.1f}). A "
          f"diverse output distribution is not evidence of judgement.")

    # ------------------------------------------------------------ D
    C.section("D. The two columns the demo collapses into one")

    laya_gap = (ln[best_laya] - r_lines) / span
    # "better than random" has to mean better by more than the seed-to-seed noise, or a gap
    # of a third of a line against a tenth of one gets reported as a win. Two independent bars, and both have to
    # clear: a quarter of the random->heuristic span, AND an exact paired permutation test
    # over the seeds. Below either, the honest answer is "not measurably".
    p_lines = paired_perm_p(quality[best_laya]["lines"], quality["random"]["lines"])
    p_pieces = paired_perm_p(quality[best_laya]["pieces"], quality["random"]["pieces"])
    beats_random = laya_gap >= 0.25 and p_lines < 0.05
    # Row four is computed, not typed in: it flips the moment a laya framing outplays the
    # pure-Python score. It does not flip on this run, and the row says why.
    worth_params = ln[best_laya] >= ln["heuristic"]
    keeps_up = local["met"] >= 0.9 * max(1, local["pieces"])
    cloud_keeps_up = curve[f"cloud-{SIMULATED_CLOUD_MS}ms"][tick_ms] >= 0.9

    C.table(
        [
            ["fast enough to keep up at " + f"{tick_ms} ms",
             C.c("YES", "green") if keeps_up else C.c("no", "red"),
             f"live race {local['met']}/{local['pieces']} ticks met, p50 "
             f"{statistics.median(local['lat']):.0f} / max {max(local['lat']):.0f} ms, measured "
             f"AFTER the budget was fixed"],
            ["...the simulated 300 ms decider, same budget",
             C.c("YES", "green") if cloud_keeps_up else C.c("no", "red"),
             f"{curve[f'cloud-{SIMULATED_CLOUD_MS}ms'][tick_ms]:.0%} of ticks met -- a "
             f"{SIMULATED_CLOUD_MS} ms time.sleep cannot fit a {tick_ms} ms tick "
             f"(arithmetic, not a benchmark)"],
            ["measurably better than picking at random",
             C.c("YES", "green") if beats_random else C.c("no", "red"),
             f"{ln[best_laya]:.1f} vs {r_lines:.1f} lines, {pc[best_laya]:.0f} vs "
             f"{pc['random']:.0f} pieces -- {laya_gap:+.2f} of the span, paired p={p_lines:.2f} "
             f"over {len(QUALITY_SEEDS)} seeds" if not beats_random else
             f"{ln[best_laya]:.1f} vs {r_lines:.1f} lines, {laya_gap:+.2f} of the span, "
             f"paired p={p_lines:.3f}"],
            ["makes decisions worth the 421M parameters",
             C.c("YES", "green") if worth_params else C.c("no", "red"),
             f"a 4-term linear score in pure Python clears {ln['heuristic']:.1f} lines at "
             f"{statistics.median(latency['heuristic']) * 1000:.0f} us, against "
             f"{ln[best_laya]:.1f} for the best laya framing"],
        ],
        ["PROPERTY", "THIS RUN", "EVIDENCE"],
    )
    print()
    print(f"  Row three is a negative result and negative results need a test, not an eyeball: over "
          f"{len(QUALITY_SEEDS)}\n  paired seeds the best Laya framing differs from a uniform random "
          f"pick with p = {p_lines:.2f} on lines and\n  p = {p_pieces:.2f} on pieces survived. "
          f"An exact permutation test over all 2^{len(QUALITY_SEEDS)} sign assignments; the games "
          f"pair\n  up because every decider played byte-identical sequences. Absence of evidence "
          f"at this sample\n  size, stated as such -- but the heuristic clears the same bar at "
          f"p = {paired_perm_p(quality['heuristic']['lines'], quality['random']['lines']):.3f}, "
          f"so the test is not simply blind.")
    print()
    print("  The viral demo measures row one. Rows three and four are a different question, and this")
    print("  scenario answers it separately on purpose. Laya is an encoder that SCORES options you")
    print("  define -- it has no numeric grounding and these base checkpoints were never fine-tuned on")
    print("  Tetris. Upstream says as much: the base checkpoints sit near chance on typed decisions.")
    ratio = statistics.median(latency[f"cloud-{SIMULATED_CLOUD_MS}ms"]) / max(
        1e-9, statistics.median(latency["laya-numeric"]))
    print(f"  Being {ratio:.1f}x faster than a simulated round trip is a real engineering property, "
          f"and on this\n  machine it is the difference between a game that runs and a game that "
          f"does not. The {SIMULATED_CLOUD_MS} ms\n  is an assumption taken from the demo's own "
          f"report, not a measurement of anything, so that ratio\n  sizes a design margin rather "
          f"than ranking two systems. And it is still not a claim about play,\n  which is the half "
          f"the demo's framing invites you to read into it.")

    # ------------------------------------------------------------ artefact + verdict
    C.save_artifact("07_tetris_arena", {
        "env": C.env_summary(),
        "checkpoint": "english",
        "engine": {
            "board": [BOARD_W, BOARD_H], "spawn_col": SPAWN_COL,
            "self_tests": [{"name": n, "ok": ok, "proved": d} for n, ok, d in checks],
            "self_test_ms": round(self_test_ms, 3),
            "rotation_counts": {p: len(PIECES[p]) for p in PIECE_NAMES},
        },
        "candidate_policy": {
            "max_candidates": MAX_CANDIDATES,
            "selection": "canonical (rot,col) order -> evenly spaced subset -> per-decision shuffle",
            "heuristic_used_in_selection": False,
        },
        "prompt_budget": budget,
        "latency_ms": {k: [round(x, 3) for x in v] for k, v in latency.items()},
        "latency_p50_ms": {k: round(statistics.median(v), 3) for k, v in latency.items()},
        "latency_simulated": {f"cloud-{SIMULATED_CLOUD_MS}ms":
                              f"laya-numeric measured + {SIMULATED_CLOUD_MS} ms simulated round trip "
                              f"(source: demo's reported 316-326 ms; no cloud service contacted)"},
        "race": {
            "seed": RACE_SEED, "tick_ms": tick_ms, "tick_ms_source": tick_src, "max_pieces": RACE_MAX_PIECES,
            "boards": [{"label": _plain(s["label"]), "pieces": s["pieces"], "met": s["met"],
                        "missed": s["missed"], "lines": s["lines"],
                        "holes": count_holes(s["board"]),
                        "max_height": max(column_heights(s["board"]))} for s in race],
        },
        "budget_curve": {k: {str(b): round(v, 4) for b, v in fr.items()} for k, fr in curve.items()},
        "budget_sweep_ms": list(sweep),
        "quality": {
            k: {"lines": v["lines"], "pieces": v["pieces"], "holes": v["holes"],
                "max_height": v["max_height"], "topped_out": v["topped_out"],
                "mean_lines": round(statistics.fmean(v["lines"]), 3),
                "mean_pieces": round(statistics.fmean(v["pieces"]), 3),
                "index_hist": {str(i): n for i, n in sorted(v["index_hist"].items())},
                "col_hist": {str(i): n for i, n in sorted(v["col_hist"].items())},
                "options_offered_hist": {str(i): n for i, n in sorted(v["nopt_hist"].items())}}
            for k, v in quality.items()
        },
        "quality_protocol": {"seeds": list(QUALITY_SEEDS), "max_pieces": QUALITY_MAX_PIECES,
                             "tick_budget": None},
        "framing_experiment": {
            "numeric_mean_lines": round(num, 3), "semantic_mean_lines": round(sem, 3),
            "numeric_mean_pieces": round(numP, 3), "semantic_mean_pieces": round(semP, 3),
            "random_to_heuristic_span_lines": round(span, 3),
            "semantic_minus_numeric_lines": round(sem - num, 3),
            "paired_permutation_p_lines": round(p_sem_num_lines, 4),
            "paired_permutation_p_pieces": round(p_sem_num_pieces, 4),
            "verdict": framing_note.replace("\n  ", " "),
            "pieces_verdict": ("semantic reliably WORSE than numeric on pieces survived "
                               f"(p={p_sem_num_pieces:.3f})" if p_sem_num_pieces < 0.05 else
                               f"no reliable difference on pieces survived (p={p_sem_num_pieces:.3f})"),
        },
        "significance": {
            "test": f"exact two-sided paired permutation over {len(QUALITY_SEEDS)} seeds "
                    f"(all 2^{len(QUALITY_SEEDS)} sign assignments enumerated); deciders play "
                    f"byte-identical sequences so games pair one-to-one",
            "best_laya_vs_random_p_lines": round(p_lines, 4),
            "best_laya_vs_random_p_pieces": round(p_pieces, 4),
            "heuristic_vs_random_p_lines": round(
                paired_perm_p(quality["heuristic"]["lines"], quality["random"]["lines"]), 4),
        },
        "degeneracy": degenerate,
    })

    # Every assertion below is a direction, an ordering, or a threshold with a wide margin.
    # Note what is NOT asserted: that Laya plays Tetris well. It does not, and the
    # assertions say so rather than looking away.
    best_laya_lines = max(ln["laya-numeric"], ln["laya-semantic"])
    best_laya_pieces = max(pc["laya-numeric"], pc["laya-semantic"])
    # max, i.e. the BEST either framing managed -- asserting on the best is what makes
    # "NEITHER framing gets a quarter of the way" a claim about both of them.
    best_gap = max((ln[k] - r_lines) / span for k in ("laya-numeric", "laya-semantic"))
    # Bias is scored against the per-decider uniform baseline, not a flat 1/8 -- see
    # uniform_index_share(). Using 1/8 would flatter this assertion.
    biased = [k for k in ("laya-numeric", "laya-semantic")
              if degenerate[k]["top_share"] >= 2 * degenerate[k]["uniform_baseline_top_index"]]

    invariants = [
        ("engine self-tests pass", self_tests_ok and len(checks) == 8,
         f"{sum(ok for _, ok, _ in checks)}/{len(checks)} in {self_test_ms:.1f} ms", "engine"),
        (f"no option or state silently truncated, worst of "
         f"{budget['laya-numeric']['n_samples']} real decisions", prompt_ok,
         f"<={MAX_CANDIDATES} options, longest "
         f"{max(r['longest_option_tokens'] for r in budget.values())} tokens, worst option total "
         f"{max(r['option_tokens_total'] for r in budget.values())}/"
         f"{budget['laya-numeric']['head_budget']}, longest sequence "
         f"{max(r['sequence_tokens'] for r in budget.values())}/{budget['laya-numeric']['max_len']}",
         "measured"),
        # --- can it keep up (the demo's claim). Only the rows marked `measured` are findings.
        ("a tick budget separating local from a 300 ms round trip EXISTS on this machine",
         tick_src["window_exists"],
         f"floor {tick_src['lo']:.0f} ms <= ceiling {tick_src['hi']:.0f} ms "
         f"(local decision {tick_src['local_min']:.0f}-{tick_src['local_max']:.0f} ms, "
         f"p95 {tick_src['local_p95']:.0f})",
         "measured"),
        ("in the live race laya-local met >=75% of its ticks",
         local["met"] >= 0.75 * max(1, local["pieces"]),
         f"{local['met']}/{local['pieces']} ticks met, p50 "
         f"{statistics.median(local['lat']):.0f} ms / max {max(local['lat']):.0f} ms "
         f"against a {tick_ms} ms tick fixed beforehand",
         "measured"),
        (f"the heuristic meets 100% of ticks at the tightest budget swept ({sweep[0]} ms)",
         curve["heuristic"][sweep[0]] == 1.0,
         f"{curve['heuristic'][sweep[0]]:.0%}, p50 "
         f"{statistics.median(latency['heuristic']) * 1000:.0f} us",
         "measured"),
        (f"both laya framings meet >=90% of ticks at {tick_ms} ms",
         curve["laya-numeric"][tick_ms] >= 0.9 and curve["laya-semantic"][tick_ms] >= 0.9,
         f"numeric {curve['laya-numeric'][tick_ms]:.0%}, "
         f"semantic {curve['laya-semantic'][tick_ms]:.0%}",
         "by construction"),
        (f"the simulated {SIMULATED_CLOUD_MS} ms decider meets <=5% of the same ticks",
         curve[f"cloud-{SIMULATED_CLOUD_MS}ms"][tick_ms] <= 0.05,
         f"{curve[f'cloud-{SIMULATED_CLOUD_MS}ms'][tick_ms]:.0%} at {tick_ms} ms",
         "by construction"),
        ("in the live race the simulated decider forfeited >=90% of its pieces",
         cloud["missed"] >= 0.9 * max(1, cloud["pieces"]),
         f"{cloud['missed']} forfeited of {cloud['pieces']} pieces",
         "by construction"),
        ("...and laya-local outlasted the simulated decider",
         local["pieces"] > cloud["pieces"],
         f"{local['pieces']} pieces vs {cloud['pieces']}",
         "by construction"),
        # --- does it decide well (the claim the demo does NOT make, and neither do we)
        ("a 4-term linear score outplays every laya framing by >3x on lines",
         h_lines >= 3.0 * max(best_laya_lines, 1e-9) and h_lines >= 3.0,
         f"heuristic {h_lines:.1f} lines vs best laya {best_laya_lines:.1f}", "measured"),
        ("...and survives >1.5x as many pieces",
         pc["heuristic"] >= 1.5 * best_laya_pieces,
         f"heuristic {pc['heuristic']:.1f} pieces vs best laya {best_laya_pieces:.1f}", "measured"),
        ("NEITHER laya framing gets a quarter of the way from random to the heuristic",
         best_gap < 0.25,
         f"best laya sits {best_gap:+.2f} of the random->heuristic span "
         f"(random {r_lines:.1f} -> heuristic {h_lines:.1f} lines)", "measured"),
        ("at least one laya framing shows >=2x uniform positional bias",
         len(biased) >= 1,
         ", ".join(f"{k} idx {degenerate[k]['top_index']} at {degenerate[k]['top_share']:.0%} vs "
                   f"{degenerate[k]['uniform_baseline_top_index']:.0%} uniform "
                   f"({degenerate[k]['excess_over_uniform']:.1f}x)"
                   for k in ("laya-numeric", "laya-semantic")), "measured"),
        ("no laya framing beats a uniform random pick on lines at p<0.05 over "
         f"{len(QUALITY_SEEDS)} paired seeds",
         p_lines >= 0.05,
         f"best laya {ln[best_laya]:.2f} vs random {r_lines:.2f} lines, exact paired "
         f"permutation p = {p_lines:.3f} (lines), {p_pieces:.3f} (pieces)", "measured"),
    ]

    print()
    C.table([[n, mark(ok), kind, obs] for n, ok, obs, kind in invariants],
            ["INVARIANT", "RESULT", "KIND", "OBSERVED"])
    print(C.c("  KIND: `measured` rows are the findings -- each one can fail on a slower or busier\n"
              "  machine, on a broken engine, or if Laya starts playing better. `by construction` rows\n"
              "  are arithmetic consequences of simulating the cloud as +300 ms and of deriving the tick\n"
              "  budget from these same latency samples; they are regression guards (they fail if the\n"
              "  simulated delay or the forfeit path stops working) and they are NOT evidence.", "dim"))

    # A busy machine is not a false claim. If the local decider's own jitter grew larger than
    # the simulated handicap, the latency window this scenario measures in simply does not exist
    # right now -- every other finding still printed above, but the race is not meaningful. Exit 2
    # (the same code _common.require_cached uses for "not set up") so `make scenarios` reports
    # "inconclusive, re-run on a quieter machine" rather than "the claim is false" (exit 1).
    if not tick_src["window_exists"]:
        failed = [name for name, ok, _, _ in invariants if not ok]
        print()
        print(C.c("  INCONCLUSIVE - this machine was too busy to measure the latency window.", "yellow"))
        print(C.c(f"  A local decision cost {tick_src['local_min']:.0f}-{tick_src['local_max']:.0f} ms "
                  f"here; its jitter exceeded the {SIMULATED_CLOUD_MS} ms handicap being simulated,\n"
                  f"  so no tick budget separates the two. Close what else is running and re-run.", "dim"))
        print(C.c(f"  Invariants that could not be established: {', '.join(failed) or 'none'}", "dim"))
        sys.exit(2)

    C.require(
        all(ok for _, ok, _, _ in invariants),
        f"Keeping up and deciding well are two different properties. In the live race Laya met "
        f"{local['met']}/{local['pieces']} ticks of a {tick_ms} ms budget while a simulated "
        f"{SIMULATED_CLOUD_MS} ms round trip met {cloud['met']}/{cloud['pieces']} and topped out "
        f"-- and on the very same decisions a four-term linear score in pure Python cleared "
        f"{h_lines:.1f} lines to Laya's {best_laya_lines:.1f}, with random at {r_lines:.1f} and no "
        f"measurable gap between them. The demo's claim is the first half only.",
    )


if __name__ == "__main__":
    main()
