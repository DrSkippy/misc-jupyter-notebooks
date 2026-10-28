# Iterated Prisoner's Dilemma

Which strategies do best when the same two players meet for 200 rounds of the prisoner's dilemma? Every
strategy plays every other in a round robin, and then in a population where each strategy's share of the next
generation is its share of the points scored.

Rewritten in 2026 from a set of Python 2 scripts of February 2008: one class per strategy and an
`agentManager` that ran the population game and wrote tab-separated records. Before the scripts were removed,
the original classes were run under Python 3 against the rewrite: all 49 pairings agree (exactly for the
deterministic pairs, in mean score for the random ones). The 2008 population record and match statistics are kept in `resources/`.

## Layout

| Path | Contents |
|---|---|
| `prisoners_dilemma/strategies.py` | the seven 2008 strategies and five classics (Cooperate, Grudger, Pavlov, TF2T, SuspiciousT4T), each playing many matches at once on numpy arrays |
| `prisoners_dilemma/game.py` | payoffs (T=5, R=3, P=1, S=0) and matches, with optional noise (each move flipped with a given probability) |
| `prisoners_dilemma/tournament.py` | head-to-head table of every pair, score matrix, ranking |
| `prisoners_dilemma/evolution.py` | the 2008 population game (round robin or random pairing), largest-remainder apportionment, replicator dynamics |
| `prisoners_dilemma/plotting.py` | shared chart style: one fixed color per strategy |
| `notebooks/strategy-tournament.ipynb` | head-to-head scores, ranking vs matches won, how the ranking changes with noise (about 5 s) |
| `notebooks/population-dynamics.ipynb` | the 2008 run replayed and compared with its record, replicator dynamics, random pairing, all twelve strategies (50 agents each) with and without noise (about 1.5 min) |
| `tests/` | unit tests, including checks against the 2008 rules, match scores and population record |
| `resources/` | 2008 outputs the notebooks and tests check against: `poprecord.csv` (population after each generation) and `game.scores` (win/loss/tie and point shares per pairing) |

The notebooks add the project folder to `sys.path` themselves, so nothing needs installing in the Jupyter image.

## Adding a strategy

A strategy is a function `move(view, rng)` returning a bool array (True = cooperate), one entry per match. The
`strategies.View` it receives holds the round number `t`, `my_last`, `opp_last`, `opp_last2` and the opponent's
`opp_coops`/`opp_defects` so far. Wrap it as `Strategy(name, description, move, random=...)` (set `random=True`
if it uses `rng`, so matches against it are repeated) and either add it to `CLASSICS` in `strategies.py` or pass
the object directly to `tournament.head_to_head` and `evolution.evolve` alongside strategy names.

## Results

- Defect never loses a match and finishes last; tit for tat never wins one and finishes near the top. Points
  come from mutual cooperation (600 a match) rather than from beating opponents.
- The 2008 population (25 each of seven strategies) loses Defect within about six generations, then Pred1,
  Random and T4TDefect, and settles at AvgT4T 44%, T4TForgive 30%, T4T 24%. The rewrite reproduces the 2008
  population counts exactly for the first two generations.
- With all twelve strategies (50 agents each) and no mistakes, every strategy that defects first dies out and
  the nice ones freeze in place. With 5% of moves flipped, tit for two tats takes over and Grudger and Pavlov
  die out. With 50 agents per strategy the population follows the infinite-population limit to within a couple
  of percentage points; with 15, rounding to whole agents stalled the strategies whose scores were close.

## What changed from 2008

Bugs fixed:

- **Population drift.** Each strategy's new count was rounded on its own, so the population grew from 175 to
  201 agents over 40 generations; the extra agents kept AvgT4T's share climbing to 50% long after the
  matches had stopped changing anything (the rewrite settles at 44%). Counts are now
  apportioned by largest remainder and the total is fixed.
- The constructor divided the caller's `allocation` dict in place, and `makeAgent` returned `None` for an
  unknown name (failing later). Strategies are now looked up by name with a `KeyError` listing the choices.
- The random-pairing tournament ended each generation by catching `IndexError`, which would also have hidden
  any `IndexError` raised inside a match.
- `agentManager.py` opened and truncated three CSV files when imported; `showPool` mixed tabs and spaces (a
  `TabError` in Python 3); `historyLength` did not exist until `set_history` was first called.
- Nothing was seeded, so no run could be repeated. Every function now takes `rng`.

Also changed: Python 3; matches are vectorized (all matches between two strategies play at once, and a pair
with no randomness is played once), so a 40-generation, 175-agent round robin takes about 4 s; payoffs, match
length and noise are parameters; win/loss/tie and score tables come back as DataFrames instead of
tab-separated files.

## Tests

```bash
pip install pytest      # not in the Jupyter image
python -m pytest        # from this folder
```
