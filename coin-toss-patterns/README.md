# Coin-Toss Patterns

How many tosses of a fair coin until a pattern such as `HT`, `THT` or `TTTH` first appears? Every pattern of
length k is equally likely in any k given tosses, yet `HT` takes 4 tosses on average and `HH` takes 6. Patterns
that overlap themselves take longer.

Rewritten in 2026 from two Python 2 scripts of April 2008, which simulated the waiting times and counted, by
generating every sequence, how many sequences of length n avoid a pattern. Plots are drawn with pandas and
matplotlib.

## Layout

| Path | Contents |
|---|---|
| `coin_toss_patterns/patterns.py` | vectorized simulation of waiting times; exact counts of sequences without a pattern (a small state machine, plus brute force for checking); exact waiting-time distribution and mean |
| `notebooks/coin-toss-patterns.ipynb` | mean waits for all patterns of length 1–4, the `TTTH` waiting-time distribution, sequences avoiding `TH`, `HH`, `THT`, `THH`, and a run of five heads |
| `tests/` | unit tests, including checks against the 2008 count tables |

The notebook adds the project folder to `sys.path` itself, so nothing needs installing in the Jupyter image.

## Results

- The mean wait is the sum of 2^j over every j where the pattern's first j tosses equal its last j (Conway):
  2^k for patterns that do not overlap themselves (`HT` 4, `TTTH` 16), up to 2^(k+1) − 2 for a run of one face
  (`HHHH` 30, `HHHHH` 62). Simulations of 20,000 runs agree to within about 1%.
- Sequences of length n without `TH` number n + 1; without `HH`, the Fibonacci numbers. `THT` and `THH` are
  avoided equally often up to n = 4 and then diverge (mean waits 10 and 8).
- The exact counts reproduce the 2008 brute-force tables.

## Tests

```bash
pip install pytest      # not in the Jupyter image
python -m pytest        # from this folder
```
