# Checkout Queues

Simulations of shoppers in checkout lines.

1. **Power of d choices.** Each arriving shopper inspects `d` randomly chosen lines out of `n` and joins the
   shortest. Sweeping `d` from 1 to `n` shows that most of the benefit comes from looking at just two lines.
2. **One line or several?** Three clerks served either from one shared line or from a line each (customers join
   the shortest line and stay in it).

## Layout

| Path | Contents |
|---|---|
| `checkout_queues/stores.py` | `Line`, `Shopper` and `Store` classes |
| `checkout_queues/sim_worker.py` | power-of-d simulations: shared random inputs, the object model, and an array-only version |
| `checkout_queues/clerks.py` | single shared line vs a line per clerk |
| `notebooks/power-of-d-choices.ipynb` | sweep over `d` (runs in parallel processes), line length and wait vs `d`, Little's law check |
| `notebooks/power-of-d-vectorized.ipynb` | small array-only run printed step by step, checked against the object model |
| `notebooks/single-vs-multiple-queues.ipynb` | the two layouts across arrival rates |
| `tests/` | unit tests |

Notebooks add the project folder to `sys.path` themselves, so nothing needs installing in the Jupyter image.

## Results

With 7 lines, arrival probability 0.98 and checkout probability 0.15 per step (93% utilization), inspecting 1 line
gives an average of about 10.5 shoppers per line and inspecting 2 gives about 2.7; checking all 7 lines gives
1.7. The measured waits agree with Little's law (W = L / λ) to within about 1%.

A single shared line beats a line per clerk on every measure, but its edge in average wait is modest (about 12% at
80% load, shrinking to nothing when overloaded). The bigger differences are in the worst waits and in fairness.

## Tests

```bash
pip install pytest      # not in the Jupyter image
python -m pytest        # from this folder
```
