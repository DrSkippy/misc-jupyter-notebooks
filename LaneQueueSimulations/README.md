# Lane Queue Simulations

Notebook-driven simulation of shoppers choosing checkout lines. Each arriving shopper inspects `d` randomly chosen lines out of `n` and joins the shortest one (the "power of d choices"). The notebooks sweep `d` from 1 to `n` and compare average queue length and wait time.

## Layout

- `lane_queues/stores.py` — `Line`, `Shopper` and `Store` classes
- `lane_queues/sim_worker.py` — simulation workers (`random_worker`, `lockstep_worker`) and pre-computed random inputs
- `notebooks/sim.ipynb` — parallel sweep over `d` (multiprocessing), with plots of queue length and wait time vs. `d`
- `notebooks/sim2.ipynb` — small vectorized (numpy-only) version that prints each step
- `tests/` — unit tests for the classes

## Running

```bash
poetry install
poetry run pytest
poetry run jupyter notebook notebooks/
```

`sim.ipynb` locates the `lane_queues` package itself, so it also runs in any other Jupyter environment (for example JupyterLab in Docker) without an install. `sim.ipynb` runs in about 15 seconds.

With 7 lines, arrival probability 0.98 and checkout probability 0.15 per step, inspecting 1 line gives an average of about 10 shoppers per line and inspecting 2 gives about 2.5. Checking all 7 lines gives only about 1.5.
