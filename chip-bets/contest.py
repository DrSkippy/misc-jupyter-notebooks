""" Contest engine for scoring strategies on any seed, with a per-session trace.

    run_contest() follows the same rules and makes the same sequence of `random` calls as
    sim.get_bank(), so at seed 14338 it reproduces the official score. Strategies that call
    `random` themselves (e.g. Random) share the global generator, just as they do in sim.py.

    run_seeds() scores every submission on many seeds in parallel worker processes.
"""
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor

# sim_parameters.py calls logging.basicConfig(filename='session.log', level=DEBUG). Give the
# root logger a handler first so that call is a no-op and thousands of contests don't write
# a huge debug log.
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger().setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

from sim_parameters import n_wd_w, n_wd_r, n_rd_w, n_rd_r, chip_cost
from submissions import load_all

OFFICIAL_SEED = 14338
N_SESSIONS = 100
START_BANK = 1000


def run_contest(bet_strategy, seed, n_sessions=N_SESSIONS, bank=START_BANK, **kwargs):
    """Play one contest. Returns (final bank, DataFrame with one row per session)."""
    random.seed(a=seed)
    white_dominant = ["w"] * n_wd_w + ["r"] * n_wd_r
    red_dominant = ["r"] * n_rd_r + ["w"] * n_rd_w
    random.shuffle(white_dominant)
    random.shuffle(red_dominant)
    bags = {"wd": white_dominant, "rd": red_dominant}

    bag_keys = [random.choice(["rd", "wd"]) for _ in range(n_sessions)]

    trace = []
    n_session = 0
    while bank >= 0 and n_session < n_sessions:
        bag_key = bag_keys[n_session]
        chip_color = "u"
        session_history = []
        rec = dict(session=n_session, bag=bag_key, chips=0, bet=0,
                   bet_bag=None, correct=None, error=None)
        while True:
            move_choice, bet, bet_bag = bet_strategy(n_session, bank, chip_color, session_history, **kwargs)
            if move_choice.lower() == "bet":
                chip_color = "u"
                bet = max(int(bet), 1)
                if bank > bet:
                    bank += bet if bet_bag == bag_key else -bet
                    rec.update(bet=bet, bet_bag=bet_bag, correct=(bet_bag == bag_key))
                    break
                rec["error"] = "bet >= bank"
                break
            elif move_choice.lower() == "buy":
                try:
                    chip_color = bags[bag_key].pop()
                except IndexError:
                    rec["error"] = "bag empty"
                    break
                if bank > chip_cost:
                    bank -= chip_cost
                    rec["chips"] += 1
                else:
                    rec["error"] = "bank < chip cost"
                    break
            else:
                rec["error"] = "invalid move"
                break
            session_history.append((move_choice, bet, bet_bag, chip_color))
        rec["bank"] = bank
        trace.append(rec)
        n_session += 1
    return bank, pd.DataFrame(trace)


def summarize(final, trace):
    """One contest's headline numbers, including how the strategy plays a session."""
    placed = trace.bet > 0
    bank_before = np.concatenate([[START_BANK], trace.bank.values[:-1]])
    bank_before_bet = bank_before - trace.chips * chip_cost
    return dict(
        final_bank=final,
        bets_placed=int(placed.sum()),
        accuracy=trace.correct[placed].astype(float).mean(),
        chips_bought=int(trace.chips.sum()),
        errors=int(trace.error.notna().sum()),
        chips_per_session=trace.chips.mean(),
        bet_fraction=(trace.bet[placed] / bank_before_bet[placed]).mean(),
    )


_strategies = None


def _score_seed(seed):
    """Worker: play every submission at one seed."""
    global _strategies
    if _strategies is None:
        _strategies = load_all()
    return [dict(seed=seed, strategy=name, **summarize(*run_contest(fn, seed)))
            for name, fn in _strategies.items()]


def run_seeds(seeds, processes=None):
    """Score every strategy.py.<Name> submission on each seed; one row per (seed, strategy).

    Seeds are spread over `processes` worker processes (default: all CPUs). Each contest is
    seeded on its own, so the results do not depend on how seeds are split between workers.
    """
    seeds = list(seeds)
    processes = processes or os.cpu_count()
    with ProcessPoolExecutor(processes) as pool:
        chunks = pool.map(_score_seed, seeds, chunksize=max(1, len(seeds) // (4 * processes)))
        return pd.DataFrame([row for chunk in chunks for row in chunk])
