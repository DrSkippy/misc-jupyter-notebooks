#!/usr/bin/env python3
""" Example file for meta searches of the parameter space.
    Use this one as an example and do whatever you want!

    This file searchs the bet_frac (fraction of the bank to bet)
    for optimization.

    ./meta.py           search strategy.py
    ./meta.py Jeff      search strategy.py.Jeff (it must read kwargs["bet_frac"])"""

import sys
from sim import *

strategy_fn = load_strategy(sys.argv[1]) if len(sys.argv) > 1 else None
for i in range(60):
    bet_frac = 0.05 + i*(.002)
    print("{}, {}".format(bet_frac, get_bank(strategy_fn, bet_frac=bet_frac)))
