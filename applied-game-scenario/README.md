# Negotiation Scenarios: the Group's Center of Mass

Where will a decision land when eleven parties pull in different directions? Each party has a position on a
0–100 scale, an influence and a salience (how much it cares). The prediction is the group's center of mass:
positions weighted by salience × influence. Parties can also settle among themselves first and enter the final
negotiation as one coalition. Played out over every possible set of coalitions, that moves the outcome
anywhere from 66 to 84 around the single-negotiation prediction of 76.6.

Rewritten in 2026 from Python 2 scripts and a spreadsheet of 2009–2010. Before the scripts were removed, the
original classes were run under Python 3: the package gives the same weighted average and balance of power, and the same final
position for each of the 115,975 coalition structures of the scripts' 10 parties and the 678,570 of the
spreadsheet's 11 (to 1e-13). It computes them all in under half a second, where the original took 36 s.

## Layout

| Path | Contents |
|---|---|
| `negotiations/game.py` | one negotiation: party table, weighted average, balance of power, the spreadsheet's working columns, merging a coalition into one party |
| `negotiations/model.py` | negotiations in succession: `succession` plays out a plan of nested coalitions step by step; `all_structures` gives the outcome of every coalition structure at once, plus pair statistics |
| `negotiations/partitions.py` | every set partition as restricted growth strings, in the order of Knuth's Algorithm H; Bell numbers |
| `negotiations/example.py` | the worked example's parties, and a standard-library reader for its `.ods` sheet |
| `notebooks/negotiation-scenarios.ipynb` | the spreadsheet example, coalitions and their effect, a nested succession, all 678,570 coalition structures (about 6 s) |
| `tests/` | unit tests, including checks against the spreadsheet's figures and the 2009 code's outcomes |
| `resources/predictioneersgame.ods` | the 2009 spreadsheet: 11 parties, the S × I weighted average (76.6) and cumulative influence |
| `resources/partitions_example.txt` | 2009 output of `Partitioner.py`: the 15 partitions of a, b, c, d |

The notebook adds the project folder to `sys.path` itself, so nothing needs installing in the Jupyter image.

## The model

- **One negotiation:** outcome = Σ S·I·P / Σ S·I. The balance of power is where the cumulative influence,
  counted from 0, reaches half. The 2009 code interpolates within the span below the position where that
  happens (`balance_of_power(..., interpolate=False)` gives the position itself).
- **Coalition:** members settle at their own weighted average and enter as one party with their combined
  influence and salience 0.8 × the keenest member's + 0.2 × the mean of the rest (`game.OTHERS_WEIGHT`).
  If all members care equally, merging changes nothing. Otherwise the keen member lends its salience to the
  others' influence.
- **Succession:** `model.succession(parties, plan)` takes nested lists of names, e.g.
  `[[["d1", "me"], "engineer"], ["legal", "advisor"]]`. Inner lists settle first, and unnamed parties join the final
  round alone.
- **All coalition structures:** `model.all_structures(parties)` evaluates every partition of the parties into
  coalitions, followed by the final negotiation (Bell number B(n): 678,570 for n = 11, 27.6 million for n = 13).

## Results

- The single negotiation predicts 76.6, just above "me" at 75. Salience pulls the outcome below the three
  parties at 100, because the advisor and the investor hold influence but barely care.
- One two-party coalition moves the outcome by up to 5.6 points: d2 (keen, at 15) with the potential investor
  (influential, indifferent, at 100).
- Over all coalition structures the outcome ranges from 65.9 to 83.8. Half land between 74.6 and 77.2.
- What matters most is where the influential, indifferent parties go: with d2 they pull the mean outcome down
  by up to 2.2 points, with legal they push it up by 1.4 to 2.2.
