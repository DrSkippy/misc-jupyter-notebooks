"""One negotiation among parties placed along a single issue dimension.

Each party has a ``position`` (0 to 100), an ``influence`` (any scale; only shares matter) and a ``salience``
(0 to 100, how much the party cares about this issue). The predicted outcome is the group's center of mass:
positions weighted by salience x influence.

- ``parties``: build the party table from ``{name: [position, influence, salience]}``
- ``weighted_average``: the salience x influence weighted mean position (the predicted outcome)
- ``balance_of_power``: the position where half the influence lies on either side
- ``summary``: the spreadsheet's working columns: influence share, S x I, S x I x P, cumulative influence
- ``coalition``: a group of parties that has settled among itself, as one new party
"""
import numpy as np
import pandas as pd

COLUMNS = ["position", "influence", "salience"]
OTHERS_WEIGHT = 0.2  # a coalition's salience: 0.8 x its keenest member's + 0.2 x the mean of the rest


def parties(data):
    """Party table indexed by name, from ``{name: [position, influence, salience]}`` or a DataFrame."""
    if isinstance(data, pd.DataFrame):
        table = data[COLUMNS]
    else:
        table = pd.DataFrame.from_dict(data, orient="index", columns=COLUMNS)
    table = table.astype(float)
    table.index.name = "party"
    return table


def weighted_average(table):
    """Predicted outcome: mean position weighted by salience x influence."""
    w = table["salience"] * table["influence"]
    return float((w * table["position"]).sum() / w.sum())


def balance_of_power(table, interpolate=True):
    """Position where the cumulative influence, counted from the low end, reaches half.

    With ``interpolate`` (the 2009 rule) the influence at each position is spread evenly back to the previous
    occupied position (to 0 for the lowest), and the halfway point is found within that span. Without it, the
    answer is the occupied position where the majority is reached (the weighted median).
    """
    share = table.groupby("position")["influence"].sum().sort_index()
    share = share / share.sum()
    cum = share.cumsum().to_numpy()
    pos = share.index.to_numpy()
    i = int(np.searchsorted(cum, 0.5 - 1e-12))
    if not interpolate:
        return float(pos[i])
    below, last = (cum[i - 1], pos[i - 1]) if i else (0.0, 0.0)
    return float(last + (0.5 - below) * (pos[i] - last) / share.iloc[i])


def summary(table):
    """The spreadsheet's working: influence as a percent share, S x I, S x I x P, and cumulative influence
    in position order."""
    out = table.sort_values("position").copy()
    out["influence %"] = 100 * out["influence"] / out["influence"].sum()
    out["S x I"] = out["salience"] * out["influence %"]
    out["S x I x P"] = out["S x I"] * out["position"]
    out["cumulative %"] = out["influence %"].cumsum()
    return out


def coalition_salience(salience):
    """Salience of a coalition from its members' saliences: the keenest member counts 0.8, the mean of the rest
    0.2."""
    s = np.asarray(salience, dtype=float)
    top = s.max()
    if len(s) == 1:
        return float(top)
    return float((1 - OTHERS_WEIGHT) * top + OTHERS_WEIGHT * (s.sum() - top) / (len(s) - 1))


def coalition(table, members=None, name=None):
    """Members that have negotiated among themselves, as a single party: at their weighted average position,
    with their combined influence and ``coalition_salience``. Returns a one-row party table."""
    group = table if members is None else table.loc[list(members)]
    if name is None:
        name = "+".join(group.index)
    row = [weighted_average(group), group["influence"].sum(), coalition_salience(group["salience"])]
    return parties({name: row})
