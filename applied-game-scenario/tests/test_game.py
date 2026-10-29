import pandas as pd
import pytest

from negotiations import example, game


@pytest.fixture
def table():
    return game.parties(example.SPREADSHEET)


def test_spreadsheet_weighted_average(table):
    # Sheet1!I14 = SUM(S x I x P) / SUM(S x I)
    assert game.weighted_average(table) == pytest.approx(76.60086663456909)
    assert round(game.weighted_average(table), 1) == 76.6


def test_spreadsheet_cumulative_influence(table):
    # Sheet1!E18:E23, the cumulative influence at each occupied position; the majority is reached at 75
    cum = game.summary(table).groupby("position")["cumulative %"].max()
    assert cum.round(2).tolist() == [3.74, 16.82, 19.63, 27.10, 57.01, 100.0]
    assert game.balance_of_power(table, interpolate=False) == 75


def test_balance_of_power_2009_rule(table):
    # halfway point interpolated between 60 (27.1% below) and 75 (29.9% there), as the 2009 code did
    assert game.balance_of_power(table) == pytest.approx(71.484375)
    assert game.balance_of_power(game.parties(example.SCRIPT_2009)) == pytest.approx(71.71875)


def test_balance_of_power_majority_at_lowest_position():
    t = game.parties({"a": [40, 60, 1], "b": [90, 40, 1]})
    assert game.balance_of_power(t) == pytest.approx(40 * 0.5 / 0.6)
    assert game.balance_of_power(t, interpolate=False) == 40


def test_influence_scale_does_not_matter(table):
    scaled = table.assign(influence=table["influence"] * 7)
    assert game.weighted_average(scaled) == pytest.approx(game.weighted_average(table))


def test_coalition_salience():
    assert game.coalition_salience([60]) == 60
    assert game.coalition_salience([100, 20, 5]) == pytest.approx(0.8 * 100 + 0.2 * 12.5)


def test_coalition(table):
    c = game.coalition(table, ["d2", "legal"])
    assert c.index.tolist() == ["d2+legal"]
    row = c.iloc[0]
    assert row["position"] == pytest.approx((80 * 20 * 15 + 100 * 70 * 100) / (80 * 20 + 100 * 70))
    assert row["influence"] == 90
    assert row["salience"] == pytest.approx(0.8 * 100 + 0.2 * 80)


def test_equal_salience_coalition_keeps_center_of_mass(table):
    # d1, me, engineer and legal all have salience 100: merging them changes nothing
    merged = game.coalition(table, ["d1", "me", "engineer", "legal"])
    rest = table.drop(index=["d1", "me", "engineer", "legal"])
    assert game.weighted_average(pd.concat([rest, merged])) == pytest.approx(game.weighted_average(table))
