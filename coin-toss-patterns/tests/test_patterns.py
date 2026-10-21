import numpy as np
import pytest

from coin_toss_patterns import patterns


def test_all_sequences():
    assert patterns.all_sequences(2) == ["HH", "HT", "TH", "TT"]
    assert len(set(patterns.all_sequences(5))) == 32


@pytest.mark.parametrize("pattern", ["H", "TH", "HH", "THT", "THH", "TTTH", "HTHT"])
def test_count_matches_brute_force(pattern):
    for n in range(0, 11):
        assert patterns.count_without_pattern(n, pattern) == len(patterns.sequences_without_pattern(n, pattern))


def test_hh_counts_are_fibonacci():
    counts = [patterns.count_without_pattern(n, "HH") for n in range(1, 22)]
    assert counts[:5] == [2, 3, 5, 8, 13]
    assert counts[-1] == 28657


def test_counts_match_2008_brute_force_table():
    # n = 5, 10, 21 rows of the 2008 script's output
    for pattern, expected in {"TH": [6, 11, 22], "HH": [13, 144, 28657],
                              "THT": [21, 351, 170625], "THH": [20, 232, 46367]}.items():
        assert [patterns.count_without_pattern(n, pattern) for n in (5, 10, 21)] == expected


def test_survival_table():
    table = patterns.survival_table(["THT", "THH"], 21)
    np.testing.assert_allclose(table.loc[5], [21 / 32, 20 / 32])
    np.testing.assert_allclose(table.loc[21], [0.081, 0.022], atol=5e-4)


@pytest.mark.parametrize("pattern,expected", [("H", 2), ("HT", 4), ("HH", 6), ("THT", 10), ("THH", 8),
                                              ("TTTH", 16), ("HHHHH", 62)])
def test_expected_tosses(pattern, expected):
    assert patterns.expected_tosses(pattern) == expected


@pytest.mark.parametrize("pattern", ["HH", "THT", "TTTH"])
def test_pmf_sums_to_one_with_conway_mean(pattern):
    pmf = patterns.waiting_time_pmf(pattern, 400)
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf.index * pmf).sum() == pytest.approx(patterns.expected_tosses(pattern))
    assert (pmf.loc[: len(pattern) - 1] == 0).all()


def test_simulation_never_ends_before_pattern_length():
    waits = patterns.simulate_waiting_times("TTTH", 5000, rng=0)
    assert waits.min() == 4


def test_simulation_mean_is_close_to_exact():
    for pattern in ["H", "HT", "HH", "THT", "TTTH"]:
        waits = patterns.simulate_waiting_times(pattern, 20_000, rng=1)
        assert waits.mean() == pytest.approx(patterns.expected_tosses(pattern), rel=0.05)


def test_simulation_is_reproducible():
    a = patterns.simulate_waiting_times("HTH", 1000, rng=3)
    b = patterns.simulate_waiting_times("HTH", 1000, rng=3)
    np.testing.assert_array_equal(a, b)


def test_waiting_time_counts_fills_gaps():
    counts = patterns.waiting_time_counts(np.array([2, 2, 5]))
    assert counts.to_dict() == {2: 2, 3: 0, 4: 0, 5: 1}


def test_bad_pattern():
    with pytest.raises(ValueError):
        patterns.expected_tosses("HX")
