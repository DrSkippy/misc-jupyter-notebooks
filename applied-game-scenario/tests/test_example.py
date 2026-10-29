from negotiations import example, game


def test_read_spreadsheet_matches_table():
    t = example.read_spreadsheet()
    assert t.equals(game.parties(example.SPREADSHEET))
    assert list(t.index)[:2] == ["d2", "customer rep"]
