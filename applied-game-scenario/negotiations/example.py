"""The 2009 worked example, and a reader for its spreadsheet.

``resources/predictioneersgame.ods`` (Sheet1) lists 11 parties to a negotiation with their position, influence
and salience, and works out the weighted average (76.6) and where the cumulative influence passes half (75).
The 2009 scripts used the same parties less "Customer 1", with shortened names and salience 99 for 100.

- ``read_spreadsheet``: the party table from an .ods sheet laid out like the example (standard library only)
- ``SPREADSHEET``: the example's 11 parties, as typed into the spreadsheet
- ``SCRIPT_2009``: the 10 parties in the 2009 scripts
"""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from . import game

ODS = Path(__file__).resolve().parent.parent / "resources" / "predictioneersgame.ods"

SPREADSHEET = {
    "d2": [15, 20, 80],
    "customer rep": [35, 50, 5],
    "sales staff": [35, 20, 5],
    "erin": [50, 10, 20],
    "Customer 1": [50, 5, 1],
    "engineer": [60, 40, 100],
    "d1": [75, 80, 100],
    "me": [75, 80, 100],
    "advisor": [100, 80, 20],
    "potential investor": [100, 80, 5],
    "legal": [100, 70, 100],
}

SCRIPT_2009 = {
    "d2": [15, 20, 80],
    "ctcust": [35, 50, 5],
    "ctsal": [35, 20, 5],
    "er": [50, 10, 20],
    "eng": [60, 40, 99],
    "d1": [75, 80, 99],
    "me": [75, 80, 99],
    "adv": [100, 80, 20],
    "inv": [100, 80, 5],
    "legal": [100, 70, 95],
}

_NS = {"table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
       "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
       "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0"}
_VALUE = "{%s}value" % _NS["office"]  # full-precision number; the cell text is as displayed
_REPEAT = "{%s}number-columns-repeated" % _NS["table"]


def _rows(path, sheet):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    tables = {t.get("{%s}name" % _NS["table"]): t for t in root.iter("{%s}table" % _NS["table"])}
    for row in tables[sheet].iter("{%s}table-row" % _NS["table"]):
        cells = []
        for cell in row:
            text = cell.get(_VALUE) or " ".join("".join(p.itertext()) for p in cell.iter("{%s}p" % _NS["text"]))
            cells += [text] * min(int(cell.get(_REPEAT, "1")), 64)
        yield cells


def read_spreadsheet(path=ODS, sheet="Sheet1"):
    """Party table from a sheet laid out like the example: a header row with ``P``, ``Influence`` and
    ``Salience``, then one row per party with its name in the column left of ``P``, ending at the first row
    without a name."""
    rows = _rows(path, sheet)
    for header in rows:
        if "P" in header:
            break
    cols = [header.index(h) for h in ("P", "Influence", "Salience")]
    name_col = cols[0] - 1
    data = {}
    for cells in rows:
        if len(cells) <= max(cols) or not cells[name_col].strip():
            break
        data[cells[name_col].strip()] = [float(cells[c]) for c in cols]
    return game.parties(data)
