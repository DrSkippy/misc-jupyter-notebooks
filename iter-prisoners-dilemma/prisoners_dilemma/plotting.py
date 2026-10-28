"""Shared matplotlib styling for the notebooks.

Each strategy keeps the same color in every chart: the seven 2008 strategies and TF2T take the eight categorical
slots, the other classics are gray and told apart by line style and direct labels.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COLORS = {"Random": "#eda100", "T4T": "#2a78d6", "AvgT4T": "#1baf7a", "Pred1": "#e87ba4", "Defect": "#e34948",
          "T4TDefect": "#eb6834", "T4TForgive": "#6250d6", "TF2T": "#008300"}
GRAY = "#8a8985"
DASHES = {"Cooperate": (0, (4, 2)), "Grudger": (0, (1, 1.5)), "Pavlov": (0, (6, 2, 1, 2)),
          "SuspiciousT4T": (0, (2, 2))}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e1", "axes.axisbelow": True})


def color(name):
    return COLORS.get(name, GRAY)


def line_kw(name):
    return {"color": color(name), "linestyle": DASHES.get(name, "-"), "linewidth": 2}


def plot_lines(table, ax, label_min=None, legend=True):
    """One line per column of ``table`` (index on x). With ``label_min`` set, lines whose last value is at least
    that are also labeled at the right end, nudged apart so the labels do not overlap."""
    for name in table.columns:
        ax.plot(table.index, table[name], label=name, **line_kw(name))
    if label_min is not None:
        last = table.iloc[-1].sort_values()
        last = last[last >= label_min]
        lo, hi = ax.get_ylim()
        x = table.index[-1]
        dx = 0.01 * (table.index[-1] - table.index[0])
        prev = None
        for name, y in last.items():
            y_text = y if prev is None else max(y, prev + 0.045 * (hi - lo))
            ax.annotate(name, (x, y), xytext=(x + dx, y_text), va="center", fontsize=9, color="#3d3c39")
            prev = y_text
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1), frameon=False, fontsize=9)
