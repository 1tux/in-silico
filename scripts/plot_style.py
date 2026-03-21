from __future__ import annotations

import matplotlib as mpl


def set_paper_style():
    """Paper-style plotting defaults.

    Single-column: 3.3in wide. Double-column: 6.8in wide.
    """
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": ":",
    })
