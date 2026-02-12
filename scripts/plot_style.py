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
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": ":",
    })
