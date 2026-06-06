"""
Unified Visual Design Language for ML Papers
=============================================
Colorblind-safe (Paul Tol Bright), sized for NeurIPS/ICML/ICLR page geometry.

Usage:
    from style.style import apply_style, figure, save, C, METHOD, BAR, TYPE, ERRBAR, label_subplot
    apply_style()
    fig, ax = figure(width="single")
    ax.plot(x, y, **METHOD["ours"], label="Ours")
    save(fig, "figures/output/fig1_result")
"""
from __future__ import annotations
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. COLOR SYSTEM  (Paul Tol Bright — colorblind-safe)
# ---------------------------------------------------------------------------
C = {
    "ours":       "#4477AA",   # blue   — your main method, always most prominent
    "baseline_a": "#EE6677",   # red    — strongest baseline
    "baseline_b": "#228833",   # green
    "baseline_c": "#CCBB44",   # yellow
    "baseline_d": "#66CCEE",   # cyan
    "baseline_e": "#AA3377",   # purple
    "neutral":    "#BBBBBB",   # grey   — ablations, reference lines
    "dark":       "#222222",   # near-black — annotations, error bars
    "highlight":  "#EE6677",   # callout accent (same as baseline_a)
    "good":       "#228833",   # positive delta
    "bad":        "#EE6677",   # negative delta
}

PALETTE = [C["ours"], C["baseline_a"], C["baseline_b"], C["baseline_c"],
           C["baseline_d"], C["baseline_e"], C["neutral"]]

# ---------------------------------------------------------------------------
# 2. METHOD STYLE REGISTRY  — unique (color + linestyle + marker) per method
#    Splat into ax.plot():  ax.plot(x, y, **METHOD["ours"])
# ---------------------------------------------------------------------------
_LW, _MS = 1.25, 4.5

METHOD: dict[str, dict] = {
    "ours":       dict(color=C["ours"],       linestyle="-",              marker="o", linewidth=_LW, markersize=_MS,     markerfacecolor=C["ours"],       markeredgewidth=0.5, markeredgecolor="white", zorder=5),
    "baseline_a": dict(color=C["baseline_a"], linestyle="--",             marker="s", linewidth=_LW, markersize=_MS-0.5, markerfacecolor=C["baseline_a"], markeredgewidth=0.5, markeredgecolor="white", zorder=4),
    "baseline_b": dict(color=C["baseline_b"], linestyle="-.",             marker="^", linewidth=_LW, markersize=_MS,     markerfacecolor=C["baseline_b"], markeredgewidth=0.5, markeredgecolor="white", zorder=3),
    "baseline_c": dict(color=C["baseline_c"], linestyle=":",              marker="D", linewidth=_LW, markersize=_MS-0.5, markerfacecolor=C["baseline_c"], markeredgewidth=0.5, markeredgecolor="white", zorder=3),
    "baseline_d": dict(color=C["baseline_d"], linestyle=(0,(3,1,1,1)),    marker="v", linewidth=_LW, markersize=_MS,     markerfacecolor=C["baseline_d"], markeredgewidth=0.5, markeredgecolor="white", zorder=2),
    "baseline_e": dict(color=C["baseline_e"], linestyle=(0,(5,2)),        marker="P", linewidth=_LW, markersize=_MS,     markerfacecolor=C["baseline_e"], markeredgewidth=0.5, markeredgecolor="white", zorder=2),
    "ablation":   dict(color=C["neutral"],    linestyle="--",             marker="x", linewidth=0.9, markersize=_MS,     markerfacecolor=C["neutral"],    markeredgewidth=1.0, markeredgecolor=C["neutral"],  zorder=1),
}

BAR: dict[str, dict] = {k: dict(color=v["color"], edgecolor="white", linewidth=0.4) for k, v in METHOD.items()}

# ---------------------------------------------------------------------------
# 3. TYPOGRAPHY  (calibrated for NeurIPS 10pt body at final print size)
# ---------------------------------------------------------------------------
TYPE = {
    "body":       8,   # axis labels, legend
    "small":      7,   # tick labels, minor annotations
    "tiny":       6,   # panel letters (a)(b)
    "title":      8,   # subplot title
    "annotation": 7,   # data labels on bars, callout text
}

# ---------------------------------------------------------------------------
# 4. LAYOUT CONSTANTS
# ---------------------------------------------------------------------------
FIG = {"w1": 3.25, "w2": 6.75, "w15": 5.0, "h1": 2.25, "h2": 2.0, "hbar": 1.8}
LAYOUT = {"hspace": 0.35, "wspace": 0.35}
ERRBAR = dict(fmt="none", capsize=2.0, capthick=0.6, elinewidth=0.6, alpha=0.8)

SPINE_WIDTH = 0.6; TICK_WIDTH = 0.5; TICK_LENGTH = 2.5; TICK_PAD = 2.0
GRID_ALPHA  = 0.25; GRID_LW = 0.5

# ---------------------------------------------------------------------------
# 5. APPLY STYLE
# ---------------------------------------------------------------------------
def apply_style(usetex: bool = False) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "text.usetex": usetex,
        "font.size":               TYPE["body"],
        "axes.titlesize":          TYPE["title"],
        "axes.labelsize":          TYPE["body"],
        "xtick.labelsize":         TYPE["small"],
        "ytick.labelsize":         TYPE["small"],
        "legend.fontsize":         TYPE["small"],
        "legend.title_fontsize":   TYPE["body"],
        "axes.spines.top":         False,
        "axes.spines.right":       False,
        "axes.linewidth":          SPINE_WIDTH,
        "axes.labelpad":           3.0,
        "axes.titlepad":           4.0,
        "axes.prop_cycle":         mpl.cycler(color=PALETTE),
        "xtick.major.width":       TICK_WIDTH,  "ytick.major.width":  TICK_WIDTH,
        "xtick.major.size":        TICK_LENGTH, "ytick.major.size":   TICK_LENGTH,
        "xtick.major.pad":         TICK_PAD,    "ytick.major.pad":    TICK_PAD,
        "xtick.direction":         "out",       "ytick.direction":    "out",
        "axes.grid":               True,
        "axes.grid.axis":          "y",
        "grid.alpha":              GRID_ALPHA,
        "grid.linewidth":          GRID_LW,
        "grid.linestyle":          "--",
        "grid.color":              "#AAAAAA",
        "lines.linewidth":         _LW,
        "lines.markersize":        _MS,
        "legend.frameon":          False,
        "legend.handlelength":     1.5,
        "legend.handletextpad":    0.4,
        "legend.labelspacing":     0.3,
        "legend.borderpad":        0.0,
        "legend.columnspacing":    1.0,
        "figure.constrained_layout.use": True,
        "figure.dpi":              150,
        "savefig.dpi":             300,
        "savefig.bbox":            "tight",
        "savefig.pad_inches":      0.02,
        "pdf.fonttype":            42,
        "ps.fonttype":             42,
    })

# ---------------------------------------------------------------------------
# 6. FIGURE FACTORY
# ---------------------------------------------------------------------------
def figure(ncols=1, nrows=1, width="single", height=None, **subplot_kw):
    """
    width: "single" (3.25in) | "double" (6.75in) | "1.5" (5.0in) | float inches
    height: auto-derived if None
    """
    apply_style()
    w = {"single": FIG["w1"], "double": FIG["w2"], "1.5": FIG["w15"]}.get(width)
    if w is None:                       # numeric width passed directly (e.g. width=4.0)
        w = float(width)
    if height is None:
        base_h = FIG["h2"] if w >= FIG["w2"] else FIG["h1"]
        h = base_h * nrows
    else:
        h = height
    return plt.subplots(nrows, ncols, figsize=(w, h), **subplot_kw)

# ---------------------------------------------------------------------------
# 7. HELPERS
# ---------------------------------------------------------------------------
def despine(ax, left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks([])

def label_subplot(ax, letter, x=-0.18, y=1.05):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=TYPE["small"], fontweight="bold", va="top", ha="left")

def annotate_best(ax, x, y, text, color=C["highlight"]):
    ax.annotate(text, xy=(x, y), xytext=(x, y * 1.08),
                fontsize=TYPE["annotation"], color=color, ha="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6))

def add_significance_bar(ax, x1, x2, y, p_text="*", color=C["dark"]):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.6, color=color)
    ax.text((x1+x2)/2, y+h, p_text, ha="center", va="bottom",
            fontsize=TYPE["annotation"], color=color)

# ---------------------------------------------------------------------------
# 8. SAVE
# ---------------------------------------------------------------------------
def save(fig, path, formats=("pdf", "png")):
    """Save figure as PDF + PNG. Pass path WITHOUT extension."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = p.with_suffix(f".{fmt}")
        fig.savefig(out)
        print(f"  saved → {out}")
