"""
Step 2 — Protein Dataset Visualization
======================================
Reads protein_stats.csv and generates presentation-ready plots with TUM branding.

Usage:
    python visualize_stats.py \
        --input_csv ./protein_stats.csv \
        --output_dir ./plots
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  Design tokens — single source of truth for every visual decision
# ══════════════════════════════════════════════════════════════════════════════
TUM_BLUE   = "#0065BD"
TUM_DARK   = "#003359"
TUM_LIGHT  = "#5b9bd5"
TUM_ACCENT = "#e37222"
TUM_GREY   = "#a0a0a0"
BG         = "#f7f9fc"
TEXT       = "#1a1a2e"

# Ordered palette: first slot = TUM_BLUE (single-series), rest for multi-series
PALETTE = [TUM_BLUE, TUM_ACCENT, "#64b5b0", "#9b59b6", "#2ecc71"]

# Locked Meltome color mapping — same dataset gets the same colour in every plot.
# Order in legend & drawing is controlled separately (largest behind, smallest in front).
MELTOME_COLORS = {
    "Meltome_Mixed":     TUM_ACCENT,   # orange
    "Meltome_Human":     "#64b5b0",    # teal
    "Meltome_HumanCell": "#9b59b6",    # purple
}

def meltome_palette(datasets) -> dict:
    """
    Return {dataset: colour} mapping for the given datasets, falling back to
    PALETTE for any unknown dataset name.
    """
    fallback = [c for c in PALETTE[1:] if c not in MELTOME_COLORS.values()]
    out = {}
    fb = iter(fallback)
    for ds in datasets:
        out[ds] = MELTOME_COLORS.get(ds, next(fb, TUM_BLUE))
    return out

# Reference lines — used identically everywhere
LINE_MEAN   = dict(color=TUM_ACCENT, lw=1.5, ls="--")   # mean  → dashed orange
LINE_MEDIAN = dict(color=TUM_DARK,   lw=1.5, ls=":")    # median → dotted dark blue

# Single-source opacity for filled shapes (bars, histograms) — keeps every
# count-based plot at the same visual weight regardless of palette.
FILL_ALPHA = 0.50

# Legend — one fontsize, one frame style, everywhere
LEGEND_KW = dict(fontsize=9, framealpha=0.85, edgecolor="#d0d8e4")

# Bar value annotation
BAR_LABEL_KW = dict(ha="center", va="bottom", fontsize=8, fontweight="bold", color=TEXT)

# Boxplot fliers (seaborn ≥ 0.12 / < 0.13 compatible dict form)
FLIER_KW = dict(marker="o", markersize=3, alpha=0.35, markerfacecolor=TUM_GREY,
                markeredgewidth=0)

# Figure sizes
W_WIDE  = (11, 5)   # wide single-panel
W_SQ    = (9,  6)   # squarish single-panel
W_HALF  = (10, 5)   # default histogram / KDE panel

# ── Global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    # Backgrounds
    "figure.facecolor":     BG,
    "axes.facecolor":       BG,
    "savefig.facecolor":    BG,
    # Axes frame
    "axes.edgecolor":       "#d0d8e4",
    "axes.linewidth":       0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Labels & title
    "axes.labelcolor":      TEXT,
    "axes.labelsize":       11,
    "axes.titlesize":       14,
    "axes.titleweight":     "bold",
    "axes.titlecolor":      TUM_DARK,
    "axes.titlelocation":   "left",       # left-aligned titles — more editorial
    "axes.titlepad":        10,
    # Grid — horizontal only, subtle
    "axes.grid":            True,
    "grid.color":           "#dce6f0",
    "grid.linewidth":       0.6,
    "grid.linestyle":       "--",
    # Ticks
    "xtick.color":          TEXT,
    "ytick.color":          TEXT,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "xtick.major.pad":      4,
    "ytick.major.pad":      4,
    # Font
    "font.family":          "DejaVu Sans",
    "text.color":           TEXT,
    # Output
    "savefig.dpi":          200,
    "savefig.bbox":         "tight",
    "figure.constrained_layout.use": False,   # we call tight_layout() explicitly
})


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _watermark(fig: plt.Figure) -> None:
    """Italic dataset credit in the bottom-right corner."""
    fig.text(0.99, 0.01, "PP1 SoSe2026 · Protein Prediction Dataset",
             ha="right", va="bottom", fontsize=7, color=TUM_GREY, style="italic")


def save(fig: plt.Figure, out_dir: str, name: str) -> None:
    """Finalise layout, stamp watermark, write PNG, close figure."""
    # Enforce horizontal-only grid on every axes that hasn't opted out
    for ax in fig.axes:
        # Scatter plots set axis="both" explicitly; respect that.
        # All other axes: y-only grid, no vertical lines.
        if ax.get_xgridlines() and not getattr(ax, "_grid_both", False):
            ax.grid(axis="y")
            ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _watermark(fig)                         # after tight_layout so it never clips
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  {path}")


def _add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:,}", offset_frac: float = 0.01) -> None:
    """
    Print a value label above each bar.  `offset_frac` is a fraction of the
    tallest bar used as vertical padding so labels never overlap the bar top.
    """
    heights = [b.get_height() for b in bars if b.get_height() > 0]
    if not heights:
        return
    offset = max(heights) * offset_frac
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + offset, fmt.format(h),
                    **BAR_LABEL_KW)


def _vline_legend_handles(specs: list) -> list:
    """
    Build Line2D proxy artists for reference lines.
    specs: [(x_value, line_kw_dict, label_str), ...]
    """
    handles = []
    for _, kw, label in specs:
        handles.append(Line2D([0], [0], label=label, **kw))
    return handles


def _draw_vlines(ax: plt.Axes, specs: list) -> None:
    """
    Draw vertical reference lines.
    specs: [(x_value, line_kw_dict, label_str), ...]
    """
    for x, kw, _ in specs:
        ax.axvline(x, **kw)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot functions
# ══════════════════════════════════════════════════════════════════════════════

# ── 1a. DeepLoc — Sequence Length Distribution ────────────────────────────────
def plot_deeploc_length(df: pd.DataFrame, out_dir: str) -> None:
    sub = df[df["dataset"] == "DeepLoc"]["seq_length"].dropna()
    mu, med = sub.mean(), sub.median()
    n = len(sub)

    fig, ax = plt.subplots(figsize=W_HALF)
    sns.histplot(x=sub, color=TUM_BLUE, bins=50,
                 edgecolor="white", linewidth=0.5,
                 alpha=FILL_ALPHA, log_scale=True, ax=ax)

    specs = [
        (mu,  LINE_MEAN,   f"Mean   {int(mu):,}"),
        (med, LINE_MEDIAN, f"Median {int(med):,}"),
    ]
    _draw_vlines(ax, specs)
    ax.legend(handles=_vline_legend_handles(specs), **LEGEND_KW)

    ax.set_title(f"DeepLoc — Sequence Length Distribution  (n = {n:,})")
    ax.set_xlabel("Sequence Length (log scale)")
    ax.set_ylabel("Number of Proteins")
    save(fig, out_dir, "01a_deeploc_length.png")


# ── 1b. Meltome — Sequence Length Distribution ────────────────────────────────
def plot_meltome_length(df: pd.DataFrame, out_dir: str) -> None:
    sub = df[df["dataset"].str.contains("Meltome", na=False)]
    # Smallest behind, largest in front so the dominant series isn't blocked,
    # and the smaller series remain visible underneath through transparency
    datasets = (sub.groupby("dataset").size()
                .sort_values(ascending=True).index.tolist())
    pal = meltome_palette(datasets)

    fig, ax = plt.subplots(figsize=W_HALF)
    sns.histplot(data=sub, x="seq_length", hue="dataset",
                 hue_order=datasets,
                 bins=50, palette=pal,
                 multiple="layer",          # overlapping bars (not stacked)
                 edgecolor="white", linewidth=0.4,
                 alpha=FILL_ALPHA,
                 log_scale=True, ax=ax)

    median_handles = []
    for ds in datasets:
        med = sub[sub["dataset"] == ds]["seq_length"].median()
        n   = (sub["dataset"] == ds).sum()
        ax.axvline(med, color=pal[ds], lw=1.5, ls=":")
        median_handles.append(
            Line2D([0], [0], color=pal[ds], lw=1.5, ls=":",
                   label=f"{ds} median {int(med):,}  (n = {n:,})")
        )

    hue_handles = [
        Line2D([0], [0], color=pal[ds], lw=6, alpha=0.5, label=ds)
        for ds in datasets
    ]
    ax.get_legend().remove()
    ax.legend(handles=hue_handles + median_handles, ncol=2, **LEGEND_KW)

    ax.set_title("Meltome — Sequence Length Distribution")
    ax.set_xlabel("Sequence Length (log scale)")
    ax.set_ylabel("Number of Proteins")
    save(fig, out_dir, "01b_meltome_length.png")


# ── 2. DeepLoc — Subcellular Localisation Label Distribution ─────────────────
def plot_deeploc_labels(df: pd.DataFrame, out_dir: str) -> None:
    sub    = df[df["dataset"] == "DeepLoc"]
    counts = sub["primary_label"].value_counts()
    n      = len(sub)

    fig, ax = plt.subplots(figsize=W_WIDE)
    bars = ax.bar(range(len(counts)), counts.values,
                  color=TUM_BLUE, alpha=FILL_ALPHA,
                  edgecolor="white", linewidth=0.6)
    _add_bar_labels(ax, bars, fmt="{:,}")

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_title(f"DeepLoc — Subcellular Localisation — Primary Label  (n = {n:,})")
    ax.set_ylabel("Number of Proteins")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save(fig, out_dir, "02_deeploc_labels.png")


# ── 3. Meltome — Thermostability (Tm) Distribution ───────────────────────────
def plot_meltome_targets(df: pd.DataFrame, out_dir: str) -> None:
    sub      = df[df["task"] == "regression"]
    # Smallest behind, largest in front so dominant series isn't blocked
    datasets = (sub.groupby("dataset").size()
                .sort_values(ascending=True).index.tolist())
    pal      = meltome_palette(datasets)

    fig, ax = plt.subplots(figsize=W_HALF)
    sns.kdeplot(data=sub, x="target", hue="dataset",
                hue_order=datasets,
                fill=True, alpha=0.30, linewidth=1.8,
                palette=pal, ax=ax)

    median_handles = []
    for ds in datasets:
        med = sub[sub["dataset"] == ds]["target"].median()
        n   = (sub["dataset"] == ds).sum()
        ax.axvline(med, color=pal[ds], lw=1.5, ls=":")
        median_handles.append(
            Line2D([0], [0], color=pal[ds], lw=1.5, ls=":",
                   label=f"{ds} median {med:.1f} °C  (n = {n:,})")
        )

    hue_handles = [
        Line2D([0], [0], color=pal[ds], lw=6, alpha=0.5, label=ds)
        for ds in datasets
    ]
    ax.get_legend().remove()
    ax.legend(handles=hue_handles + median_handles, ncol=2, **LEGEND_KW)

    ax.set_title("Meltome — Thermostability (Tm) Distribution")
    ax.set_xlabel("Melting Temperature (°C)")
    ax.set_ylabel("Density")
    save(fig, out_dir, "03_meltome_targets.png")


# ── 4. Amino Acid Group Composition by Dataset ────────────────────────────────
def plot_composition(df: pd.DataFrame, out_dir: str) -> None:
    groups = ["pct_hydrophobic", "pct_polar", "pct_charged_pos",
              "pct_charged_neg", "pct_special"]
    label_map = {
        "pct_hydrophobic":  "Hydrophobic",
        "pct_polar":        "Polar",
        "pct_charged_pos":  "Charged (+)",
        "pct_charged_neg":  "Charged (−)",
        "pct_special":      "Special",
    }

    means  = df.groupby("dataset")[groups].mean().reset_index()
    melted = means.melt(id_vars="dataset", var_name="group", value_name="pct")
    melted["group"] = melted["group"].map(label_map)

    order = (melted.groupby("group")["pct"].mean()
             .sort_values(ascending=False).index.tolist())
    datasets = sorted(melted["dataset"].unique())
    # DeepLoc → TUM_BLUE, Meltome_* → locked colors from MELTOME_COLORS
    pal = {ds: (TUM_BLUE if ds == "DeepLoc" else MELTOME_COLORS.get(ds, TUM_BLUE))
           for ds in datasets}

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = sns.barplot(data=melted, x="group", y="pct", hue="dataset",
                     order=order, palette=pal,
                     alpha=FILL_ALPHA,
                     edgecolor="white", linewidth=0.6, ax=ax)

    # Value labels — iterate containers (one per hue group) for reliable positioning
    for container in bp.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.12, f"{h:.1f}",
                        **BAR_LABEL_KW)

    ax.set_title("Mean Amino Acid Group Composition by Dataset")
    ax.set_ylabel("Mean Percentage (%)")
    ax.set_xlabel("")
    ax.legend(title="Dataset", **LEGEND_KW)
    save(fig, out_dir, "04_aa_composition.png")


# ── 5. DeepLoc — Sequence Length vs Number of Assigned Locations ──────────────
def plot_length_vs_labels(df: pd.DataFrame, out_dir: str) -> None:
    sub = df[df["dataset"] == "DeepLoc"]
    n   = len(sub)

    fig, ax = plt.subplots(figsize=W_WIDE)

    # Boxplot — use medianprops / boxprops for consistent TUM colours
    sns.boxplot(data=sub, x="num_labels", y="seq_length",
                color=TUM_LIGHT, width=0.5,
                boxprops=dict(edgecolor=TUM_BLUE, linewidth=1, alpha=FILL_ALPHA),
                medianprops=dict(color=TUM_DARK, lw=2),
                whiskerprops=dict(color=TUM_BLUE, linewidth=1),
                capprops=dict(color=TUM_BLUE, linewidth=1),
                flierprops=FLIER_KW,
                linewidth=1.2, ax=ax)
    sns.stripplot(data=sub, x="num_labels", y="seq_length",
                  color=TUM_DARK, alpha=0.12, jitter=True, size=3, ax=ax)

    ax.set_yscale("log")
    ax.set_title(f"DeepLoc — Sequence Length vs Assigned Locations  (n = {n:,})")
    ax.set_xlabel("Number of Assigned Locations")
    ax.set_ylabel("Sequence Length (log scale)")
    save(fig, out_dir, "05_length_vs_labels.png")


# ── 6. Meltome — Sequence Length vs Melting Temperature ──────────────────────
def plot_length_vs_tm(df: pd.DataFrame, out_dir: str) -> None:
    sub      = df[df["task"] == "regression"][["seq_length", "target", "dataset"]].dropna()
    # Largest behind, smallest in front (also makes the legend order intuitive)
    datasets = (sub.groupby("dataset").size()
                .sort_values(ascending=False).index.tolist())
    pal      = meltome_palette(datasets)

    fig, ax = plt.subplots(figsize=W_SQ)

    for ds in datasets:
        s = sub[sub["dataset"] == ds]
        ax.scatter(s["seq_length"], s["target"],
                   alpha=0.22, s=14, color=pal[ds],
                   edgecolors="none", label=ds, rasterized=True)

    # Pooled OLS trend
    z  = np.polyfit(sub["seq_length"], sub["target"], 1)
    xs = np.linspace(sub["seq_length"].min(), sub["seq_length"].max(), 400)
    trend_line, = ax.plot(xs, np.poly1d(z)(xs),
                          color=TUM_DARK, lw=2, ls="--", label="Trend (pooled)")

    ax.set_xscale("log")
    ax.set_title("Meltome — Sequence Length vs Melting Temperature")
    ax.set_xlabel("Sequence Length (log scale)")
    ax.set_ylabel("Melting Temperature (°C)")
    # Scatter benefits from full grid; mark so save() preserves it
    ax._grid_both = True  # type: ignore[attr-defined]
    ax.grid(axis="both")
    ax.legend(**LEGEND_KW)
    save(fig, out_dir, "06_length_vs_tm.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Protein stats visualisation")
    parser.add_argument("--input_csv",  required=True,
                        help="Path to protein_stats.csv (output of process_stats.py)")
    parser.add_argument("--output_dir", default="protein_plots",
                        help="Destination folder for PNG outputs (created if absent)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading stats CSV …")
    df = pd.read_csv(args.input_csv)
    print(f"  {len(df):,} sequences loaded.\n")

    print("Generating plots …")
    plot_deeploc_length(df, args.output_dir)
    plot_meltome_length(df, args.output_dir)
    plot_deeploc_labels(df, args.output_dir)
    plot_meltome_targets(df, args.output_dir)
    plot_composition(df, args.output_dir)
    plot_length_vs_labels(df, args.output_dir)
    plot_length_vs_tm(df, args.output_dir)

    print(f"\nDone. All plots saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()