"""
Figures of the paper, drawn only from paper_results/*.csv  ->  paper/fig_*.pdf

    py repro/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "paper_results")
OUT = os.path.join(ROOT, "paper")

BLUE = "#2a78d6"      # with artist history (categorical slot 1, validated)
GRAY = "#b4b2a9"      # without artist history (neutral reference)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
COL_W = 3.45          # IEEE column width, inches

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42,
})


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def fig_artist_curve():
    f = rd("r1_dataset_facts.csv").set_index("fact")["value"]
    labs = ["0", "1-2", "3-5", "6-10", "11+"]
    rate = [100 * float(f[f"hit_rate_prior_{b}"]) for b in labs]
    n = [int(float(f[f"n_prior_{b}"])) for b in labs]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(labs))
    ax.bar(x, rate, width=0.62, color=BLUE, edgecolor="white", linewidth=1.0)
    for xi, r in zip(x, rate):
        ax.text(xi, r + 2, f"{r:.0f}%", ha="center", va="bottom", color=INK, fontsize=7)
    share = 100 * float(f["hit_share"])
    ax.axhline(share, color=MUTED, lw=0.8, ls="--")      # explained in the caption
    ax.set_xticks(x, [f"{b}\n(n={ni:,})" for b, ni in zip(labs, n)])
    ax.set_xlabel("Hot 100 songs by the same artist in earlier decades")
    ax.set_ylabel("Songs that are hits (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_artist_curve.pdf"))
    plt.close(fig)


def fig_tilburg():
    raw = rd("r5_tilburg_seeds.csv")
    raw = raw[raw.model == "RF (their config)"]
    pub = {"00s": 0.84, "10s": 0.82}
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, dec in zip(axes, ["00s", "10s"]):
        d = raw[raw.decade == dec]
        for i, (feat, col, lab) in enumerate([("their inputs", GRAY, "their model"),
                                              ("+ artist", BLUE, "+ artist history")]):
            a = 100 * d[d.features == feat].accuracy.values
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(a)), a, s=9, color=col,
                       edgecolor="white", linewidth=0.4, zorder=3, label=lab)
            ax.plot([i - 0.26, i + 0.26], [a.mean()] * 2, color=INK, lw=1.2, zorder=4)
        ax.axhline(100 * pub[dec], color=MUTED, lw=0.8, ls="--", zorder=2)
        ax.text(1.5, 100 * pub[dec] - 0.35, "published", ha="right", va="top",
                color=INK2, fontsize=6.5)
        ax.set_xticks([0, 1], ["their\nmodel", "+ artist\nhistory"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"20{dec[:2]}s", color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_tilburg.pdf"))
    plt.close(fig)


def fig_ismir():
    order = [("their audio (592 Essentia)", "audio", GRAY),
             ("+ year", "+ year", GRAY),
             ("+ artist", "+ artist", BLUE),
             ("+ year + artist", "+ both", BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    for ax, (fn, title) in zip(axes, [("r4_ismir_A_random_cv_summary.csv", "A: random negatives"),
                                      ("r4_ismir_B_yearmatched_cv_summary.csv", "B: year-matched")]):
        s = rd(fn).set_index("config")
        x = np.arange(len(order))
        for xi, (cfg, lab, col) in zip(x, order):
            m = 100 * s.loc[cfg, "acc_mean"]
            lo, hi = 100 * s.loc[cfg, "acc_ci95_low"], 100 * s.loc[cfg, "acc_ci95_high"]
            ax.bar(xi, m - 60, bottom=60, width=0.66, color=col, edgecolor="white", linewidth=1.0)
            ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.8, capsize=2)
            ax.text(xi, hi + 0.4, f"{m:.1f}", ha="center", va="bottom", color=INK, fontsize=6.5)
        if title.startswith("A"):
            ax.axhline(75.04, color=MUTED, lw=0.8, ls="--")
            pass                                          # explained in the caption
        ax.set_xticks(x, [o[1] for o in order], fontsize=6.3)
        ax.set_ylim(60, 80)
        ax.set_yticks(range(60, 81, 5))
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("CV accuracy (%)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_ismir.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_subset():
    combos = rd("r3_combos_cv_summary.csv")
    fft = rd("r3_fft_cv_summary.csv")
    s = pd.concat([combos, fft[~fft.config.isin(combos.config)]]).set_index("config")
    names = {"audio (13)": "audio (13)", "+ decade": "+ decade", "+ rich": "+ rich librosa",
             "+ FFT (31)": "+ FFT", "+ FFT + decade": "+ FFT + decade",
             "+ decade x audio": "+ decade x audio", "+ rich + decade": "+ rich + decade",
             "+ artist": "+ artist", "+ decade + artist": "+ decade + artist",
             "+ artist + rich": "+ artist + rich", "+ artist + rich + FFT": "+ artist + rich + FFT",
             "+ artist + rich + decade": "+ artist + rich + decade",
             "all 5 blocks": "all 5 blocks", "+ artist + FFT": "+ artist + FFT",
             "+ artist + FFT + decade": "+ artist + FFT + decade"}
    s = s.loc[list(names)].sort_values("acc_mean")
    base = 100 * s.loc["audio (13)", "acc_mean"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    y = np.arange(len(s))
    for yi, (cfg, r) in zip(y, s.iterrows()):
        m, lo, hi = 100 * r.acc_mean, 100 * r.acc_ci95_low, 100 * r.acc_ci95_high
        col = BLUE if ("artist" in cfg or cfg == "all 5 blocks") else GRAY
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter(m, yi, s=18, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(hi + 0.25, yi, f"{m:.1f}", va="center", color=INK, fontsize=6.3)
    ax.axvline(base, color=MUTED, lw=0.8, ls="--", zorder=1)       # explained in the caption
    ax.set_yticks(y, [names[c] for c in s.index], fontsize=6.5)
    ax.set_xlabel("CV accuracy (%), audio subset")
    ax.set_xlim(75, 84.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_subset.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_artist_curve()
    fig_tilburg()
    fig_ismir()
    fig_subset()
    print("figures written to", OUT)
