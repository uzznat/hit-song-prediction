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


TEXT_W = 7.16         # IEEE two-column text width, inches


def fig_traj():
    w = rd("r12_traj_weekly.csv")
    grid = rd("r12_traj_grid.csv")
    imp = rd("r12_traj_importance.csv")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.15), gridspec_kw={"width_ratios": [1.15, 1.05, 1]})
    ax = axes[0]
    for lab, col, name in [(1, BLUE, "reaches the top 10"), (0, GRAY, "does not")]:
        g = w[w.top10 == lab]
        ax.fill_between(g.week, g.q25, g.q75, color=col, alpha=0.22, lw=0)
        ax.plot(g.week, g.median_pos, color=col, lw=1.6, marker="o", ms=3, label=name)
    ax.axvspan(0.6, 3.4, color="#f3f2ee", zorder=0)
    ax.text(2, 4, "the 3 weeks\nthe model sees", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.set_ylim(102, 0)
    ax.set_yticks([1, 10, 25, 50, 75, 100])
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Week on the chart")
    ax.set_ylabel("Chart position (median, IQR)")
    ax.text(7.9, 22, "reaches the\ntop 10", ha="right", va="top", fontsize=6.5, color=BLUE)
    ax.text(5, 66, "does not", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title("(a) Median trajectory of each group", color=INK)

    ax = axes[1]
    rows = ["1-40", "41-60", "61-80", "81-90", "91-100"]
    cols = ["fell / left", "+0-9", "+10-29", "+30 or more"]
    M = np.full((len(rows), len(cols)), np.nan)
    N = np.zeros_like(M)
    for _, r in grid.iterrows():
        M[rows.index(r.debut_bin), cols.index(r.move3)] = 100 * r["mean"]
        N[rows.index(r.debut_bin), cols.index(r.move3)] = r["size"]
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=6.3,
                        color="white" if M[i, j] > 55 else INK)
    ax.set_xticks(range(len(cols)), ["fell", "+0-9", "+10-29", "+30\nor more"], fontsize=6.3)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Places climbed from week 1 to week 3")
    ax.set_ylabel("Debut position")
    ax.grid(False)
    ax.set_title("(b) Share that reaches the top 10", color=INK)

    ax = axes[2]
    names = {"pos3": "position, week 3", "slope3": "climb per week", "best3": "best position",
             "year": "year", "prior_songs_log": "artist's earlier songs", "pos2": "position, week 2",
             "pos1": "debut position", "prior_top10_log": "artist's earlier top 10s",
             "onchart3": "weeks on chart", "accel3": "acceleration", "month": "month"}
    imp = imp.sort_values("auc_drop")
    y = np.arange(len(imp))
    ctx = {"year", "month", "prior_songs_log", "prior_top10_log"}
    ax.barh(y, imp.auc_drop, color=[GRAY if c in ctx else BLUE for c in imp.input], height=0.66)
    ax.set_yticks(y, [names[c] for c in imp.input], fontsize=6.3)
    ax.set_xlabel("AUC lost when shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("(c) What the model uses", color=INK)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_traj.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_lyrics():
    s = rd("r11_lyrics_ismir_summary.csv")
    g = rd("r11_lyrics_ismir_grouped_summary.csv")
    words = rd("r11_lyrics_words.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    sets = [("A_random", "A: random\nnegatives"), ("B_yearmatched", "B: year-\nmatched"),
            ("B_yearmatched_english", "B, English\nonly"), ("B_yearmatched_english_grouped", "B, English,\nunseen artists")]
    allr = pd.concat([s, g]).set_index(["set", "config"])
    x = np.arange(len(sets))
    for off, cfg, col, lab in [(-0.19, "final: audio + year + artist", GRAY, "final model (audio + year + artist)"),
                               (0.19, "final + lexical + bag of words", BLUE, "+ lyrics")]:
        v = [100 * allr.loc[(k, cfg), "acc"] for k, _ in sets]
        ax.bar(x + off, [vi - 60 for vi in v], bottom=60, width=0.36, color=col, edgecolor="white", label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.set_xticks(x, [l for _, l in sets], fontsize=6.5)
    ax.set_ylim(60, 81)
    ax.set_ylabel("CV accuracy (%)")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Lyrics on ISMIR 2019, songs with lyrics", color=INK)
    ax = axes[1]
    w = words.sort_values("coef")
    w = pd.concat([w.head(12), w.tail(12)])
    y = np.arange(len(w))
    ax.barh(y, w.coef, color=[BLUE if c > 0 else GRAY for c in w.coef], height=0.7)
    ax.set_yticks(y, w.word, fontsize=6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("weight (left: non-hit, right: hit)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Words the lyrics model uses most", color=INK)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(os.path.join(OUT, "fig_lyrics.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


NICE = {"danceability": "danceability", "energy": "energy", "key": "key", "loudness": "loudness", "mode": "mode",
        "speechiness": "speechiness", "acousticness": "acousticness", "instrumentalness": "instrumentalness",
        "liveness": "liveness", "valence": "valence", "tempo": "tempo", "duration_ms": "duration",
        "time_signature": "time signature"}


def fig_pca():
    var = rd("r12_pca_variance.csv")
    load = rd("r12_pca_loadings.csv").set_index("feature")
    cv = rd("r12_pca_cv_summary.csv").set_index("config")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.3), gridspec_kw={"width_ratios": [1.1, 0.9, 1]})
    ax = axes[0]
    x = np.arange(1, len(var) + 1)
    ax.bar(x, 100 * var.explained, color=GRAY, width=0.7, label="variance explained")
    ax.plot(x, 100 * var.cumulative, color=INK2, lw=1, marker="o", ms=2.5, label="cumulative")
    ax.set_ylabel("Variance (%)")
    ax.set_xticks(x[::2], [str(i) for i in x[::2]], fontsize=6)
    ax.set_xlabel("Principal component")
    ax2 = ax.twinx()
    ax2.plot(x, var.auc, color=BLUE, lw=0, marker="D", ms=3.5, label="ROC-AUC for hits")
    ax2.axhline(0.5, color=BLUE, lw=0.6, ls=":")
    ax2.set_ylim(0.45, 0.75)
    ax2.set_ylabel("AUC of the component alone", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.12, 1.0), frameon=False, fontsize=6)
    ax.set_title("(a) Variance vs. separation", color=INK)

    ax = axes[1]
    L = load[["PC1", "PC2", "PC3"]]
    ax.imshow(L.to_numpy(), cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    for i in range(len(L)):
        for j in range(3):
            v = L.iloc[i, j]
            if abs(v) >= 0.25:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if abs(v) > 0.45 else INK)
    ax.set_xticks(range(3), ["PC1\nintensity", "PC2\nmood", "PC3\nmode/key"], fontsize=6.3)
    ax.set_yticks(range(len(L)), [NICE[f] for f in L.index], fontsize=6.3)
    ax.grid(False)
    ax.set_title("(b) Loadings of the first components", color=INK)

    ax = axes[2]
    ks = [1, 2, 4, 6, 8, 10, 13]
    acc = [100 * cv.loc[f"first {k} PCs", "acc_mean"] for k in ks]
    ax.plot(ks, acc, color=GRAY, marker="o", ms=3, lw=1.4, label="first $k$ components")
    ax.axhline(100 * cv.loc["Spotify (13 raw)", "acc_mean"], color=INK2, lw=0.9, ls="--")
    ax.text(1, 100 * cv.loc["Spotify (13 raw)", "acc_mean"] + 0.6, "13 raw descriptors", fontsize=6.3, color=INK2)
    ax.axhline(100 * cv.loc["final (20)", "acc_mean"], color=BLUE, lw=0.9, ls="--")
    ax.text(1, 100 * cv.loc["final (20)", "acc_mean"] + 0.6, "final model (raw + decade + artist)", fontsize=6.3,
            color=BLUE)
    ax.set_xticks(ks)
    ax.set_xlabel("Components kept ($k$)")
    ax.set_ylabel("CV accuracy (%)")
    ax.set_ylim(55, 86)
    ax.set_title("(c) RF on the first $k$ components", color=INK)
    fig.tight_layout(pad=0.3, w_pad=0.9)
    fig.savefig(os.path.join(OUT, "fig_pca.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_ablation():
    s = rd("r12_ablation_cv_summary.csv").set_index("config")
    pi = rd("r12_permutation_importance.csv")
    names = dict(NICE, **{"decade (6)": "decade (6 indicators)", "artist history": "artist history"})
    drops = [(k[2:], -100 * r.delta_vs_ref, r.corrected_p) for k, r in s.iterrows()
             if k.startswith("- ") and k != "- all 13 Spotify"]
    drops.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))
    ax = axes[0]
    y = np.arange(len(drops))
    cols = [BLUE if p < 0.05 / 400 else ("#8fb5e8" if p < 0.05 else GRAY) for _, _, p in drops]
    ax.barh(y, [d for _, d, _ in drops], color=cols, height=0.68)
    for yi, (_, d, p) in zip(y, drops):
        ax.text(max(d, 0) + 0.06, yi, f"{-d:+.1f}".replace("-", "−") if abs(d) >= 0.05 else "0.0",
                va="center", fontsize=6, color=INK)
    ax.set_yticks(y, [names[n] for n, _, _ in drops], fontsize=6.3)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Accuracy lost when the input is removed (points)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) Remove one input (final model 83.2 %)", color=INK)
    ax = axes[1]
    pi = pi.sort_values("mean")
    y = np.arange(len(pi))
    ax.barh(y, 100 * pi["mean"], xerr=100 * pi["std"], color=[BLUE if i in ("artist history", "decade (6)")
                                                             else GRAY for i in pi.input],
            height=0.68, error_kw={"lw": 0.6, "ecolor": INK2})
    ax.set_yticks(y, [names[n] for n in pi.input], fontsize=6.3)
    ax.set_xlabel("Accuracy lost when the input is shuffled (points)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Shuffle one input (permutation importance)", color=INK)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_decade():
    d = rd("r13_decade_medians.csv")
    decs = ["60s", "70s", "80s", "90s", "00s", "10s"]
    labs = ["1960s", "70s", "80s", "90s", "2000s", "10s"]
    panels = [("loudness", "Loudness (dB)", 1), ("acousticness", "Acousticness", 1),
              ("duration_ms", "Duration (min)", 1 / 60000)]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 1.85))
    x = np.arange(len(decs))
    for ax, (c, name, k) in zip(axes, panels):
        g = d[d.descriptor == c].set_index("decade").loc[decs]
        ax.plot(x, k * g.hits, color=BLUE, marker="o", ms=3.5, lw=1.6, label="hits")
        ax.plot(x, k * g.non_hits, color=GRAY, marker="o", ms=3.5, lw=1.6, label="non-hits")
        ax.set_xticks(x, labs, fontsize=6.5)
        ax.set_title(name + " (median)", color=INK)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig_decade.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["artist_curve", "tilburg", "ismir", "subset", "traj"]
    for name in todo:
        globals()[f"fig_{name}"]()
    print("figures written to", OUT)
