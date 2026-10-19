"""
feature_experiments.py  —  Phase 1/2: engineered features, one by one.
======================================================================
Builds two new features on top of the 13 audio features and measures each,
separately and combined, through the Phase 0 harness:

    Feature A  —  decade      (which era the track is from)
    Feature B  —  artist_prior_hits   (how many Billboard Hot 100 songs the
                                       artist had BEFORE this track's decade)

For every feature set it reports, via phase0_harness.evaluate_feature:
    delta accuracy vs the frozen baseline, McNemar p-value, LASSO survival,
    redundancy (corr/VIF) and the per-decade breakdown — then draws the graphs.

RUN
---
    put this next to phase0_harness.py, then:
        pip install numpy pandas scikit-learn scipy matplotlib
        py feature_experiments.py
    outputs:  results_summary.png , artist_signal.png , results.csv

Everything (Spotify data + Billboard history) is downloaded automatically the
first time.  Nothing to configure.
"""

import os, re, glob, unicodedata, urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from phase0_harness import evaluate_feature, make_baseline_rf, BASE_FEATURES_A

SEED = 42
DEC_ORDER = ["60s", "70s", "80s", "90s", "00s", "10s"]
DEC_START = {"60s": 1960, "70s": 1970, "80s": 1980, "90s": 1990, "00s": 2000, "10s": 2010}
AUDIO = BASE_FEATURES_A

SPOTIFY_DIR = "The-Spotify-Hit-Predictor-Dataset"
SPOTIFY_URL = ("https://raw.githubusercontent.com/fortyTwo102/"
               "The-Spotify-Hit-Predictor-Dataset/master/dataset-of-{d}.csv")
BILLBOARD_CSV = "hot100.csv"
BILLBOARD_URL = ("https://raw.githubusercontent.com/utdata/rwd-billboard-data/"
                 "main/data-out/hot-100-current.csv")

GREEN, BLUE, GRAY, AMBER = "#1DB954", "#2a78d6", "#B4B2A9", "#C97A10"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_data():
    os.makedirs(SPOTIFY_DIR, exist_ok=True)
    for d in DEC_ORDER:
        p = os.path.join(SPOTIFY_DIR, f"dataset-of-{d}.csv")
        if not os.path.exists(p):
            print(f"downloading dataset-of-{d}.csv ...")
            urllib.request.urlretrieve(SPOTIFY_URL.format(d=d), p)
    if not os.path.exists(BILLBOARD_CSV):
        print("downloading Billboard Hot 100 history (~19 MB) ...")
        urllib.request.urlretrieve(BILLBOARD_URL, BILLBOARD_CSV)


_SEP = re.compile(r"\s+(?:featuring|feat\.?|ft\.?|with|&|x|\+|/|,|vs\.?|and)\s+", re.I)
def normalize_artist(name):
    """Lowercase, strip accents/punctuation, keep the primary (first) artist."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    s = _SEP.split(s)[0]
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_artist_history():
    """normalized artist -> {decade_start_year: set(distinct song titles)}."""
    bb = pd.read_csv(BILLBOARD_CSV, usecols=["chart_week", "title", "performer"])
    bb["year"] = pd.to_datetime(bb["chart_week"]).dt.year
    bb["dstart"] = (bb["year"] // 10) * 10
    bb["na"] = bb["performer"].map(normalize_artist)
    hist = defaultdict(lambda: defaultdict(set))
    for na, ds, ti in zip(bb["na"], bb["dstart"], bb["title"]):
        if na:
            hist[na][ds].add(str(ti).lower().strip())
    return hist


def load_spotify_with_features():
    """Rebuild the exact prep2 dataframe (+ decade) and attach the new features."""
    hist = build_artist_history()

    def prior_hits(na, track_dstart):
        if na not in hist:
            return 0
        return sum(len(v) for d, v in hist[na].items() if d < track_dstart)

    frames = []
    for f in sorted(glob.glob(f"{SPOTIFY_DIR}/dataset-of-*.csv")):   # same order as prep2
        dec = re.search(r"dataset-of-(\w+)\.csv", os.path.basename(f)).group(1)
        t = pd.read_csv(f); t["decade"] = dec; frames.append(t)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["uri"], keep="first").reset_index(drop=True))

    df["na"] = df["artist"].map(normalize_artist)
    df["dstart"] = df["decade"].map(DEC_START)
    df["artist_prior_hits"] = [prior_hits(na, ds) for na, ds in zip(df["na"], df["dstart"])]
    df["artist_prior_hits_log"] = np.log1p(df["artist_prior_hits"])
    return df


def make_split(df):
    y = df["target"].values
    idx = np.arange(len(df))
    sel, _ = train_test_split(idx, train_size=24000, stratify=y, random_state=SEED)
    tr, tmp = train_test_split(sel, train_size=20000, stratify=y[sel], random_state=SEED)
    val, te = train_test_split(tmp, train_size=2000, stratify=y[tmp], random_state=SEED)
    base = {
        "X_tr": df[AUDIO].values[tr], "X_val": df[AUDIO].values[val], "X_te": df[AUDIO].values[te],
        "y_tr": y[tr], "y_val": y[val], "y_te": y[te], "feature_names": AUDIO,
    }
    return base, tr, val, te


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def decade_onehot(df, ix):
    return np.column_stack([(df["decade"].values[ix] == d).astype(float) for d in DEC_ORDER])


def run_all(df, base, tr, val, te):
    dec_te = df["decade"].values[te]
    col = lambda name, ix: df[name].values[ix].reshape(-1, 1)

    experiments = {
        "A: decade": (
            decade_onehot(df, tr), decade_onehot(df, val), decade_onehot(df, te),
            [f"is_{d}" for d in DEC_ORDER]),
        "B: artist history": (
            col("artist_prior_hits_log", tr), col("artist_prior_hits_log", val),
            col("artist_prior_hits_log", te), ["artist_prior_hits_log"]),
        "A+B: decade + artist": (
            np.column_stack([decade_onehot(df, tr), col("artist_prior_hits_log", tr)]),
            np.column_stack([decade_onehot(df, val), col("artist_prior_hits_log", val)]),
            np.column_stack([decade_onehot(df, te), col("artist_prior_hits_log", te)]),
            [f"is_{d}" for d in DEC_ORDER] + ["artist_prior_hits_log"]),
    }

    results = {}
    for name, (xt, xv, xe, feats) in experiments.items():
        res = evaluate_feature(base, xt, xv, xe, feature_names=feats, groups_te=dec_te)
        results[name] = res
        p = res["mcnemar"]["pvalue"]
        sig = "significant" if p < 0.05 else "n.s."
        print(f"{name:24s}  acc {res['acc_base']:.3f} -> {res['acc_aug']:.3f} "
              f"(+{res['delta_acc']:.3f})   McNemar p={p:.2g} [{sig}]")
    return results


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_summary(results, base_acc, path="results_summary.png"):
    order = ["A: decade", "B: artist history", "A+B: decade + artist"]
    accs = [base_acc] + [results[k]["acc_aug"] for k in order]
    labels = ["Baseline\n(13 audio)", "+ decade", "+ artist", "+ both"]
    colors = [GRAY, BLUE, AMBER, GREEN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # -- left: accuracy climb --
    bars = ax1.bar(labels, accs, color=colors, width=0.62)
    ax1.axhline(0.877, ls="--", lw=1, color="#888", zorder=0)
    ax1.text(3.4, 0.879, "paper 0.877", color="#888", fontsize=8, ha="right")
    ax1.set_ylim(0.74, 0.90)
    ax1.set_ylabel("test accuracy")
    ax1.set_title("Accuracy climb, one feature at a time", fontsize=11, weight="bold")
    for b, a in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width() / 2, a + 0.002, f"{a:.3f}",
                 ha="center", va="bottom", fontsize=9, weight="bold")
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # -- right: per-decade delta for each feature set --
    sub = {k: results[k]["subgroup"].set_index("group")["delta"] for k in order}
    decs = DEC_ORDER
    x = np.arange(len(decs)); w = 0.26
    ax2.bar(x - w, [sub["A: decade"].get(d, 0) * 100 for d in decs], w, label="decade", color=BLUE)
    ax2.bar(x, [sub["B: artist history"].get(d, 0) * 100 for d in decs], w, label="artist", color=AMBER)
    ax2.bar(x + w, [sub["A+B: decade + artist"].get(d, 0) * 100 for d in decs], w, label="both", color=GREEN)
    ax2.axhline(0, lw=0.8, color="#ccc")
    ax2.set_xticks(x); ax2.set_xticklabels(decs)
    ax2.set_ylabel("accuracy gain (points)")
    ax2.set_title("Where each feature helps, by decade", fontsize=11, weight="bold")
    ax2.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved {path}")


def plot_artist_signal(df, path="artist_signal.png"):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    hit = df.loc[df.target == 1, "artist_prior_hits"].clip(upper=20)
    flop = df.loc[df.target == 0, "artist_prior_hits"].clip(upper=20)
    bins = np.arange(0, 21, 1)
    ax.hist(flop, bins=bins, alpha=0.75, label=f"flops (mean {df.loc[df.target==0,'artist_prior_hits'].mean():.2f})",
            color=GRAY, density=True)
    ax.hist(hit, bins=bins, alpha=0.75, label=f"hits (mean {df.loc[df.target==1,'artist_prior_hits'].mean():.2f})",
            color=GREEN, density=True)
    ax.set_xlabel("artist's prior Billboard Hot 100 songs (capped at 20)")
    ax.set_ylabel("density")
    ax.set_title("Why artist history works: hits have an established artist", fontsize=11, weight="bold")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved {path}")


def save_table(results, base_acc, path="results.csv"):
    rows = [{"feature_set": "baseline", "test_acc": round(base_acc, 4),
             "delta": 0.0, "mcnemar_p": None, "significant": None}]
    for k, r in results.items():
        rows.append({"feature_set": k, "test_acc": r["acc_aug"], "delta": r["delta_acc"],
                     "mcnemar_p": r["mcnemar"]["pvalue"],
                     "significant": r["mcnemar"]["pvalue"] < 0.05})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
def main():
    download_data()
    df = load_spotify_with_features()
    match = df["na"].isin(build_artist_history()).mean()
    print(f"\nartist match rate vs Billboard: {match:.0%}  |  "
          f"mean prior hits  hits={df.loc[df.target==1,'artist_prior_hits'].mean():.2f} "
          f"flops={df.loc[df.target==0,'artist_prior_hits'].mean():.2f}\n")

    base, tr, val, te = make_split(df)
    base_acc = round(float(np.mean(
        make_baseline_rf().fit(base["X_tr"], base["y_tr"]).predict(base["X_te"]) == base["y_te"])), 4)
    print(f"frozen baseline test acc = {base_acc}\n")

    results = run_all(df, base, tr, val, te)
    print()
    plot_summary(results, base_acc)
    plot_artist_signal(df)
    save_table(results, base_acc)
    print("\ndone.")


if __name__ == "__main__":
    main()
