"""
R12 -- experiments asked for at the report review.

  ablation : where does the power of the final model come from? Drop one input (each of the 13 Spotify
             descriptors, the decade block, artist history) or keep one block alone, 5x5 CV, corrected
             t-test against the final model; plus permutation importance of every input on held-out folds.
  pca      : PCA of the 13 Spotify descriptors -- explained variance, loadings, how well each original
             descriptor and each component separates hits (univariate ROC-AUC), and the frozen RF on the
             first k components.
  all      : train on EVERYTHING we built (Spotify, decade, artist, FFT, chorus_hit/sections, title length,
             market trend, song title) on the 35,370 songs with a preview, then remove one block at a time
             and keep one block at a time.
  traj     : what happens in a song's first weeks on the Hot 100 -- average trajectories of songs that do
             and do not reach the top 10, hit rate by debut position and first moves, permutation
             importance of the trajectory inputs (k = 3).

    py repro/r12_report.py [ablation] [pca] [all] [traj]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from common import (ART, AUDIO, DEC_COLS, SEED, TitleOOF, corrected_ttest, cv_eval, load_kaggle,
                    make_pipeline, save, show)

FINAL = AUDIO + DEC_COLS + [ART]


def paired(folds, a, b):
    """Accuracy of config a minus config b, fold by fold, with the corrected t-test."""
    x = folds[folds.config == a].set_index("fold")
    r = folds[folds.config == b].set_index("fold")
    d = x.accuracy - r.accuracy
    return round(d.mean(), 4), int((d > 0).sum()), corrected_ttest(d.values, r.n_train.mean(), r.n_test.mean())[1]


# --------------------------------------------------------------------------- ablation
def run_ablation():
    df = load_kaggle()
    y = df["target"].values
    groups = {c: [c] for c in AUDIO}
    groups["decade (6)"] = DEC_COLS
    groups["artist history"] = [ART]
    configs = {"final (20)": make_pipeline(FINAL)}
    for g, cols in groups.items():
        configs[f"- {g}"] = make_pipeline([c for c in FINAL if c not in cols])
    configs["- all 13 Spotify"] = make_pipeline(DEC_COLS + [ART])
    configs["Spotify alone"] = make_pipeline(AUDIO)
    configs["decade alone"] = make_pipeline(DEC_COLS)
    configs["artist history alone"] = make_pipeline([ART])
    summ, folds = cv_eval(configs, df, y, label="ablation")
    show(summ)
    save(summ, "r12_ablation_cv_summary.csv")
    save(folds, "r12_ablation_cv_folds.csv")

    # permutation importance on the held-out part of the 5 folds of the first repeat
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    rng = np.random.default_rng(SEED)
    rows = []
    for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        m = clone(configs["final (20)"]).fit(df.iloc[tr], y[tr])
        Xte = df.iloc[te][FINAL].copy()
        base = accuracy_score(y[te], m.predict(Xte))
        for g, cols in groups.items():
            for r in range(5):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xte[cols].to_numpy()[perm]          # the block is shuffled as a whole
                rows.append({"fold": f, "input": g, "repeat": r,
                             "drop": base - accuracy_score(y[te], m.predict(Xp))})
        print(f"  [perm] fold {f + 1}/5", flush=True)
    pi = (pd.DataFrame(rows).groupby("input")["drop"].agg(["mean", "std"]).round(4)
          .sort_values("mean", ascending=False).reset_index())
    show(pi)
    save(pi, "r12_permutation_importance.csv")


# --------------------------------------------------------------------------- pca
def run_pca():
    df = load_kaggle()
    y = df["target"].values
    Z = StandardScaler().fit_transform(df[AUDIO].astype(float))
    pca = PCA(random_state=SEED).fit(Z)
    comp = pca.transform(Z)
    var = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(AUDIO))],
                        "explained": pca.explained_variance_ratio_.round(4),
                        "cumulative": np.cumsum(pca.explained_variance_ratio_).round(4)})
    var["auc"] = [round(max(a, 1 - a), 4) for a in (roc_auc_score(y, comp[:, i]) for i in range(len(AUDIO)))]
    load = pd.DataFrame(pca.components_.T, index=AUDIO, columns=var.component).round(3)
    auc = pd.DataFrame({"feature": AUDIO,
                        "auc": [round(max(a, 1 - a), 4) for a in (roc_auc_score(y, df[c]) for c in AUDIO)],
                        "direction": ["higher in hits" if roc_auc_score(y, df[c]) > .5 else "lower in hits"
                                      for c in AUDIO]}).sort_values("auc", ascending=False)
    show(var)
    show(load)
    show(auc)
    save(var, "r12_pca_variance.csv")
    save(load.reset_index().rename(columns={"index": "feature"}), "r12_pca_loadings.csv")
    save(auc, "r12_univariate_auc.csv")

    configs = {"Spotify (13 raw)": make_pipeline(AUDIO)}
    for k in [1, 2, 4, 6, 8, 10, 13]:
        configs[f"first {k} PCs"] = make_pipeline([], pca_cols=AUDIO, n_pca=k)
    configs["first 6 PCs + decade + artist"] = make_pipeline(DEC_COLS + [ART], pca_cols=AUDIO, n_pca=6)
    configs["final (20)"] = make_pipeline(FINAL)
    summ, folds = cv_eval(configs, df, y, label="pca")
    show(summ)
    save(summ, "r12_pca_cv_summary.csv")
    save(folds, "r12_pca_cv_folds.csv")


# --------------------------------------------------------------------------- everything together
def run_all():
    from r10_timeseries import TREND, add_trend
    from r8_future_work import load_fft
    df = add_trend(load_kaggle())
    df["title_words"] = df["track"].astype(str).str.split().str.len()
    data, fcols = load_fft(df)
    y = data["target"].values
    print(f"songs with a preview: {len(data):,} ({y.mean():.3f} hits)")
    blocks = {"Spotify (13)": AUDIO, "decade (6)": DEC_COLS, "artist history": [ART], "FFT (31)": fcols,
              "chorus_hit + sections": ["chorus_hit", "sections"], "title length": ["title_words"],
              "market trend (2)": TREND, "song title": "TITLE"}

    def P(names):
        cols = [c for b in names if blocks[b] != "TITLE" for c in blocks[b]]
        extra = [("title", TitleOOF(), ["track"])] if "song title" in names else None
        return make_pipeline(cols, extra=extra)

    allb = list(blocks)
    configs = {"final (20)": P(["Spotify (13)", "decade (6)", "artist history"]),
               "everything": P(allb),
               "everything except song title": P([b for b in allb if b != "song title"])}
    for b in allb:
        if b != "song title":
            configs[f"everything - {b}"] = P([x for x in allb if x != b])
        configs[f"{b} alone"] = P([b])
    summ, folds = cv_eval(configs, data, y, label="all")
    show(summ)
    save(summ, "r12_all_cv_summary.csv")
    save(folds, "r12_all_cv_folds.csv")
    rows = []
    for b in allb:
        name = f"everything - {b}" if b != "song title" else "everything except song title"
        d, w, p = paired(folds, name, "everything")
        rows.append({"block": b, "removed_from_everything": name, "delta": d, "wins": w, "p": p})
    d, w, p = paired(folds, "everything except song title", "final (20)")
    rows.append({"block": "(everything except title) vs final", "removed_from_everything": "", "delta": d,
                 "wins": w, "p": p})
    loo = pd.DataFrame(rows)
    show(loo)
    save(loo, "r12_all_leave_one_out.csv")


# --------------------------------------------------------------------------- trajectories
def run_traj():
    import r10_timeseries as r10
    from r4_external import rf200
    r10.KS = list(range(1, 9))                     # positions over the first 8 weeks, for the plots
    t = r10.add_dynamics(r10.build_trajectories())
    pos = [f"pos{i}" for i in r10.KS]
    rows = []
    for lab, g in t.groupby("big_top10"):
        P = g[pos].to_numpy(float)
        for i, c in enumerate(pos):
            on = P[:, i] <= 100
            rows.append({"top10": lab, "week": i + 1, "on_chart": round(on.mean(), 4),
                         "median_pos": float(np.median(P[:, i])), "q25": float(np.quantile(P[:, i], .25)),
                         "q75": float(np.quantile(P[:, i], .75)),
                         "mean_pos_on_chart": round(P[on, i].mean(), 2), "n": len(g)})
    weekly = pd.DataFrame(rows)
    show(weekly)
    save(weekly, "r12_traj_weekly.csv")

    # top-10 rate by debut position and by the move between week 1 and week 3
    t["debut_bin"] = pd.cut(t.pos1, [0, 40, 60, 80, 90, 100], labels=["1-40", "41-60", "61-80", "81-90", "91-100"])
    t["move3"] = pd.cut(t.pos1 - t.pos3, [-1000, -1, 9, 29, 1000],
                        labels=["fell / left", "+0-9", "+10-29", "+30 or more"])
    grid = (t.groupby(["debut_bin", "move3"], observed=True).big_top10.agg(["mean", "size"]).round(4)
            .reset_index())
    show(grid)
    save(grid, "r12_traj_grid.csv")

    # permutation importance of the k = 3 model (random forest, 75/25 split)
    ctx = ["year", "month", "prior_songs_log", "prior_top10_log"]
    feats = ["pos1", "pos2", "pos3", "best3", "onchart3", "slope3", "accel3"] + ctx
    y = t.big_top10.to_numpy()
    tr, te = train_test_split(np.arange(len(t)), test_size=.25, stratify=y, random_state=SEED)
    X = t[feats].to_numpy(float)
    m = rf200().fit(X[tr], y[tr])
    base = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    rng = np.random.default_rng(SEED)
    imp = []
    for j, c in enumerate(feats):
        drops = []
        for _ in range(5):
            Xp = X[te].copy()
            Xp[:, j] = Xp[rng.permutation(len(te)), j]
            drops.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
        imp.append({"input": c, "auc_drop": round(float(np.mean(drops)), 4), "std": round(float(np.std(drops)), 4)})
    imp = pd.DataFrame(imp).sort_values("auc_drop", ascending=False)
    print(f"held-out AUC {base:.4f}")
    show(imp)
    save(imp, "r12_traj_importance.csv")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["ablation", "pca", "all", "traj"]
    for name in todo:
        {"ablation": run_ablation, "pca": run_pca, "all": run_all, "traj": run_traj}[name]()
