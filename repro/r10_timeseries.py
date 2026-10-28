"""
R10 -- two uses of time series for hit prediction.

  trajectory : AFTER release. From a song's first k weeks on the Hot 100 (positions, slope, momentum),
               predict whether it becomes a BIG hit (peak in the top 10; >= 20 weeks on the chart).
               Every song here already entered the chart, so this is "big hit vs small hit".
  trend      : BEFORE release. Market-trend features for the Kaggle benchmark: how far a song's audio is
               from the hits of the previous decade, and whether it moves in the direction of the trend.
               Same frozen RF and 5x5 CV as R1.

    py repro/r10_timeseries.py [trajectory] [trend]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline as sk_pipe
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from common import (ART, AUDIO, DATA, DEC_COLS, DEC_ORDER, SEED, corrected_ttest, cv_eval,
                    load_kaggle, make_pipeline, save, show)
from feature_experiments import normalize_artist
from r4_external import rf200

KS = [1, 2, 3, 4]


# --------------------------------------------------------------------------- trajectory
def build_trajectories():
    d = pd.read_csv(f"{DATA}/hot100.csv", parse_dates=["chart_week"])
    d["song"] = d["title"].astype(str) + " || " + d["performer"].astype(str)
    last_week = d.chart_week.max()
    g = d.groupby("song")
    info = pd.DataFrame({"debut": g.chart_week.min(), "peak": g.current_week.min(),
                         "weeks": g.chart_week.nunique(), "performer": g.performer.first()})
    first_peak = d.merge(info[["peak"]], left_on="song", right_index=True)
    info["peak_date"] = (first_peak[first_peak.current_week == first_peak.peak]
                         .groupby("song").chart_week.min())
    # complete histories only: not on the first chart (left-censored), >= 1 year of follow-up
    info = info[(info.debut > d.chart_week.min()) & (info.debut <= last_week - pd.Timedelta(weeks=52))]
    # position in each of the first 4 calendar weeks after the debut (101 = not on the chart)
    d = d[d.song.isin(info.index)].merge(info[["debut"]], left_on="song", right_index=True)
    d["wk"] = ((d.chart_week - d.debut).dt.days // 7) + 1
    pos = (d[d.wk <= max(KS)].pivot_table(index="song", columns="wk", values="current_week", aggfunc="min")
             .reindex(columns=range(1, max(KS) + 1)).fillna(101))
    pos.columns = [f"pos{k}" for k in pos.columns]
    t = info.join(pos)
    t["year"] = t.debut.dt.year
    t["month"] = t.debut.dt.month
    # artist history at debut: earlier songs of the same artist and earlier top-10s (strictly before)
    t["na"] = t.performer.map(normalize_artist)
    t = t.sort_values("debut")
    t["prior_songs"] = t.groupby("na").cumcount()
    # earlier top-10s of the artist that had ALREADY reached the top 10 before this song's debut
    top10_dates = (t[t.peak <= 10].groupby("na").peak_date.apply(lambda s: np.sort(s.to_numpy())))
    t["prior_top10"] = [int(np.searchsorted(top10_dates[a], deb, side="left")) if a in top10_dates.index else 0
                        for a, deb in zip(t.na, t.debut.to_numpy())]
    t["prior_songs_log"] = np.log1p(t.prior_songs)
    t["prior_top10_log"] = np.log1p(t.prior_top10)
    t["big_top10"] = (t.peak <= 10).astype(int)
    t["big_20wk"] = (t.weeks >= 20).astype(int)
    return t.reset_index()


def traj_features(k):
    cols = [f"pos{i}" for i in range(1, k + 1)]
    return cols


def add_dynamics(t):
    for k in KS:
        p = t[[f"pos{i}" for i in range(1, k + 1)]].to_numpy(float)
        t[f"best{k}"] = p.min(axis=1)
        t[f"onchart{k}"] = (p <= 100).sum(axis=1)
        if k > 1:
            t[f"slope{k}"] = (p[:, -1] - p[:, 0]) / (k - 1)          # negative = climbing
            if k > 2:
                t[f"accel{k}"] = (p[:, -1] - p[:, -2]) - (p[:, -2] - p[:, -3])   # change of the weekly move
    return t


def run_trajectory():
    t = add_dynamics(build_trajectories())
    print(f"songs with complete history: {len(t):,} ({t.debut.min():%Y} to {t.debut.max():%Y}); "
          f"top-10 share {t.big_top10.mean():.3f}, >=20 weeks share {t.big_20wk.mean():.3f}")
    ctx = ["year", "month", "prior_songs_log", "prior_top10_log"]
    rows, fold_rows = [], []
    for target in ["big_top10", "big_20wk"]:
        y = t[target].to_numpy()
        for k in KS:
            dyn = traj_features(k) + [f"best{k}", f"onchart{k}"] + ([f"slope{k}"] if k > 1 else []) \
                + ([f"accel{k}"] if k > 2 else [])
            cfg = {
                "context only (year, month, artist history)": ctx,
                "debut position": ["pos1"],
                f"first {k} week(s)": dyn,
                f"first {k} week(s) + context": dyn + ctx,
            }
            if k > 1:
                cfg.pop("debut position")
            models = {"logistic": lambda: sk_pipe(StandardScaler(), LogisticRegression(max_iter=2000)),
                      "random forest": rf200}
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
            splits = list(cv.split(np.zeros(len(y)), y))
            for mname, mk in models.items():
                for cname, cols in cfg.items():
                    X = t[cols].to_numpy(float)
                    for f, (tr, te) in enumerate(splits):
                        m = mk().fit(X[tr], y[tr])
                        s = m.predict_proba(X[te])[:, 1]
                        p = (s >= .5).astype(int)
                        fold_rows.append({"target": target, "k": k, "model": mname, "config": cname,
                                          "fold": f, "n_train": len(tr), "n_test": len(te),
                                          "accuracy": accuracy_score(y[te], p), "auc": roc_auc_score(y[te], s),
                                          "f1": f1_score(y[te], p)})
            print(f"  [{target}] k={k} done", flush=True)
    fd = pd.DataFrame(fold_rows)
    save(fd, "r10_trajectory_folds.csv")
    summ = (fd.groupby(["target", "k", "model", "config"])[["accuracy", "auc", "f1"]].mean().round(4)
              .reset_index())
    # significance: each "+ context" vs the same weeks without context, and k vs k-1 (random forest, AUC)
    tests = []
    for target in ["big_top10", "big_20wk"]:
        for k in KS:
            a = fd[(fd.target == target) & (fd.k == k) & (fd.model == "random forest")]
            base = a[a.config == f"first {k} week(s)"].set_index("fold")
            plus = a[a.config == f"first {k} week(s) + context"].set_index("fold")
            ctxo = a[a.config == "context only (year, month, artist history)"].set_index("fold")
            for name, other, ref in [("+ context vs weeks only", plus, base),
                                     ("weeks only vs context only", base, ctxo)]:
                dd = other.auc - ref.auc
                tests.append({"target": target, "k": k, "comparison": name, "delta_auc": round(dd.mean(), 4),
                              "wins": int((dd > 0).sum()), "n": len(dd),
                              "p": corrected_ttest(dd.values, ref.n_train.mean(), ref.n_test.mean())[1]})
    tests = pd.DataFrame(tests)
    # songs whose fate is still open after k weeks: not yet in the top 10
    open_rows = []
    for k in KS:
        sub = t[t[f"best{k}"] > 10]
        y = sub.big_top10.to_numpy()
        dyn = traj_features(k) + [f"best{k}", f"onchart{k}"] + ([f"slope{k}"] if k > 1 else []) \
            + ([f"accel{k}"] if k > 2 else [])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
        aucs = []
        for tr, te in cv.split(np.zeros(len(y)), y):
            m = rf200().fit(sub[dyn + ctx].to_numpy(float)[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(sub[dyn + ctx].to_numpy(float)[te])[:, 1]))
        open_rows.append({"k": k, "n_songs_not_yet_top10": len(sub), "later_top10_share": round(y.mean(), 4),
                          "auc_rf_weeks_plus_context": round(float(np.mean(aucs)), 4)})
    open_df = pd.DataFrame(open_rows)
    # temporal hold-out: train before 2000, test 2000 onwards (k = 3, random forest)
    temporal = []
    for target in ["big_top10", "big_20wk"]:
        dyn = traj_features(3) + ["best3", "onchart3", "slope3", "accel3"]
        tr, te = t.year < 2000, t.year >= 2000
        for cname, cols in [("first 3 weeks", dyn), ("first 3 weeks + context", dyn + ctx),
                            ("context only", ctx)]:
            m = rf200().fit(t.loc[tr, cols].to_numpy(float), t.loc[tr, target])
            s = m.predict_proba(t.loc[te, cols].to_numpy(float))[:, 1]
            temporal.append({"target": target, "config": cname, "auc": round(roc_auc_score(t.loc[te, target], s), 4),
                             "accuracy": round(accuracy_score(t.loc[te, target], (s >= .5).astype(int)), 4)})
    temporal = pd.DataFrame(temporal)
    for name, df_ in [("summary", summ), ("tests", tests), ("open", open_df), ("temporal", temporal)]:
        print(f"\n=== trajectory: {name} ===")
        show(df_)
        save(df_, f"r10_trajectory_{name}.csv")


# --------------------------------------------------------------------------- trend
TREND = ["trend_dist", "trend_along"]
TDEV = [f"tdev_{c}" for c in AUDIO]


def add_trend(df):
    """Market-trend features (strictly earlier decades only): distance of the standardised audio to the
    previous decade's hit centroid, projection on the direction that centroid moved, and the 13
    per-feature deviations."""
    y = df["target"].values
    Z = df[AUDIO].astype(float)
    Z = (Z - Z.mean()) / Z.std()
    dec_idx = df["decade"].map({d: i for i, d in enumerate(DEC_ORDER)}).to_numpy()
    hit_cent = {i: Z[(dec_idx == i) & (y == 1)].mean() for i in range(len(DEC_ORDER))}
    prev = np.full((len(df), len(AUDIO)), np.nan)
    move = np.full((len(df), len(AUDIO)), np.nan)
    for i in range(1, len(DEC_ORDER)):
        rows = dec_idx == i
        prev[rows] = Z[rows].to_numpy() - hit_cent[i - 1].to_numpy()                # distance to last decade's hits
        if i >= 2:
            move[rows] = hit_cent[i - 1].to_numpy() - hit_cent[i - 2].to_numpy()       # direction of the trend
    df["trend_dist"] = np.sqrt(np.nansum(prev ** 2, axis=1))
    df.loc[dec_idx == 0, "trend_dist"] = np.nan
    proj = np.nansum(prev * move, axis=1) / np.maximum(np.sqrt(np.nansum(move ** 2, axis=1)), 1e-9)
    df["trend_along"] = np.where(dec_idx >= 2, proj, np.nan)
    df[TREND] = df[TREND].fillna(df[TREND].median())
    for j, c in enumerate(AUDIO):
        df[f"tdev_{c}"] = np.nan_to_num(prev[:, j])
    return df


def run_trend():
    df = add_trend(load_kaggle())
    y = df["target"].values
    print("Previous-decade hit centroids are computed from the benchmark's own hits (strictly earlier decade).")
    configs = {
        "main (audio + decade + artist)": make_pipeline(AUDIO + DEC_COLS + [ART]),
        "+ trend distance & direction (2)": make_pipeline(AUDIO + DEC_COLS + [ART] + TREND),
        "+ per-feature deviation from last decade's hits (13)": make_pipeline(AUDIO + DEC_COLS + [ART] + TDEV),
        "audio only": make_pipeline(AUDIO),
        "audio + trend (2)": make_pipeline(AUDIO + TREND),
    }
    summ, folds = cv_eval(configs, df, y, label="trend")
    print("\n=== trend: 5x5 CV (ref = main model) ===")
    show(summ)
    save(summ, "r10_trend_cv_summary.csv")
    save(folds, "r10_trend_cv_folds.csv")
    summ2, _ = cv_eval({"audio only": configs["audio only"], "audio + trend (2)": configs["audio + trend (2)"]},
                       df, y, label="trend-audio")
    print("\n=== trend on audio only ===")
    show(summ2)
    save(summ2, "r10_trend_audio_cv_summary.csv")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["trajectory", "trend"]
    if "trajectory" in todo:
        run_trajectory()
    if "trend" in todo:
        run_trend()
