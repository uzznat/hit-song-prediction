"""
R6 -- two robustness checks a reviewer would ask for first.

  1. Artist-grouped CV: does the gain survive when no artist has songs in both
     the training and the test part of a fold?  (StratifiedGroupKFold by
     normalised artist, same number of folds / repeats as the main protocol.)
  2. Cold start: accuracy restricted to songs whose artist has NO earlier hit
     ("new") vs songs whose artist has at least one ("known").

  Run on the Kaggle benchmark (5x5) and on both ISMIR 2019 protocols (5x2).

    py repro/r6_robustness.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (AUDIO, ART, DEC_COLS, ROOT, SEED, corrected_ttest, cv_eval, load_kaggle,
                    make_pipeline, save, show)


def grouped_splits(y, groups, n_splits, n_repeats):
    out = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + r)
        out += list(cv.split(np.zeros(len(y)), y, groups))
    return out


def fill_empty_groups(g):
    """Songs with no usable artist name become their own group."""
    g = pd.Series(g).astype(str).reset_index(drop=True)
    empty = (g == "") | (g == "nan")
    g[empty] = ["__solo_%d" % i for i in np.flatnonzero(empty)]
    return g.values


def cold_start_table(folds, ref, tag, cv_name, y, status):
    """Per-status accuracy (mean over folds) + corrected t-test vs ref."""
    rows = []
    base = folds[folds.config == ref].set_index("fold")
    for name, g in folds.groupby("config", sort=False):
        g = g.set_index("fold")
        for st in ("new", "known"):
            col = f"acc_{st}"
            row = {"data": tag, "cv": cv_name, "config": name, "artists": st,
                   "n_songs": int((status == st).sum()),
                   "hit_share": round(float(y[status == st].mean()), 4),
                   "acc_mean": round(g[col].mean(), 4)}
            if name != ref:
                d = (g[col] - base[col]).values
                t, p = corrected_ttest(d, g.n_train.mean(), g.n_test.mean())
                row.update(delta_vs_ref=round(d.mean(), 4), corrected_p=p,
                           wins=int((d > 0).sum()), losses=int((d < 0).sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def run(tag, data, y, configs, ref, artist, status, n_splits, n_repeats):
    summaries, cold = [], []
    groups = fill_empty_groups(artist)
    for cv_name, splits in [("stratified", None),
                            ("artist-grouped", grouped_splits(y, groups, n_splits, n_repeats))]:
        if splits is not None:           # sanity check: no artist on both sides
            tr, te = splits[0]
            assert not set(groups[tr]) & set(groups[te])
        summ, folds = cv_eval(configs, data, y, ref=ref, n_splits=n_splits,
                              n_repeats=n_repeats, label=f"{tag} {cv_name}",
                              groups=status, splits=splits)
        summ.insert(0, "cv", cv_name)
        summ.insert(0, "data", tag)
        print(f"\n=== {tag}: {cv_name} CV ===")
        show(summ)
        summaries.append(summ)
        save(folds, f"r6_{tag}_{cv_name}_folds.csv")
        c = cold_start_table(folds, ref, tag, cv_name, y, status)
        print(f"\n=== {tag}: {cv_name} CV, new vs known artists ===")
        show(c)
        cold.append(c)
    return pd.concat(summaries), pd.concat(cold)


def kaggle():
    df = load_kaggle()
    y = df["target"].values
    status = np.where(df["artist_prior_hits"].values > 0, "known", "new")
    configs = {"audio (13)":        make_pipeline(AUDIO),
               "+ decade":          make_pipeline(AUDIO + DEC_COLS),
               "+ artist":          make_pipeline(AUDIO + [ART]),
               "+ decade + artist": make_pipeline(AUDIO + DEC_COLS + [ART])}
    return run("kaggle", df, y, configs, "audio (13)", df["na"].values, status, 5, 5)


def ismir():
    sys.path.insert(0, ROOT)
    import run_ismir_v2 as ri
    from r4_external import rf200
    df = ri.add_our_features(ri.load_labels())
    af = pd.read_pickle(ri.CACHE)
    data = df.merge(af, left_on="msd_id", right_index=True, how="inner")
    audio = [c for c in af.columns if data[c].notna().mean() > 0.95]
    data = data.dropna(subset=audio).reset_index(drop=True)
    out_s, out_c = [], []
    for name, matched in [("ismir_A_random", False), ("ismir_B_yearmatched", True)]:
        s = (ri.balanced_sample(data, matched, None)
               .sample(frac=1.0, random_state=SEED).reset_index(drop=True))
        s["year_n"] = s["year"].astype(float)
        y = s["target"].values
        status = np.where(s["prior"].fillna(0).values > 0, "known", "new")
        P = lambda cols: make_pipeline(cols, model=rf200())
        configs = {"their audio (592 Essentia)": P(audio),
                   "+ year":                     P(audio + ["year_n"]),
                   "+ artist":                   P(audio + [ART]),
                   "+ year + artist":            P(audio + ["year_n", ART])}
        a, b = run(name, s, y, configs, "their audio (592 Essentia)",
                   s["artist_norm"].values, status, 5, 2)
        out_s.append(a)
        out_c.append(b)
    return pd.concat(out_s), pd.concat(out_c)


def main():
    which = sys.argv[1:] or ["kaggle", "ismir"]
    for w in which:
        s, c = kaggle() if w == "kaggle" else ismir()
        save(s, f"r6_{w}_cv_summary.csv")
        save(c, f"r6_{w}_cold_start.csv")


if __name__ == "__main__":
    main()
