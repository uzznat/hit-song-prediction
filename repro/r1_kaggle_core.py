"""
R1 -- main result on the full Kaggle dataset (40,560 songs).

  * canonical split  : must reproduce 0.773 / 0.7955 / 0.800 / 0.8215
  * 5x5 repeated CV  : mean, 95 % CI, corrected paired t-test vs the audio baseline
  * per-decade accuracy (CV) for baseline vs + decade + artist
  * dataset facts used in the paper (hit share per decade, artist match rate,
    mean prior hits for hits / flops, hit rate by prior-hit bucket)
  * candidate features rejected in the deck: song title (artifact), title
    length, chorus_hit / sections (leaky columns)

    py repro/r1_kaggle_core.py
"""
import numpy as np
import pandas as pd

from common import (AUDIO, ART, DEC_COLS, TitleOOF, canonical_eval, canonical_split,
                    cv_eval, load_kaggle, make_pipeline, save, show)


def main():
    df = load_kaggle()
    y = df["target"].values

    # ---------------------------------------------------------------- dataset facts
    from feature_experiments import build_artist_history
    hist = build_artist_history()
    df["title_words"] = df["track"].astype(str).str.split().str.len()
    facts = {
        "n_songs": len(df),
        "hit_share": round(y.mean(), 4),
        "artist_match_rate": round(df["na"].isin(hist).mean(), 4),
        "mean_prior_hits_hits": round(df.loc[df.target == 1, "artist_prior_hits"].mean(), 3),
        "mean_prior_hits_flops": round(df.loc[df.target == 0, "artist_prior_hits"].mean(), 3),
    }
    for d, g in df.groupby("decade"):
        facts[f"n_{d}"] = len(g)
        facts[f"hit_share_{d}"] = round(g.target.mean(), 4)
    b = pd.cut(df["artist_prior_hits"], [-1, 0, 2, 5, 10, np.inf],
               labels=["0", "1-2", "3-5", "6-10", "11+"])
    for lab, g in df.groupby(b, observed=True):
        facts[f"hit_rate_prior_{lab}"] = round(g.target.mean(), 4)
        facts[f"n_prior_{lab}"] = len(g)
    save(pd.DataFrame([facts]).T.reset_index().rename(columns={"index": "fact", 0: "value"}),
         "r1_dataset_facts.csv")

    # ---------------------------------------------------------------- configs
    configs = {
        "audio (13)":               make_pipeline(AUDIO),
        "+ decade":                 make_pipeline(AUDIO + DEC_COLS),
        "+ artist":                 make_pipeline(AUDIO + [ART]),
        "+ decade + artist":        make_pipeline(AUDIO + DEC_COLS + [ART]),
        "decade + artist only":     make_pipeline(DEC_COLS + [ART]),
        # rejected candidates
        "+ title length (words)":   make_pipeline(AUDIO + ["title_words"]),
        "+ chorus_hit + sections":  make_pipeline(AUDIO + ["chorus_hit", "sections"]),
        "+ song title (TF-IDF)":    make_pipeline(AUDIO, extra=[("title", TitleOOF(), ["track"])]),
    }

    tr, _, te = canonical_split(df)
    can = canonical_eval(configs, df, y, tr, te)
    print("\n=== canonical split (seed 42, 20k train / 2k test) ===")
    show(can)
    save(can, "r1_canonical.csv")

    summ, folds = cv_eval(configs, df, y, label="R1", groups=df["decade"].values)
    print("\n=== 5x5 repeated stratified CV ===")
    show(summ)
    save(summ, "r1_cv_summary.csv")
    save(folds, "r1_cv_folds.csv")

    dec = (folds.groupby("config")[[c for c in folds.columns if c.startswith("acc_")]]
                .mean().round(4).reset_index())
    print("\n=== per-decade accuracy (CV mean) ===")
    show(dec)
    save(dec, "r1_cv_per_decade.csv")


if __name__ == "__main__":
    main()
