"""
R9 -- precision, recall and F1 of the hit class for the headline comparisons
(reviewers of imbalanced tasks expect more than accuracy).

  * Kaggle Hit Predictor, frozen RF, 5x5 CV : audio (13) vs + decade + artist
  * SpotGenTrack (19.4 % hits), RF-200, 5x1 CV : 13 Spotify vs + decade + artist

Same folds as R1 / R4 (RepeatedStratifiedKFold, seed 42).

    py repro/r9_metrics.py
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from common import ART, AUDIO, DEC_COLS, SEED, corrected_ttest, load_kaggle, make_pipeline, save
from r4_external import SPOTIFY_13, load_spotgentrack, rf200


def run(tag, data, y, configs, n_repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        for name, pipe in configs.items():
            p = clone(pipe).fit(data.iloc[tr], y[tr]).predict(data.iloc[te])
            rows.append({"fold": f, "config": name, "n_train": len(tr), "n_test": len(te),
                         "accuracy": accuracy_score(y[te], p),
                         "precision": precision_score(y[te], p, zero_division=0),
                         "recall": recall_score(y[te], p),
                         "f1": f1_score(y[te], p),
                         "tp": int(((p == 1) & (y[te] == 1)).sum()),
                         "fp": int(((p == 1) & (y[te] == 0)).sum()),
                         "fn": int(((p == 0) & (y[te] == 1)).sum()),
                         "tn": int(((p == 0) & (y[te] == 0)).sum())})
        print(f"  [{tag}] fold {f + 1}", flush=True)
    fd = pd.DataFrame(rows)
    names = list(configs)
    base = fd[fd.config == names[0]].set_index("fold")
    out = []
    for name in names:
        g = fd[fd.config == name].set_index("fold")
        r = {"dataset": tag, "config": name}
        for m in ["accuracy", "precision", "recall", "f1"]:
            r[m] = round(g[m].mean(), 4)
        if name != names[0]:
            d = g.f1 - base.f1
            r["f1_delta"] = round(d.mean(), 4)
            r["f1_p"] = corrected_ttest(d.values, g.n_train.mean(), g.n_test.mean())[1]
            r["f1_wins"] = int((d > 0).sum())
        out.append(r)
    save(fd, f"r9_{tag}_folds.csv")
    return pd.DataFrame(out)


def main():
    df = load_kaggle()
    k = run("kaggle", df, df["target"].values,
            {"audio (13)": make_pipeline(AUDIO),
             "+ decade + artist": make_pipeline(AUDIO + DEC_COLS + [ART])}, 5)
    data, _ = load_spotgentrack()
    P = lambda cols: make_pipeline(cols, model=rf200())
    s = run("spotgentrack", data, data["target"].values,
            {"their 13 Spotify": P(SPOTIFY_13),
             "+ decade + artist": P(SPOTIFY_13 + ["decade_n", "artist_prior_log"])}, 1)
    res = pd.concat([k, s], ignore_index=True)
    print(res.to_string(index=False))
    save(res, "r9_metrics_summary.csv")


if __name__ == "__main__":
    main()
