"""
R5 -- head-to-head with the only same-dataset academic work:
      J. L. Du, "Spotify between decades: Hit or Miss predictor",
      MSc thesis, Tilburg University, 2022.

Their protocol (from the thesis text):
  * 2000s and 2010s only, ONE model per decade
  * 70/30 split (caTools::sample.split, set.seed(1)) -- R's RNG cannot be
    replayed in Python, so we repeat the 70/30 split over 50 seeds and report
    the distribution (and the fraction of seeds where we beat their number)
  * all numeric columns: 13 audio + chorus_hit + sections
  * randomForest: ntree=450, mtry=2, maxnodes=30   -> RF 00s 84 %, 10s 82 %
  * logistic regression, cutoff 0.5                -> LR 00s 82 %, 10s 79 %

We run THEIR model on THEIR inputs, then the SAME model + our artist feature
(decade is constant inside a per-decade model, so it cannot help there), and
also our frozen RF with + artist. Paired comparison per seed (same split).

    py repro/r5_tilburg.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import AUDIO, ART, FrozenRF, corrected_ttest, load_kaggle, save, show

THEIR = AUDIO + ["chorus_hit", "sections"]
PUBLISHED = {("RF", "00s"): 0.84, ("RF", "10s"): 0.82, ("LR", "00s"): 0.82, ("LR", "10s"): 0.79}
N_SEEDS = 50


def their_rf(seed):
    return RandomForestClassifier(n_estimators=450, max_features=2, max_leaf_nodes=30,
                                  random_state=seed, n_jobs=-1)


def their_lr(seed):
    return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])


def main():
    df = load_kaggle()
    rows = []
    for dec in ("00s", "10s"):
        d = df[df["decade"] == dec].reset_index(drop=True)
        y = d["target"].values
        print(f"{dec}: {len(d)} songs after uri de-duplication, hit share {y.mean():.3f}")
        for seed in range(N_SEEDS):
            tr, te = train_test_split(np.arange(len(d)), test_size=0.30, stratify=y,
                                      random_state=seed)
            for model, factory in [("RF", their_rf), ("LR", their_lr)]:
                for feats, cols in [("their inputs", THEIR), ("+ artist", THEIR + [ART])]:
                    m = factory(seed).fit(d.loc[tr, cols].values, y[tr])
                    rows.append({"decade": dec, "seed": seed, "model": f"{model} (their config)",
                                 "features": feats,
                                 "accuracy": accuracy_score(y[te], m.predict(d.loc[te, cols].values))})
            for feats, cols in [("audio (13)", AUDIO), ("+ artist", AUDIO + [ART])]:
                m = FrozenRF(random_state=seed).fit(d.loc[tr, cols].values, y[tr])
                rows.append({"decade": dec, "seed": seed, "model": "RF (frozen, ours)",
                             "features": feats,
                             "accuracy": accuracy_score(y[te], m.predict(d.loc[te, cols].values))})
        print(f"  {dec} done", flush=True)
    raw = pd.DataFrame(rows)
    save(raw, "r5_tilburg_seeds.csv")

    out = []
    for (dec, model, feats), g in raw.groupby(["decade", "model", "features"], sort=False):
        a = g.accuracy.values
        pub = PUBLISHED.get((model.split()[0], dec)) if "their" in model else None
        row = {"decade": dec, "model": model, "features": feats,
               "acc_mean": round(a.mean(), 4), "acc_std": round(a.std(ddof=1), 4),
               "acc_min": round(a.min(), 4), "acc_max": round(a.max(), 4),
               "published": pub}
        if pub is not None:
            row["share_seeds_above_published"] = round((a > pub).mean(), 2)
        out.append(row)
    summ = pd.DataFrame(out)
    # paired gain of + artist over the same model's baseline, per seed
    gains = []
    for (dec, model), g in raw.groupby(["decade", "model"], sort=False):
        piv = g.pivot(index="seed", columns="features", values="accuracy")
        base = [c for c in piv.columns if c != "+ artist"][0]
        diff = piv["+ artist"] - piv[base]
        gains.append({"decade": dec, "model": model, "baseline_features": base,
                      "mean_gain_artist": round(diff.mean(), 4),
                      "seeds_with_gain": int((diff > 0).sum()), "n_seeds": len(diff),
                      # splits overlap across seeds -> Nadeau-Bengio correction
                      "corrected_p": corrected_ttest(diff.values, 0.70, 0.30)[1]})
    gains = pd.DataFrame(gains)
    print("\n=== Tilburg protocol, 50 seeds of the 70/30 split ===")
    show(summ)
    print("\n=== paired gain of the artist feature ===")
    show(gains)
    save(summ, "r5_tilburg_summary.csv")
    save(gains, "r5_tilburg_artist_gain.csv")


if __name__ == "__main__":
    main()
