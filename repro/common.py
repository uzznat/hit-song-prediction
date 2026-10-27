"""
repro/common.py -- shared data loading and evaluation for the paper.
=====================================================================
Every number in the paper is produced by a script in repro/ and written to
paper_results/. Two evaluation modes are used everywhere:

  canonical : the exact split of the original experiments (seed 42) -- used to
              check that the numbers shown in the presentation are reproduced.
  cv        : repeated stratified K-fold (default 5 folds x 5 repeats). Gives a
              mean, a 95 % CI and a paired corrected resampled t-test
              (Nadeau & Bengio 2003) against a reference configuration.

All learned transformations (scaling, PCA, title encoder) live inside a
sklearn Pipeline, so they are re-fitted on the training part of every fold.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                                   # the older loaders use paths relative to the root
DATA = os.path.join(ROOT, "data")                # raw datasets (see docs/DATA.md)
OUT = os.path.join(ROOT, "paper_results")
os.makedirs(OUT, exist_ok=True)

from phase0_harness import BASE_FEATURES_A as AUDIO, mcnemar_test  # noqa: E402

SEED = 42
DEC_ORDER = ["60s", "70s", "80s", "90s", "00s", "10s"]
DEC_COLS = [f"is_{d}" for d in DEC_ORDER]
ART = "artist_prior_hits_log"


# --------------------------------------------------------------------------- data
def load_kaggle():
    """The 40,560-song Kaggle frame with decade one-hot and the artist feature,
    rebuilt exactly as in feature_experiments.py."""
    from feature_experiments import load_spotify_with_features
    df = load_spotify_with_features()
    for d, c in zip(DEC_ORDER, DEC_COLS):
        df[c] = (df["decade"] == d).astype(float)
    return df


def canonical_split(df):
    """Exact split of the original experiments: 20k train / 2k val / 2k test."""
    y = df["target"].values
    idx = np.arange(len(df))
    sel, _ = train_test_split(idx, train_size=24000, stratify=y, random_state=SEED)
    tr, tmp = train_test_split(sel, train_size=20000, stratify=y[sel], random_state=SEED)
    val, te = train_test_split(tmp, train_size=2000, stratify=y[tmp], random_state=SEED)
    return tr, val, te


def load_audio_subset(df):
    """Kaggle songs that have a Deezer clip, with rich (185) and FFT (31) features."""
    rich = pd.read_csv(os.path.join(ROOT, "cache", "rich_features.csv"))
    fft = pd.read_csv(os.path.join(ROOT, "cache", "fft_features.csv"))
    rcols = [c for c in rich.columns if c != "uri"]
    fcols = [c for c in fft.columns if c != "uri"]
    fft = fft.rename(columns={c: "fft_" + c for c in fcols})
    fcols = ["fft_" + c for c in fcols]
    data = df.merge(rich, on="uri", how="inner").dropna(subset=rcols)
    data = data.merge(fft, on="uri", how="left").reset_index(drop=True)
    return data, rcols, fcols


# --------------------------------------------------------------------------- models
class FrozenRF(RandomForestClassifier):
    """The frozen baseline RF of the original experiments (80 trees,
    max_features=8), with max_features capped at the number of columns so the
    same factory also works on small feature blocks."""

    def __init__(self, n_estimators=80, max_features=8, random_state=SEED, n_jobs=-1):
        super().__init__(n_estimators=n_estimators, max_features=max_features,
                         min_samples_split=2, criterion="gini",
                         random_state=random_state, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        if isinstance(self.max_features, int):
            self.max_features = min(self.max_features, X.shape[1])
        return super().fit(X, y, sample_weight)


def make_pipeline(plain_cols, pca_cols=None, n_pca=40, model=None, extra=None):
    """ColumnTransformer: plain columns passed through, `pca_cols` scaled + PCA
    (fitted on train only), optional `extra` = [(name, transformer, cols)]."""
    parts = []
    if plain_cols:
        parts.append(("plain", "passthrough", list(plain_cols)))
    if pca_cols:
        red = ("passthrough" if n_pca is None else
               Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=n_pca, random_state=SEED))]))
        parts.append(("rich", red, list(pca_cols)))
    for e in extra or []:
        parts.append(e)
    return Pipeline([("cols", ColumnTransformer(parts, remainder="drop")),
                     ("clf", model if model is not None else FrozenRF())])


class TitleOOF(BaseEstimator, TransformerMixin):
    """Song-title model as ONE numeric feature: TF-IDF word n-grams ->
    logistic regression P(hit). On the training part it returns OUT-OF-FOLD
    probabilities (no row sees its own label); on new rows, the refit model."""

    def __init__(self, inner_folds=5):
        self.inner_folds = inner_folds

    def _model(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                                sublinear_tf=True)),
                         ("lr", LogisticRegression(C=4.0, max_iter=2000))])

    @staticmethod
    def _text(X):
        s = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.asarray(X).ravel())
        return np.asarray(s.fillna("").astype(str).tolist(), dtype=object)

    def fit(self, X, y):
        self.m_ = self._model().fit(self._text(X), y)
        return self

    def transform(self, X):
        return self.m_.predict_proba(self._text(X))[:, [1]]

    def fit_transform(self, X, y=None, **kw):
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        t = self._text(X)
        oof = cross_val_predict(self._model(), t, y, method="predict_proba",
                                cv=StratifiedKFold(self.inner_folds, shuffle=True,
                                                   random_state=SEED))[:, [1]]
        self.m_ = self._model().fit(t, y)
        return oof


# --------------------------------------------------------------------------- evaluation
def _scores(model, X, y):
    p = model.predict(X)
    try:
        s = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, s)
    except (AttributeError, ValueError):
        auc = float("nan")
    return p, accuracy_score(y, p), auc


def canonical_eval(configs, data, y, tr, te, ref=None):
    """Fit every config on `tr`, score on `te`, McNemar vs `ref`."""
    ref = ref or next(iter(configs))
    preds, rows = {}, []
    for name, pipe in configs.items():
        m = clone(pipe).fit(data.iloc[tr], y[tr])
        p, acc, auc = _scores(m, data.iloc[te], y[te])
        preds[name] = p
        rows.append({"config": name, "accuracy": round(acc, 4), "roc_auc": round(auc, 4)})
    ref_acc = next(r["accuracy"] for r in rows if r["config"] == ref)
    for r in rows:
        if r["config"] != ref:
            mc = mcnemar_test(y[te], preds[ref], preds[r["config"]])
            r.update(delta=round(r["accuracy"] - ref_acc, 4), mcnemar_p=mc["pvalue"],
                     new_right_only=mc["n_new_only"], ref_right_only=mc["n_base_only"])
    return pd.DataFrame(rows)


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold diffs."""
    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def cv_eval(configs, data, y, ref=None, n_splits=5, n_repeats=5, seed=SEED,
            label="", groups=None, splits=None):
    """Repeated stratified K-fold. Returns (summary DataFrame, per-fold DataFrame).
    `groups` (array aligned with data) -> also per-group accuracy per fold.
    `splits` (list of (train_idx, test_idx)) replaces the default splitter."""
    ref = ref or next(iter(configs))
    if splits is None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        splits = list(cv.split(np.zeros(len(y)), y))
    folds = []
    t0 = time.time()
    for f, (tr, te) in enumerate(splits):
        for name, pipe in configs.items():
            m = clone(pipe).fit(data.iloc[tr], y[tr])
            p, acc, auc = _scores(m, data.iloc[te], y[te])
            row = {"fold": f, "config": name, "accuracy": acc, "roc_auc": auc,
                   "n_train": len(tr), "n_test": len(te)}
            if groups is not None:
                g = np.asarray(groups)[te]
                for gv in np.unique(g):
                    row[f"acc_{gv}"] = accuracy_score(y[te][g == gv], p[g == gv])
            folds.append(row)
        print(f"  [{label}] fold {f + 1}/{len(splits)}  ({time.time() - t0:.0f} s)",
              flush=True)
    fd = pd.DataFrame(folds)
    rows = []
    base = fd[fd.config == ref].set_index("fold")
    for name in configs:
        g = fd[fd.config == name].set_index("fold")
        k = len(g)
        acc_m, acc_s = g.accuracy.mean(), g.accuracy.std(ddof=1)
        half = stats.t.ppf(0.975, k - 1) * acc_s / np.sqrt(k)
        row = {"config": name, "acc_mean": round(acc_m, 4), "acc_std": round(acc_s, 4),
               "acc_ci95_low": round(acc_m - half, 4), "acc_ci95_high": round(acc_m + half, 4),
               "auc_mean": round(g.roc_auc.mean(), 4), "n_folds": k}
        if name != ref:
            diff = g.accuracy - base.accuracy
            t, p = corrected_ttest(diff.values, g.n_train.mean(), g.n_test.mean())
            row.update(delta_vs_ref=round(diff.mean(), 4), corrected_t=round(t, 3),
                       corrected_p=p, wins=int((diff > 0).sum()), losses=int((diff < 0).sum()))
        rows.append(row)
    return pd.DataFrame(rows), fd


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"-> {path}")
    return path


def show(df):
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))
