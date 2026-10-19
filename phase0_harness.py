"""
phase0_harness.py
=================
Phase 0 of the Hit-Song reproduction project: the single evaluation harness that
EVERY candidate feature (artist history, FFT / rhythmic-temporal, lyrics, ...) is
put through, so all comparisons are fair, reproducible, and statistically honest.

Design contract
---------------
- The reproduced Random Forest baseline (paper's config, set A) is FROZEN and is
  the one reference every feature is measured against.
- One fixed stratified split (seed 42), produced by prep2.py -> psplit_{tag}.npz.
- A new feature (or block of features) is judged on FOUR axes, not just accuracy:
    1. Predictive lift ....... delta accuracy (base vs base+new), same split, test used once
    2. Significance .......... McNemar's test on the shared test set  (keep if p < 0.05)
    3. Non-redundancy ........ max |correlation| with the 13 base features + VIF
    4. Selection survival .... LASSO coefficient (does it survive L1 shrinkage?)
                              + permutation importance inside the augmented model
- Plus a subgroup breakdown (accuracy per decade / genre), because the target
  outcome is "significant lift on a characterised subgroup", not global SOTA.

Dependencies: numpy, pandas, scipy, scikit-learn  (no statsmodels needed).

Typical use
-----------
    from phase0_harness import (load_base_split, freeze_baseline,
                                evaluate_feature, print_report)

    base = load_base_split(tag="A")                 # 13 audio features, fixed split
    freeze_baseline(base)                            # sanity: should print RF ~0.773

    # x_new_* are the new feature column(s) aligned to the SAME rows as the split.
    res = evaluate_feature(base, x_new_tr, x_new_val, x_new_te,
                           feature_names=["artist_prior_hits"],
                           groups_te=decade_te)      # decade_te optional -> subgroups
    print_report(res)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score

SEED = 42

# The 13 audio features of set A, in the exact order prep2.py stores them.
BASE_FEATURES_A = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature",
]


# --------------------------------------------------------------------------- #
# Baseline model (paper's RF, set A) -- FROZEN. Do not tune this.
# --------------------------------------------------------------------------- #
def make_baseline_rf() -> RandomForestClassifier:
    """The reproduced Random Forest baseline, identical to run_paper.py."""
    return RandomForestClassifier(
        n_estimators=80, max_features=8, min_samples_split=2,
        criterion="gini", random_state=SEED, n_jobs=-1,
    )


def load_base_split(tag: str = "A",
                    path_tmpl: str = "/home/claude/psplit_{tag}.npz") -> dict:
    """Load the fixed split produced by prep2.py."""
    d = np.load(path_tmpl.format(tag=tag))
    keys = ["X_tr", "X_val", "X_te", "y_tr", "y_val", "y_te"]
    split = {k: d[k] for k in keys}
    split["feature_names"] = list(BASE_FEATURES_A) if tag == "A" else None
    return split


def freeze_baseline(base: dict) -> dict:
    """Fit the frozen baseline and report val/test accuracy (should match ~0.773)."""
    m = make_baseline_rf().fit(base["X_tr"], base["y_tr"])
    out = {}
    for name, X, y in [("val", base["X_val"], base["y_val"]),
                       ("test", base["X_te"], base["y_te"])]:
        p = m.predict(X)
        out[name] = {
            "acc": round(float(accuracy_score(y, p)), 4),
            "prec": round(float(precision_score(y, p)), 4),
            "rec": round(float(recall_score(y, p)), 4),
        }
    print(f"[baseline] frozen RF  val_acc={out['val']['acc']}  "
          f"test_acc={out['test']['acc']}")
    return out


# --------------------------------------------------------------------------- #
# Statistical primitives
# --------------------------------------------------------------------------- #
def mcnemar_test(y_true, pred_base, pred_aug) -> dict:
    """
    McNemar's test comparing two classifiers on the SAME test set.
    n01 = base right / aug wrong ; n10 = base wrong / aug right (the discordant pairs).
    Exact binomial when few discordant pairs, chi-square (continuity-corrected) otherwise.
    """
    y_true = np.asarray(y_true)
    base_ok = np.asarray(pred_base) == y_true
    aug_ok = np.asarray(pred_aug) == y_true
    n01 = int(np.sum(base_ok & ~aug_ok))   # only the baseline got it right
    n10 = int(np.sum(~base_ok & aug_ok))   # only the augmented model got it right
    n_disc = n01 + n10

    if n_disc == 0:
        return {"pvalue": 1.0, "statistic": 0.0, "n_base_only": n01,
                "n_new_only": n10, "method": "no discordant pairs"}

    if n_disc < 25:  # exact two-sided binomial
        p = float(stats.binomtest(min(n01, n10), n_disc, 0.5).pvalue)
        return {"pvalue": p, "statistic": float(min(n01, n10)),
                "n_base_only": n01, "n_new_only": n10, "method": "exact binomial"}

    stat = (abs(n01 - n10) - 1) ** 2 / n_disc           # continuity correction
    p = float(stats.chi2.sf(stat, df=1))
    return {"pvalue": p, "statistic": float(stat), "n_base_only": n01,
            "n_new_only": n10, "method": "chi2 (corrected)"}


def redundancy_check(X_base, new_block, base_names=None) -> list[dict]:
    """
    For each new feature: max |Pearson corr| with any base feature, which base
    feature that is, and its VIF (regress the new feature on ALL base features;
    VIF = 1/(1-R^2)). High corr / high VIF  =>  the info is already in set A.
    """
    X_base = np.asarray(X_base, float)
    new_block = np.asarray(new_block, float)
    if new_block.ndim == 1:
        new_block = new_block[:, None]
    n_base = X_base.shape[1]
    base_names = base_names or [f"f{j}" for j in range(n_base)]

    out = []
    for c in range(new_block.shape[1]):
        x = new_block[:, c]
        corrs = np.array([abs(np.corrcoef(X_base[:, j], x)[0, 1])
                          for j in range(n_base)])
        corrs = np.nan_to_num(corrs)
        j = int(np.argmax(corrs))
        r2 = LinearRegression().fit(X_base, x).score(X_base, x)
        vif = float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))
        out.append({
            "max_abs_corr": round(float(corrs[j]), 3),
            "closest_base_feature": base_names[j],
            "vif": round(vif, 2),
            "verdict": _redundancy_verdict(corrs[j], vif),
        })
    return out


def _redundancy_verdict(max_corr, vif) -> str:
    if max_corr >= 0.9 or vif >= 10:
        return "REDUNDANT (already captured by set A)"
    if max_corr >= 0.7 or vif >= 5:
        return "partly redundant"
    return "orthogonal (adds new signal)"


def lasso_survival(X_base, new_block, y, base_names=None) -> list[dict]:
    """
    Fit LASSO (5-fold CV) on [base | new] standardised, report each NEW feature's
    coefficient. |coef| ~ 0 means L1 shrinks it away -> not a good feature.
    """
    X_base = np.asarray(X_base, float)
    new_block = np.asarray(new_block, float)
    if new_block.ndim == 1:
        new_block = new_block[:, None]
    n_new = new_block.shape[1]
    X = np.column_stack([X_base, new_block])
    Xs = StandardScaler().fit_transform(X)
    m = LassoCV(cv=5, random_state=SEED, max_iter=20000, n_jobs=-1).fit(Xs, y)
    names = _new_names(base_names, X_base.shape[1], n_new)
    coefs = m.coef_[-n_new:]
    return [{"feature": names[i], "lasso_coef": round(float(coefs[i]), 4),
             "survives_lasso": bool(abs(coefs[i]) > 1e-6)} for i in range(n_new)]


def _new_names(base_names, n_base, n_new):
    # names passed to evaluate_feature refer to the NEW block only
    if base_names and len(base_names) == n_new:
        return list(base_names)
    return [f"new_{i}" for i in range(n_new)]


def subgroup_breakdown(y_true, pred_base, pred_aug, groups) -> pd.DataFrame:
    """Per-group accuracy for baseline vs augmented, with the delta and support."""
    df = pd.DataFrame({"y": np.asarray(y_true),
                       "pb": np.asarray(pred_base),
                       "pa": np.asarray(pred_aug),
                       "g": np.asarray(groups)})
    rows = []
    for g, sub in df.groupby("g"):
        ab = accuracy_score(sub.y, sub.pb)
        aa = accuracy_score(sub.y, sub.pa)
        rows.append({"group": g, "n": int(len(sub)),
                     "acc_base": round(ab, 4), "acc_aug": round(aa, 4),
                     "delta": round(aa - ab, 4)})
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def evaluate_feature(base: dict, x_new_tr, x_new_val, x_new_te,
                     feature_names=None, groups_te=None,
                     model_factory=make_baseline_rf) -> dict:
    """
    Evaluate a new feature (or block of features) against the frozen baseline.

    Parameters
    ----------
    base        : dict from load_base_split (X_tr/X_val/X_te, y_tr/y_val/y_te, feature_names)
    x_new_*     : new feature column(s) for each split, aligned to the SAME rows.
                  Shape (n,) for a single feature or (n, k) for a block.
    feature_names : names of the NEW feature(s) only (len k).
    groups_te   : optional per-row group labels for the TEST set (e.g. decade, genre)
                  -> triggers the subgroup breakdown.

    Returns a dict with delta accuracy, McNemar, redundancy, LASSO, permutation
    importance, and (optionally) the subgroup table.
    """
    def col(a):
        a = np.asarray(a, float)
        return a[:, None] if a.ndim == 1 else a

    Xtr_b, Xval_b, Xte_b = base["X_tr"], base["X_val"], base["X_te"]
    ytr, yval, yte = base["y_tr"], base["y_val"], base["y_te"]
    base_names = base.get("feature_names")

    ntr, nval, nte = col(x_new_tr), col(x_new_val), col(x_new_te)
    Xtr_a = np.column_stack([Xtr_b, ntr])
    Xte_a = np.column_stack([Xte_b, nte])
    n_new = ntr.shape[1]

    # baseline (set A) vs augmented (set A + new) -- both frozen RF, test used once
    m_base = model_factory().fit(Xtr_b, ytr)
    m_aug = model_factory().fit(Xtr_a, ytr)
    p_base = m_base.predict(Xte_b)
    p_aug = m_aug.predict(Xte_a)

    acc_base = float(accuracy_score(yte, p_base))
    acc_aug = float(accuracy_score(yte, p_aug))

    # permutation importance of the new column(s) inside the augmented model
    pi = permutation_importance(m_aug, Xte_a, yte, n_repeats=20,
                                random_state=SEED, n_jobs=-1)
    perm_new = [round(float(pi.importances_mean[Xtr_b.shape[1] + i]), 5)
                for i in range(n_new)]

    res = {
        "feature_names": feature_names or _new_names(None, 0, n_new),
        "n_new_features": n_new,
        "acc_base": round(acc_base, 4),
        "acc_aug": round(acc_aug, 4),
        "delta_acc": round(acc_aug - acc_base, 4),
        "mcnemar": mcnemar_test(yte, p_base, p_aug),
        "redundancy": redundancy_check(Xtr_b, ntr, base_names),
        "lasso": lasso_survival(Xtr_b, ntr, ytr, feature_names),
        "perm_importance_new": perm_new,
    }
    if groups_te is not None:
        res["subgroup"] = subgroup_breakdown(yte, p_base, p_aug, groups_te)
    return res


def print_report(res: dict) -> None:
    """Human-readable summary of an evaluate_feature() result."""
    names = ", ".join(map(str, res["feature_names"]))
    mc = res["mcnemar"]
    sig = "SIGNIFICANT" if mc["pvalue"] < 0.05 else "not significant"
    print("=" * 68)
    print(f"Feature(s): {names}   ({res['n_new_features']} column(s))")
    print("-" * 68)
    print(f"accuracy   base={res['acc_base']:.4f}  ->  aug={res['acc_aug']:.4f}"
          f"   (delta {res['delta_acc']:+.4f})")
    print(f"McNemar    p={mc['pvalue']:.4g}  [{sig}]   "
          f"base-only={mc['n_base_only']}  new-only={mc['n_new_only']}  "
          f"({mc['method']})")
    for nm, red, la, pi in zip(res["feature_names"], res["redundancy"],
                               res["lasso"], res["perm_importance_new"]):
        print(f"  - {nm}")
        print(f"      redundancy : max|corr|={red['max_abs_corr']} "
              f"(vs {red['closest_base_feature']})  VIF={red['vif']}  "
              f"-> {red['verdict']}")
        print(f"      LASSO      : coef={la['lasso_coef']}  "
              f"survives={la['survives_lasso']}")
        print(f"      perm. imp. : {pi}")
    if "subgroup" in res:
        print("-" * 68)
        print("Subgroup lift (sorted by delta):")
        print(res["subgroup"].to_string(index=False))
    print("=" * 68)


if __name__ == "__main__":
    # Smoke test on synthetic data with the real shapes (no Kaggle data required).
    rng = np.random.default_rng(SEED)
    n_tr, n_val, n_te, d = 20000, 2000, 2000, 13

    def make(n):
        X = rng.normal(size=(n, d))
        signal = 0.8 * X[:, 0] - 0.6 * X[:, 3]        # depends on 2 base features
        strong_new = signal + rng.normal(scale=0.5, size=n)  # useful new feature
        redundant_new = X[:, 0] + rng.normal(scale=0.01, size=n)  # ~ base feature 0
        y = (signal + 0.9 * (strong_new - signal) + rng.normal(size=n) > 0).astype(int)
        return X, y, strong_new, redundant_new

    Xtr, ytr, s_tr, r_tr = make(n_tr)
    Xval, yval, s_val, r_val = make(n_val)
    Xte, yte, s_te, r_te = make(n_te)
    groups_te = rng.choice(["1960s", "1980s", "2000s", "2010s"], size=n_te)

    base = {"X_tr": Xtr, "X_val": Xval, "X_te": Xte,
            "y_tr": ytr, "y_val": yval, "y_te": yte,
            "feature_names": BASE_FEATURES_A}

    freeze_baseline(base)
    print("\n### useful (orthogonal) feature ###")
    print_report(evaluate_feature(base, s_tr, s_val, s_te,
                                  feature_names=["strong_new"], groups_te=groups_te))
    print("\n### redundant feature (should be flagged) ###")
    print_report(evaluate_feature(base, r_tr, r_val, r_te,
                                  feature_names=["redundant_new"]))
