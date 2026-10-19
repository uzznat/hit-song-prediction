"""
R2 -- is the gain the features or the model?  And does it hold on future songs?

  * 5 classifiers (RF, XGBoost, logistic regression, k-NN, RBF-SVM), each with
    the 13 audio features vs + decade + artist  (5-fold CV, corrected t-test)
  * RF vs XGBoost on the canonical split (McNemar)
  * TreeSHAP attributions from XGBoost (xgboost's native pred_contribs)
  * temporal hold-out: train on all EARLIER decades, test on 2000s / 2010s

    py repro/r2_models_time.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import (AUDIO, ART, DEC_COLS, SEED, FrozenRF, canonical_eval, canonical_split,
                    cv_eval, load_kaggle, make_pipeline, mcnemar_test, save, show)

OURS = AUDIO + DEC_COLS + [ART]


def scaled(clf):
    return Pipeline([("sc", StandardScaler()), ("clf", clf)])


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         random_state=SEED, n_jobs=-1)


MODELS = {
    "RF":      lambda: FrozenRF(),
    "XGBoost": xgb,
    "LogReg":  lambda: scaled(LogisticRegression(max_iter=3000)),
    "kNN":     lambda: scaled(KNeighborsClassifier(n_neighbors=25)),
    "SVM-RBF": lambda: scaled(SVC(C=1.0, gamma="scale", probability=False)),
}


def main():
    df = load_kaggle()
    y = df["target"].values

    # ---------------------------------------------------------------- 5 models
    rows = []
    for mname, factory in MODELS.items():
        cfg = {f"{mname} | audio": make_pipeline(AUDIO, model=factory()),
               f"{mname} | + decade + artist": make_pipeline(OURS, model=factory())}
        summ, folds = cv_eval(cfg, df, y, n_repeats=1, label=f"R2 {mname}")
        show(summ)
        summ.insert(0, "model", mname)
        rows.append(summ)
        save(folds, f"r2_models_folds_{mname}.csv")
    allm = pd.concat(rows, ignore_index=True)
    save(allm, "r2_models_cv_summary.csv")

    # ---------------------------------------------------------------- RF vs XGB, canonical
    tr, _, te = canonical_split(df)
    can = canonical_eval({"RF | + decade + artist": make_pipeline(OURS),
                          "XGBoost | + decade + artist": make_pipeline(OURS, model=xgb()),
                          "XGBoost | audio": make_pipeline(AUDIO, model=xgb())},
                         df, y, tr, te)
    print("\n=== RF vs XGBoost, canonical split ===")
    show(can)
    save(can, "r2_rf_vs_xgb_canonical.csv")

    # ---------------------------------------------------------------- TreeSHAP
    m = xgb().fit(df[OURS].values[tr], y[tr])
    contrib = m.get_booster().predict(
        __import__("xgboost").DMatrix(df[OURS].values[te], feature_names=OURS),
        pred_contribs=True)[:, :-1]                    # last column = bias
    shap = pd.Series(np.abs(contrib).mean(0), index=OURS)
    agg = shap[AUDIO + [ART]].to_dict()
    agg["decade (6 one-hot, summed)"] = shap[DEC_COLS].sum()
    shap_df = (pd.Series(agg).sort_values(ascending=False).round(4)
                 .reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}))
    print("\n=== TreeSHAP (XGBoost, canonical test set) ===")
    show(shap_df)
    save(shap_df, "r2_shap.csv")

    # ---------------------------------------------------------------- temporal hold-out
    order = ["60s", "70s", "80s", "90s", "00s", "10s"]
    trows = []
    for target_dec in ("90s", "00s", "10s"):
        past = order[:order.index(target_dec)]
        trm = df["decade"].isin(past).values
        tem = (df["decade"] == target_dec).values
        preds = {}
        for fname, cols in [("audio", AUDIO), ("audio + artist", AUDIO + [ART])]:
            for mname, factory in [("RF", FrozenRF), ("XGBoost", xgb)]:
                mdl = factory().fit(df.loc[trm, cols].values, y[trm])
                p = mdl.predict(df.loc[tem, cols].values)
                s = mdl.predict_proba(df.loc[tem, cols].values)[:, 1]
                preds[(fname, mname)] = p
                trows.append({"test_decade": target_dec, "train_decades": "+".join(past),
                              "n_train": int(trm.sum()), "n_test": int(tem.sum()),
                              "features": fname, "model": mname,
                              "accuracy": round(accuracy_score(y[tem], p), 4),
                              "roc_auc": round(roc_auc_score(y[tem], s), 4)})
        for mname in ("RF", "XGBoost"):
            mc = mcnemar_test(y[tem], preds[("audio", mname)], preds[("audio + artist", mname)])
            for r in trows:
                if (r["test_decade"] == target_dec and r["model"] == mname
                        and r["features"] == "audio + artist"):
                    r["mcnemar_p_artist_vs_audio"] = mc["pvalue"]
    tdf = pd.DataFrame(trows)
    print("\n=== temporal hold-out (train on earlier decades only) ===")
    show(tdf)
    save(tdf, "r2_temporal.csv")


if __name__ == "__main__":
    main()
