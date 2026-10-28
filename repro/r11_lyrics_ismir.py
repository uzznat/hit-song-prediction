"""
R11 -- lyrics on the ISMIR 2019 dataset (musiXmatch bag-of-words, the official MSD lyrics release).

Only songs WITH lyrics are used on both sides (56 % of hits vs 36 % of the non-hit pool have lyrics, so
"has lyrics" would itself be a shortcut). Protocols A (random non-hits) and B (year-matched), as in R4, and
an English-only control (non-English lyrics are much more common among non-hits).

Lyrics features:
  * lexical (8): words, distinct words, richness, repetition, share of I/me/my, you/your, love, English
    stop words -- computed from the musiXmatch top-5000-word counts;
  * bag of words (1): TF-IDF + logistic regression trained out-of-fold inside each training fold, its
    probability used as one feature (same scheme as the title feature of R1).
Reference = our final ISMIR model (592 Essentia + year + artist history), Random Forest 200, 5x2 CV.

    py repro/r11_lyrics_ismir.py [grouped]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from common import DATA, ROOT, SEED, corrected_ttest, make_pipeline, save, show
from r4_external import rf200

MXM = os.path.join(DATA, "musixmatch")
EN_STOP = ["the", "i", "you", "and", "to", "a", "me", "my", "it", "in", "of", "is", "that", "your", "on",
           "be", "we", "all", "love", "do", "so", "no", "for", "what", "with", "can", "this", "but", "don't"]


def load_mxm(ids):
    ids = set(ids)
    vocab, rows, cols, vals, tids = None, [], [], [], []
    for f in ["mxm_dataset_train.txt", "mxm_dataset_test.txt"]:
        for line in open(os.path.join(MXM, f), encoding="utf-8"):
            if line[0] == "#":
                continue
            if line[0] == "%":
                vocab = line[1:].strip().split(",")
                continue
            parts = line.strip().split(",")
            if parts[0] not in ids:
                continue
            r = len(tids)
            tids.append(parts[0])
            for wc in parts[2:]:
                w, c = wc.split(":")
                rows.append(r)
                cols.append(int(w) - 1)
                vals.append(int(c))
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(len(tids), len(vocab)), dtype=np.float32)
    return tids, vocab, M


def lexical(M, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    tot = np.asarray(M.sum(axis=1)).ravel()
    uniq = np.asarray((M > 0).sum(axis=1)).ravel()
    mx = np.asarray(M.max(axis=1).todense()).ravel()
    def share(words):
        cols = [idx[w] for w in words if w in idx]
        return np.asarray(M[:, cols].sum(axis=1)).ravel() / np.maximum(tot, 1)
    return pd.DataFrame({"lyr_words": np.log1p(tot), "lyr_unique": np.log1p(uniq),
                         "lyr_richness": uniq / np.maximum(tot, 1), "lyr_repetition": mx / np.maximum(tot, 1),
                         "lyr_i": share(["i", "me", "my"]), "lyr_you": share(["you", "your"]),
                         "lyr_love": share(["love"]), "lyr_english": share(EN_STOP)})


LEX = ["lyr_words", "lyr_unique", "lyr_richness", "lyr_repetition", "lyr_i", "lyr_you", "lyr_love", "lyr_english"]


def bow_oof(B, y, tr, te):
    """Logistic regression on TF-IDF lyrics: out-of-fold probabilities on tr, fitted probabilities on te."""
    tf = TfidfTransformer(sublinear_tf=True).fit(B[tr])
    Xtr, Xte = tf.transform(B[tr]), tf.transform(B[te])
    lr = LogisticRegression(C=1.0, max_iter=2000)
    oof = np.zeros(len(tr))
    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y[tr]):
        oof[ite] = clone(lr).fit(Xtr[itr], y[tr][itr]).predict_proba(Xtr[ite])[:, 1]
    return oof, clone(lr).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]


def run(tag, s, B, audio, groups=None, configs_keep=None):
    y = s["target"].to_numpy()
    final = audio + ["year_n", "artist_prior_hits_log"]
    configs = {"audio (592)": (audio, False),
               "final: audio + year + artist": (final, False),
               "final + lexical (8)": (final + LEX, False),
               "final + bag of words": (final, True),
               "final + lexical + bag of words": (final + LEX, True),
               "lyrics only (lexical + bag of words)": (LEX, True)}
    if configs_keep:
        configs = {k: v for k, v in configs.items() if k in configs_keep}
    if groups is None:
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(np.zeros(len(y)), y))
    else:   # no artist appears in both training and test folds
        from sklearn.model_selection import StratifiedGroupKFold
        splits = [sp for r in range(2) for sp in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r)
                  .split(np.zeros(len(y)), y, groups)]
    rows = []
    for f, (tr, te) in enumerate(splits):
        oof, pte = bow_oof(B, y, tr, te)
        for name, (cols, use_bow) in configs.items():
            Xtr, Xte = s.iloc[tr][cols].copy(), s.iloc[te][cols].copy()
            if use_bow:
                Xtr["lyr_bow"], Xte["lyr_bow"] = oof, pte
            m = make_pipeline(list(Xtr.columns), model=rf200()).fit(Xtr, y[tr])
            sc = m.predict_proba(Xte)[:, 1]
            rows.append({"set": tag, "fold": f, "config": name, "n_train": len(tr), "n_test": len(te),
                         "accuracy": accuracy_score(y[te], (sc >= .5).astype(int)), "auc": roc_auc_score(y[te], sc)})
        print(f"  [{tag}] fold {f + 1}/{len(splits)}", flush=True)
    fd = pd.DataFrame(rows)
    ref = fd[fd.config == "final: audio + year + artist"].set_index("fold")
    out = []
    for name in configs:
        g = fd[fd.config == name].set_index("fold")
        r = {"set": tag, "n": len(y), "config": name, "acc": round(g.accuracy.mean(), 4), "auc": round(g.auc.mean(), 4)}
        if name != "final: audio + year + artist":
            for m in ["accuracy", "auc"]:
                d = g[m] - ref[m]
                r[f"d_{m}"] = round(d.mean(), 4)
                r[f"p_{m}"] = corrected_ttest(d.values, g.n_train.mean(), g.n_test.mean())[1]
            r["wins_acc"] = int(((g.accuracy - ref.accuracy) > 0).sum())
        out.append(r)
    return pd.DataFrame(out), fd


def main():
    sys.path.insert(0, ROOT)
    import run_ismir_v2 as ri
    df = ri.add_our_features(ri.load_labels())
    af = pd.read_pickle(ri.CACHE)
    data = df.merge(af, left_on="msd_id", right_index=True, how="inner")
    audio = [c for c in af.columns if data[c].notna().mean() > 0.95]
    data = data.dropna(subset=audio).reset_index(drop=True)
    tids, vocab, M = load_mxm(data["msd_id"])
    print(f"ISMIR songs with audio: {len(data):,} (hits {int(data.target.sum()):,}); with lyrics: {len(tids):,}")
    print(f"lyrics coverage: hits {data[data.target == 1].msd_id.isin(tids).mean():.3f}, "
          f"non-hits {data[data.target == 0].msd_id.isin(tids).mean():.3f}")
    lex = lexical(M, vocab)
    lex["msd_id"] = tids
    data = data.merge(lex, on="msd_id", how="inner").reset_index(drop=True)
    row_of = {t: i for i, t in enumerate(tids)}
    eng = data["lyr_english"] >= 0.25
    print(f"English share (>= 25 % common English words): hits {eng[data.target == 1].mean():.3f}, "
          f"non-hits {eng[data.target == 0].mean():.3f}")
    summaries, folds = [], []
    if "words" in sys.argv[1:]:     # what the lyrics model reads: year-matched, English-only sample
        s = ri.balanced_sample(data[eng], True, None).reset_index(drop=True)
        B = M[[row_of[t] for t in s["msd_id"]]]
        X = TfidfTransformer(sublinear_tf=True).fit_transform(B)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, s["target"])
        coef = pd.Series(lr.coef_[0], index=vocab)
        df_w = (B > 0).sum(axis=0).A1
        coef = coef[df_w >= 30]                          # words used in at least 30 songs
        top = pd.concat([coef.nlargest(15).rename("coef").to_frame().assign(side="hit"),
                         coef.nsmallest(15).rename("coef").to_frame().assign(side="non-hit")])
        show(top.reset_index().rename(columns={"index": "word"}))
        save(top.reset_index().rename(columns={"index": "word"}), "r11_lyrics_words.csv")
        lexm = s.groupby("target")[LEX].mean().T.rename(columns={0: "non_hits", 1: "hits"}).round(4)
        show(lexm)
        save(lexm.reset_index().rename(columns={"index": "feature"}), "r11_lyrics_lexical_means.csv")
        return
    if "grouped" in sys.argv[1:]:
        from feature_experiments import normalize_artist
        for name, matched in [("A_random_english_grouped", False), ("B_yearmatched_english_grouped", True)]:
            s = ri.balanced_sample(data[eng], matched, None).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
            s["year_n"] = s["year"].astype(float)
            B = M[[row_of[t] for t in s["msd_id"]]]
            g = s["artist"].astype(str).map(normalize_artist).to_numpy()
            summ, fd = run(name, s, B, audio, groups=g, configs_keep=["final: audio + year + artist",
                                                                       "final + lexical + bag of words",
                                                                       "lyrics only (lexical + bag of words)"])
            show(summ)
            summaries.append(summ)
        save(pd.concat(summaries, ignore_index=True), "r11_lyrics_ismir_grouped_summary.csv")
        return
    for name, matched, only_en in [("A_random", False, False), ("B_yearmatched", True, False),
                                   ("A_random_english", False, True), ("B_yearmatched_english", True, True)]:
        base = data[eng] if only_en else data
        s = ri.balanced_sample(base, matched, None).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        s["year_n"] = s["year"].astype(float)
        B = M[[row_of[t] for t in s["msd_id"]]]
        summ, fd = run(name, s, B, audio)
        show(summ)
        summaries.append(summ)
        folds.append(fd)
    res = pd.concat(summaries, ignore_index=True)
    print("\n=== R11 lyrics on ISMIR ===")
    show(res)
    save(res, "r11_lyrics_ismir_summary.csv")
    save(pd.concat(folds, ignore_index=True), "r11_lyrics_ismir_folds.csv")


if __name__ == "__main__":
    main()
