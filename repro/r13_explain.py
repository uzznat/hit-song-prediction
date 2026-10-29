"""
R13 -- material to explain the algorithm to a reader who has never seen the project.

  * one real song followed through the whole pipeline: its 20 inputs, the path it takes in one tree of the
    frozen Random Forest, the votes of the 80 trees and the final probability (the song is in the test part);
  * what the decade feature is for: hits and non-hits per decade for a few descriptors, which shows that the
    audio of a typical hit changes from decade to decade.

    py repro/r13_explain.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import ART, AUDIO, DEC_COLS, DEC_ORDER, SEED, FrozenRF, load_kaggle, save, show

FINAL = AUDIO + DEC_COLS + [ART]


def main():
    df = load_kaggle()
    y = df["target"].to_numpy()
    ex = df.index[(df.track == "Vogue") & (df.na == "madonna")][0]
    idx = np.arange(len(df))
    tr, te = train_test_split(np.delete(idx, ex), test_size=0.2, stratify=np.delete(y, ex), random_state=SEED)
    te = np.append(te, ex)                            # the example song is never seen in training
    m = FrozenRF().fit(df.loc[tr, FINAL], y[tr])
    x = df.loc[[ex], FINAL]
    votes = np.array([t.predict_proba(x.to_numpy())[0, 1] for t in m.estimators_])
    p = m.predict_proba(x)[0, 1]
    acc = (m.predict(df.loc[te, FINAL]) == y[te]).mean()
    print(f"example: {df.loc[ex, 'track']} / {df.loc[ex, 'artist']} ({df.loc[ex, 'decade']}), label {y[ex]}")
    print(f"P(hit) = {p:.3f}; trees voting hit (leaf share >= .5): {(votes >= .5).sum()}/{len(votes)}; "
          f"test accuracy of this model {acc:.4f}")
    inputs = x.T.reset_index().rename(columns={"index": "input", ex: "value"})
    show(inputs)
    save(inputs, "r13_example_inputs.csv")

    # the path of the example in the first tree, with the share of training hits at each node
    t = m.estimators_[0].tree_
    node, rows = 0, []
    xv = x.to_numpy(float)[0]
    while t.children_left[node] != -1:
        f, thr = t.feature[node], t.threshold[node]
        go_left = xv[f] <= thr
        v = t.value[node][0]
        rows.append({"depth": len(rows), "question": f"{FINAL[f]} <= {thr:.3g}", "value": round(xv[f], 4),
                     "answer": "yes" if go_left else "no", "n_songs": int(t.n_node_samples[node]),
                     "hit_share": round(v[1] / v.sum(), 3)})
        node = t.children_left[node] if go_left else t.children_right[node]
    v = t.value[node][0]
    rows.append({"depth": len(rows), "question": "leaf", "value": None, "answer": "",
                 "n_songs": int(t.n_node_samples[node]), "hit_share": round(v[1] / v.sum(), 3)})
    path = pd.DataFrame(rows)
    show(path)
    save(path, "r13_example_path.csv")
    save(pd.DataFrame({"p_hit": [round(p, 4)], "trees_hit": [int((votes >= .5).sum())], "n_trees": [len(votes)],
                       "test_accuracy": [round(acc, 4)], "depth_tree0": [len(rows) - 1]}),
         "r13_example_summary.csv")

    # what the decade is for: typical hit vs non-hit audio in each decade
    rows = []
    for d in DEC_ORDER:
        g = df[df.decade == d]
        for c in ["loudness", "acousticness", "danceability", "instrumentalness", "duration_ms"]:
            rows.append({"decade": d, "descriptor": c, "hits": round(g.loc[g.target == 1, c].median(), 4),
                         "non_hits": round(g.loc[g.target == 0, c].median(), 4), "n": len(g),
                         "hit_share": round(g.target.mean(), 4)})
    dec = pd.DataFrame(rows)
    show(dec)
    save(dec, "r13_decade_medians.csv")


if __name__ == "__main__":
    main()
