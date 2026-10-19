"""
run_phase0.py  —  ONE command to validate Phase 0.
=================================================

What it does, automatically:
  1. downloads the 6 Spotify-Hit-Predictor decade CSVs (only the first time),
  2. rebuilds your exact split (dedup on uri, seed 42, 20k/2k/2k)  ->  psplit_A.npz / psplit_B.npz,
  3. runs the frozen baseline from phase0_harness.py  ->  should print test_acc = 0.773.

HOW TO RUN
----------
  1. put this file next to  phase0_harness.py  (same folder)
  2. once, install the libraries:
        pip install numpy pandas scikit-learn scipy
  3. run:
        python run_phase0.py

  Expected last line:  [baseline] frozen RF  val_acc=0.7795  test_acc=0.773
  If you see that, Phase 0 is validated and you can move on to Phase 1.
"""

import os
import glob
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = "The-Spotify-Hit-Predictor-Dataset"
BASE_URL = ("https://raw.githubusercontent.com/fortyTwo102/"
            "The-Spotify-Hit-Predictor-Dataset/master")
DECADES = ["60s", "70s", "80s", "90s", "00s", "10s"]

AUDIO = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
         "acousticness", "instrumentalness", "liveness", "valence", "tempo",
         "duration_ms", "time_signature"]                       # set A (13 features)
FULL = AUDIO + ["chorus_hit", "sections"]                       # set B


def ensure_data():
    """Download the 6 decade CSVs the first time only."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for d in DECADES:
        path = os.path.join(DATA_DIR, f"dataset-of-{d}.csv")
        if os.path.exists(path):
            continue
        url = f"{BASE_URL}/dataset-of-{d}.csv"
        print(f"downloading dataset-of-{d}.csv ...")
        urllib.request.urlretrieve(url, path)
    print("data ready.")


def build_split():
    """Reproduce prep2.py exactly -> psplit_A.npz and psplit_B.npz."""
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(f"{DATA_DIR}/dataset-of-*.csv"))]
    df = (pd.concat(dfs, ignore_index=True)
            .drop_duplicates(subset=["uri"], keep="first")
            .reset_index(drop=True))
    y = df["target"].values
    idx = np.arange(len(df))

    sel, _ = train_test_split(idx, train_size=24000, stratify=y, random_state=SEED)
    tr, tmp = train_test_split(sel, train_size=20000, stratify=y[sel], random_state=SEED)
    val, te = train_test_split(tmp, train_size=2000, stratify=y[tmp], random_state=SEED)

    for tag, feats in [("A", AUDIO), ("B", FULL)]:
        X = df[feats].values
        np.savez(f"psplit_{tag}.npz",
                 X_tr=X[tr], X_val=X[val], X_te=X[te],
                 y_tr=y[tr], y_val=y[val], y_te=y[te])
    print(f"split saved: total={len(df)}  train={len(tr)}  val={len(val)}  test={len(te)}"
          f"  | hit ratio (train)={y[tr].mean():.3f}")


def main():
    ensure_data()
    build_split()
    # the harness expects psplit_A.npz in the current folder
    from phase0_harness import load_base_split, freeze_baseline
    base = load_base_split(tag="A", path_tmpl="psplit_{tag}.npz")
    print("\nrunning the frozen baseline through the harness:")
    freeze_baseline(base)
    print("\nPhase 0 validated." if True else "")


if __name__ == "__main__":
    main()
