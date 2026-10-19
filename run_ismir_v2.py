"""
run_ismir_v2.py  --  external validation #3 (the only peer-reviewed dataset)
============================================================================
  Zangerle, Voetter, Huber, Yang. "Hit Song Prediction: Leveraging Low- and
  High-Level Audio Features." ISMIR 2019.  Zenodo 10.5281/zenodo.3258042

Fixes vs run_ismir.py (which was never run):
  1. PATH   : the tarball extracts to features_tracks_<c>/ where <c> is the
              3rd character of the MSD id (TRM... -> features_tracks_m), with
              <id>.lowlevel.json + <id>.mp3.highlevel.json. We build the paths
              directly instead of globbing ~1.7 M files.
  2. BALANCE: 5.8k hits vs 89k non-hits -> "always flop" already scores ~94 %.
              Accuracy is only meaningful on a balanced sample (as on Kaggle,
              which is 50/50). We also report ROC-AUC.
  3. BOTH OF OUR FEATURES, and a LEAK CHECK on the decade. Our headline claim
     is decade + artist_prior_hits. The song_title lesson says: a feature can
     detect how a dataset was BUILT. If hits and non-hits were sampled from
     different years, the decade separates them for a bad reason. So:
       protocol A = balanced, random non-hits
       protocol B = balanced, non-hits matched YEAR BY YEAR to the hits
                    (decade carries zero information by construction; any
                    artist gain in B is not a date artifact)
  4. METADATA EXCLUDED: bit_rate, replay_gain, codec, md5... describe the
     FILE, not the music.

Same classifier as the two other external validations: RF(200, sqrt),
stratified 75/25 split, seed 42, McNemar vs their audio features.

RUN:  py run_ismir_v2.py          (feature extraction cached in ext_cache/)
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from phase0_harness import mcnemar_test

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ext_cache", "ismir_essentia.pkl")
OUT_CSV = os.path.join(HERE, "ismir_results.csv")


# ---------------------------------------------------------------- labels
def load_labels():
    hits = pd.read_csv(os.path.join(HERE, "msd_bb_matches.csv"))
    flops = pd.read_csv(os.path.join(HERE, "msd_bb_non_matches.csv"))
    hits["target"], flops["target"] = 1, 0
    cols = ["msd_id", "artist", "title", "year", "target"]
    df = pd.concat([hits[cols], flops[cols]], ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["msd_id", "artist", "year"])
    df = df[df["year"] > 0].drop_duplicates("msd_id").reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    print(f"   {n0} rows -> {len(df)} with a valid year "
          f"({df.target.sum()} hits / {(df.target == 0).sum()} non-hits)")
    return df


def add_our_features(df):
    """decade, and artist_prior_hits = hits by the same artist in STRICTLY
    EARLIER years (never the current year, never the future)."""
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    df["artist_norm"] = df["artist"].astype(str).str.lower().str.strip()
    per = (df.groupby(["artist_norm", "year"])["target"].sum()
             .rename("h").reset_index().sort_values(["artist_norm", "year"]))
    per["prior"] = per.groupby("artist_norm")["h"].cumsum() - per["h"]
    df = df.merge(per[["artist_norm", "year", "prior"]],
                  on=["artist_norm", "year"], how="left")
    df["artist_prior_hits_log"] = np.log1p(df["prior"].fillna(0))
    return df


# ---------------------------------------------------------------- audio
SKIP_TOP = {"metadata"}
SKIP_KEYS = {"beats_position"}          # positions in seconds, not a descriptor


def _flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            _flatten(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    elif isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
        out[prefix + ".mean"] = float(np.mean(obj))


def extract_one(msd_id):
    d = os.path.join(HERE, "features_tracks_" + msd_id[2].lower())
    flat = {}
    for suffix in (".lowlevel.json", ".mp3.highlevel.json"):
        try:
            with open(os.path.join(d, msd_id + suffix), encoding="utf-8") as f:
                js = json.load(f)
        except (OSError, ValueError):
            return msd_id, None                # need BOTH files
        for top, v in js.items():
            if top not in SKIP_TOP:
                _flatten(v, top, flat)
    return msd_id, flat


def load_audio(ids):
    if os.path.exists(CACHE):
        af = pd.read_pickle(CACHE)
        print(f"   cache: {len(af)} tracks, {af.shape[1]} columns")
        return af
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    feats, missing, t0 = {}, 0, time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        for i, (mid, flat) in enumerate(pool.imap_unordered(extract_one, ids,
                                                            chunksize=64), 1):
            if flat is None:
                missing += 1
            else:
                feats[mid] = flat
            if i % 5000 == 0:
                el = time.time() - t0
                print(f"   {i}/{len(ids)}  ({el:.0f} s, ~{el / i * (len(ids) - i):.0f} s left)",
                      flush=True)
    af = pd.DataFrame.from_dict(feats, orient="index").astype("float32")
    af.index.name = "msd_id"
    af.to_pickle(CACHE)
    print(f"   extracted {len(af)} tracks ({missing} missing a json), "
          f"{af.shape[1]} columns, {time.time() - t0:.0f} s")
    return af


# ---------------------------------------------------------------- experiment
def balanced_sample(data, year_matched, rng):
    hits = data[data.target == 1]
    non = data[data.target == 0]
    if not year_matched:
        n = min(len(hits), len(non))
        return pd.concat([hits.sample(n, random_state=SEED),
                          non.sample(n, random_state=SEED)])
    parts = []
    for yr, h in hits.groupby("year"):
        pool = non[non.year == yr]
        k = min(len(h), len(pool))
        if k == 0:
            continue
        parts += [h.sample(k, random_state=SEED), pool.sample(k, random_state=SEED)]
    return pd.concat(parts)


def run_protocol(name, s, audio_cols, rows):
    s = s.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    y = s["target"].values
    tr, te = train_test_split(np.arange(len(s)), test_size=0.25, stratify=y,
                              random_state=SEED)
    A = s[audio_cols].values
    DEC = s[["decade"]].values
    ART = s[["artist_prior_hits_log"]].values
    sets = {
        "their audio (Essentia)":         A,
        "+ decade":                       np.column_stack([A, DEC]),
        "+ artist":                       np.column_stack([A, ART]),
        "+ decade + artist":              np.column_stack([A, DEC, ART]),
        "our 2 features alone":           np.column_stack([DEC, ART]),
        "decade alone (leak check)":      DEC,
    }
    hit_share = s.groupby("decade")["target"].mean()
    print(f"\n=== protocol {name}: {len(s)} songs ({y.sum()} hits), "
          f"test {len(te)} ===")
    print("   hit share per decade: " +
          ", ".join(f"{int(d)}s {100 * v:.0f}%" for d, v in hit_share.items()))
    base_pred = None
    for label, X in sets.items():
        m = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                   random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        if base_pred is None:
            base_pred, base_acc = pred, acc
            mc = {"pvalue": float("nan"), "n_base_only": 0, "n_new_only": 0}
        else:
            mc = mcnemar_test(y[te], base_pred, pred)
        print(f"   {label:28s} acc {acc:.4f} ({acc - base_acc:+.4f})  AUC {auc:.4f}"
              + ("" if base_pred is pred else
                 f"  McNemar p={mc['pvalue']:.2g} "
                 f"(+{mc['n_new_only']}/-{mc['n_base_only']})"), flush=True)
        rows.append({"protocol": name, "features": label, "n_songs": len(s),
                     "n_test": len(te), "accuracy": round(acc, 4),
                     "delta_vs_audio": round(acc - base_acc, 4),
                     "roc_auc": round(auc, 4), "mcnemar_p": mc["pvalue"],
                     "new_right_only": mc["n_new_only"],
                     "base_right_only": mc["n_base_only"]})


def main():
    print("1) labels")
    df = add_our_features(load_labels())
    print("2) Essentia features (cached after first run)")
    af = load_audio(df["msd_id"].tolist())
    data = df.merge(af, left_on="msd_id", right_index=True, how="inner")
    audio_cols = [c for c in af.columns if data[c].notna().mean() > 0.95]
    data = data.dropna(subset=audio_cols).reset_index(drop=True)
    print(f"   {len(data)} songs with audio ({data.target.sum()} hits), "
          f"{len(audio_cols)} audio features")
    yr = data.groupby("target")["year"].describe()[["25%", "50%", "75%"]]
    print("   year quartiles (0 = non-hit, 1 = hit):\n" + yr.to_string())

    rng = np.random.default_rng(SEED)
    rows = []
    run_protocol("A balanced-random", balanced_sample(data, False, rng), audio_cols, rows)
    run_protocol("B balanced-year-matched", balanced_sample(data, True, rng), audio_cols, rows)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
