"""
external_validation_ismir.py
============================================================================
Test OUR artist feature on an EXTERNAL, peer-reviewed dataset:

  Zangerle, Vötter, Huber, Yang.
  "Hit Song Prediction: Leveraging Low- and High-Level Audio Features."
  ISMIR 2019.   (Zenodo: 10.5281/zenodo.3258042, CC-BY)

WHY this dataset: it already ships Essentia audio features (no audio to
re-download), plus artist name + title + Billboard info. That lets us add
OUR leakage-free "artist past hits" feature and see if it still helps on a
completely different dataset built by other researchers.

------------------------------------------------------------------------
WHAT TO DOWNLOAD FIRST (once, on your PC), from
https://zenodo.org/records/3258042 :
  1. msd_bb_matches.csv          (hits;      ~0.5 MB)
  2. msd_bb_non_matches.csv      (non-hits;  ~7 MB)
  3. msd_audio_features.tar.gz   (18.6 GB) -> extract it next to this script
     so you get a folder  msd_audio_features/  with json files.
Put all of them in the SAME folder as this script.

RUN:
  py external_validation_ismir.py
------------------------------------------------------------------------
"""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

SEED = 42
AUDIO_DIR = "msd_audio_features"     # extracted from the .tar.gz
MATCHES = "msd_bb_matches.csv"
NON_MATCHES = "msd_bb_non_matches.csv"


# ---------- 1. labels + artist/title/year ----------
def load_labels():
    hits = pd.read_csv(MATCHES)
    flops = pd.read_csv(NON_MATCHES)
    # normalise column names (the files use these headers)
    def norm(df, target):
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        # expected columns: msd_id, echo_nest_id, artist, title, year, (peak, weeks)
        keep = {}
        for cand in ["msd_id", "msdid", "track_id"]:
            if cand in df.columns: keep["msd_id"] = df[cand]; break
        for cand in ["artist", "artist_name"]:
            if cand in df.columns: keep["artist"] = df[cand]; break
        for cand in ["title", "track_title", "song"]:
            if cand in df.columns: keep["title"] = df[cand]; break
        for cand in ["year", "release_year"]:
            if cand in df.columns: keep["year"] = df[cand]; break
        out = pd.DataFrame(keep); out["target"] = target
        return out
    df = pd.concat([norm(hits, 1), norm(flops, 0)], ignore_index=True)
    df = df.dropna(subset=["msd_id", "artist", "year"]).reset_index(drop=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"] > 0].reset_index(drop=True)
    return df


# ---------- 2. OUR artist feature (leakage-free) ----------
def add_artist_feature(df):
    """For each song: how many hits that artist had in EARLIER years only."""
    df = df.sort_values("year").reset_index(drop=True)
    df["artist_norm"] = df["artist"].astype(str).str.lower().str.strip()
    prior = {}
    counts = []
    # process year by year so we never look into the future
    for yr in sorted(df["year"].unique()):
        rows = df[df["year"] == yr]
        for _, r in rows.iterrows():
            counts.append((r.name, prior.get(r["artist_norm"], 0)))
        # after scoring this year, add this year's hits to the history
        for _, r in rows[rows["target"] == 1].iterrows():
            prior[r["artist_norm"]] = prior.get(r["artist_norm"], 0) + 1
    s = pd.Series(dict(counts))
    df["artist_prior_hits"] = s.reindex(df.index).fillna(0).values
    df["artist_prior_hits_log"] = np.log1p(df["artist_prior_hits"])
    return df


# ---------- 3. their Essentia audio features ----------
def load_audio_features(msd_ids):
    """Read the low-level Essentia json for each track; flatten scalar values."""
    feats = {}
    wanted = set(msd_ids)
    files = glob.glob(os.path.join(AUDIO_DIR, "**", "*.json"), recursive=True)
    print(f"  scanning {len(files)} json files...")
    for f in files:
        mid = os.path.basename(f).split(".")[0]
        # some files are named <id>.lowlevel.json etc.
        mid = mid.split("_")[0]
        if mid not in wanted:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        flat = {}
        def walk(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(obj, (int, float)):
                flat[prefix] = obj
            elif isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                # summarise vectors by mean (matches our 'summary feature' idea)
                flat[prefix + ".mean"] = float(np.mean(obj))
        walk("", d)
        feats.setdefault(mid, {}).update(flat)
    af = pd.DataFrame.from_dict(feats, orient="index")
    af.index.name = "msd_id"
    return af.reset_index()


def main():
    print("1) loading labels...")
    df = load_labels()
    print(f"   {len(df)} songs ({df.target.sum()} hits / {(df.target==0).sum()} flops)")

    print("2) building our artist feature (leakage-free)...")
    df = add_artist_feature(df)

    print("3) loading their Essentia audio features (this is the slow part)...")
    af = load_audio_features(df["msd_id"].tolist())
    print(f"   matched audio for {len(af)} tracks")
    data = df.merge(af, on="msd_id", how="inner")
    audio_cols = [c for c in af.columns if c != "msd_id"]
    data = data.dropna(subset=audio_cols).reset_index(drop=True)
    # keep only numeric audio columns present for (almost) everyone
    good = [c for c in audio_cols if data[c].notna().mean() > 0.95]
    print(f"   usable audio features: {len(good)}   final songs: {len(data)}")

    y = data["target"].values
    idx = np.arange(len(data))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)

    def run(X):
        m = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                   random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        return accuracy_score(y[te], m.predict(X[te]))

    Xaudio = data[good].values
    Xours = np.column_stack([Xaudio, data["artist_prior_hits_log"].values])

    a1 = run(Xaudio)
    a2 = run(Xours)
    print("\n================= RESULT (external ISMIR dataset) =================")
    print(f"  their audio features only          : {a1:.4f}")
    print(f"  + OUR artist feature               : {a2:.4f}")
    print(f"  gain from our artist feature        : {a2-a1:+.4f}")
    print("===================================================================")
    print("If the gain is positive, our artist feature helps even on a")
    print("different, peer-reviewed dataset with different audio features.")


if __name__ == "__main__":
    main()
