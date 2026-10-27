"""
R4 -- external validation of the artist (and decade) features on data built by
other teams, all with the same protocol: RF(200, sqrt) as in the original
external scripts, canonical 75/25 split (seed 42) + 5x5 repeated CV.

  1. ISMIR 2019 (Zangerle et al.)  -- Essentia features, cached by run_ismir_v2.py
       A random-balanced / B year-matched (decade carries no information)
  2. SpotGenTrack                  -- label = Spotify popularity >= 55
  3. Billboard year-end 2000-2023  -- only charted songs; task = top-40 vs 41-100

    py repro/r4_external.py [ismir] [spotgentrack] [billboard]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from common import DATA, ROOT, SEED, canonical_eval, cv_eval, make_pipeline, save, show

SPOTIFY_13 = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
              "acousticness", "instrumentalness", "liveness", "valence", "tempo",
              "duration_ms", "time_signature"]


def rf200():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  random_state=SEED, n_jobs=-1)


def prior_by_year(df, artist_col, year_col, hit_col):
    """Hits by the same artist in STRICTLY earlier years."""
    per = (df.groupby([artist_col, year_col])[hit_col].sum().rename("h").reset_index()
             .sort_values([artist_col, year_col]))
    per["prior"] = per.groupby(artist_col)["h"].cumsum() - per["h"]
    out = df.merge(per[[artist_col, year_col, "prior"]], on=[artist_col, year_col], how="left")
    return np.log1p(out["prior"].fillna(0).values)


def run_block(tag, data, y, configs, ref, n_repeats=5):
    idx = np.arange(len(data))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    can = canonical_eval(configs, data, y, tr, te, ref=ref)
    print(f"\n=== {tag}: canonical 75/25 ({len(tr)} / {len(te)}) ===")
    show(can)
    save(can, f"r4_{tag}_canonical.csv")
    summ, folds = cv_eval(configs, data, y, ref=ref, label=tag, n_repeats=n_repeats)
    print(f"\n=== {tag}: 5x{n_repeats} repeated CV ===")
    show(summ)
    save(summ, f"r4_{tag}_cv_summary.csv")
    save(folds, f"r4_{tag}_cv_folds.csv")


# --------------------------------------------------------------------------- ISMIR
def ismir():
    sys.path.insert(0, ROOT)
    import run_ismir_v2 as ri
    df = ri.add_our_features(ri.load_labels())
    af = pd.read_pickle(ri.CACHE)
    data = df.merge(af, left_on="msd_id", right_index=True, how="inner")
    audio = [c for c in af.columns if data[c].notna().mean() > 0.95]
    data = data.dropna(subset=audio).reset_index(drop=True)
    for name, matched in [("ismir_A_random", False), ("ismir_B_yearmatched", True)]:
        s = (ri.balanced_sample(data, matched, None)
               .sample(frac=1.0, random_state=SEED).reset_index(drop=True))
        s["decade_n"] = s["decade"].astype(float)
        s["year_n"] = s["year"].astype(float)
        y = s["target"].values
        P = lambda cols: make_pipeline(cols, model=rf200())
        # "+ year" mirrors Zangerle et al.'s best inputs (audio + release year,
        # 75.04 % with a wide-and-deep net, 5-fold CV on 11,664 balanced songs)
        cfg = {"their audio (592 Essentia)": P(audio),
               "+ decade": P(audio + ["decade_n"]),
               "+ year": P(audio + ["year_n"]),
               "+ artist": P(audio + ["artist_prior_hits_log"]),
               "+ decade + artist": P(audio + ["decade_n", "artist_prior_hits_log"]),
               "+ year + artist": P(audio + ["year_n", "artist_prior_hits_log"]),
               "decade alone": P(["decade_n"]),
               "year alone": P(["year_n"])}
        run_block(name, s, y, cfg, "their audio (592 Essentia)", n_repeats=2)


# --------------------------------------------------------------------------- SpotGenTrack
def load_spotgentrack():
    t = pd.read_csv(os.path.join(DATA, "spotify_tracks.csv"))
    alb = pd.read_csv(os.path.join(DATA, "spotify_albums.csv"), usecols=["id", "release_date"])
    art = pd.read_csv(os.path.join(DATA, "spotify_artists.csv"), usecols=["id", "name"])
    t["year"] = pd.to_datetime(t["album_id"].map(dict(zip(alb.id, alb.release_date))),
                               errors="coerce", format="mixed").dt.year
    amap = dict(zip(art.id.astype(str), art.name.astype(str)))
    first = t["artists_id"].astype(str).str.strip("[]").str.split(",").str[0].str.strip(" '\"")
    t["artist_norm"] = first.map(amap).fillna(first).str.lower().str.strip()
    t["target"] = (pd.to_numeric(t["popularity"], errors="coerce") >= 55).astype(int)
    low = pd.read_csv(os.path.join(DATA, "low_level_audio_features.csv"))
    lid = "Track_id" if "Track_id" in low.columns else next(c for c in low.columns
                                                          if c.lower() in ("track_id", "id"))
    rich = [c for c in low.columns if c != lid and not c.lower().startswith("unnamed")
            and low[c].dtype != object]
    data = t.merge(low[[lid] + rich], left_on="id", right_on=lid, how="inner")
    data = data.dropna(subset=SPOTIFY_13 + ["year"]).reset_index(drop=True)
    data["year"] = data["year"].astype(int)
    data["decade_n"] = (data["year"] // 10 * 10).astype(float)
    data["artist_prior_log"] = prior_by_year(data, "artist_norm", "year", "target")
    data[rich] = data[rich].apply(pd.to_numeric, errors="coerce").fillna(0)
    return data, rich


def spotgentrack():
    data, rich = load_spotgentrack()
    y = data["target"].values
    print(f"SpotGenTrack: {len(data)} tracks, hit share {y.mean():.3f}, "
          f"{len(rich)} low-level features")
    save(data.groupby("decade_n")["target"].agg(["size", "mean"]).reset_index(),
         "r4_spotgentrack_hit_share_by_decade.csv")
    P = lambda cols, pca=None: make_pipeline(cols, pca, model=rf200())
    cfg = {"their 13 Spotify": P(SPOTIFY_13),
           "+ artist": P(SPOTIFY_13 + ["artist_prior_log"]),
           "+ decade": P(SPOTIFY_13 + ["decade_n"]),
           "+ decade + artist": P(SPOTIFY_13 + ["decade_n", "artist_prior_log"]),
           "+ artist + low-level (PCA 40)": P(SPOTIFY_13 + ["artist_prior_log"], rich),
           "decade + artist only": P(["decade_n", "artist_prior_log"])}
    run_block("spotgentrack", data, y, cfg, "their 13 Spotify", n_repeats=1)


# --------------------------------------------------------------------------- Billboard year-end
def billboard():
    b = pd.read_csv(os.path.join(DATA, "billboard_24years_lyrics_spotify.csv"),
                    usecols=["ranking", "song", "band_singer", "year"] + SPOTIFY_13)
    b["artist_norm"] = (b["band_singer"].astype(str).str.lower()
                        .str.split(r"\s+(?:featuring|feat\.?|ft\.?|&|x|with|and)\s+").str[0]
                        .str.strip())
    b["key_song"] = b["artist_norm"] + "|" + b["song"].astype(str).str.lower().str.strip()
    n_rows = len(b)
    b = b.dropna(subset=SPOTIFY_13)
    g = b.sort_values("year").groupby("key_song")
    songs = g.first().reset_index()
    songs["best_rank"] = g["ranking"].min().values
    songs["target"] = (songs["best_rank"] <= 40).astype(int)
    songs["charted"] = 1
    songs["artist_prior_log"] = prior_by_year(songs, "artist_norm", "year", "charted")
    songs["artist_prior_top40_log"] = prior_by_year(songs, "artist_norm", "year", "target")
    y = songs["target"].values
    print(f"Billboard year-end: {n_rows} rows -> {len(songs)} unique songs with Spotify "
          f"features, top-40 share {y.mean():.3f}")
    P = lambda cols: make_pipeline(cols, model=rf200())
    cfg = {"their 13 Spotify": P(SPOTIFY_13),
           "+ artist (prior charted songs)": P(SPOTIFY_13 + ["artist_prior_log"]),
           "+ artist (prior top-40 songs)": P(SPOTIFY_13 + ["artist_prior_top40_log"])}
    run_block("billboard", songs, y, cfg, "their 13 Spotify")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["ismir", "spotgentrack", "billboard"]
    for name in todo:
        globals()[name]()
