"""
R8 -- the Future Work items of the paper that can be run on the data we have.

  artist   (i)   better artist resolution: how many HITS are found in the chart
                 history (should be ~100 %), and does a stricter/wider matching
                 (full name first, leading "the", featured credits) change the gain?
  genre    (ii)  does the FFT gain for new artists survive genre control?
                 Spotify artist genres (SpotGenTrack's spotify_artists.csv) are
                 attached by artist name; we (a) add genre families as features and
                 (b) re-evaluate on a genre-matched sample (equal hits and flops
                 inside every genre family).
  decay    (iii) rolling origin: train on ONE decade, test on every later decade,
                 to see how fast a model trained on the past decays.
  lyrics   (iv)  lyrics descriptors (SpotGenTrack lyrics_features.csv) on top of
                 the 13 Spotify descriptors and artist history.

    py repro/r8_future_work.py [artist|genre|decay|lyrics]   (no argument = all)
"""
import ast
import os
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (AUDIO, ART, DATA, DEC_COLS, DEC_ORDER, ROOT, SEED, FrozenRF, cv_eval,
                    load_kaggle, make_pipeline, save, show)

MAIN = AUDIO + DEC_COLS + [ART]


def groups_of(data, col="artist_prior_hits"):
    return (data[col] > 0).map({True: "known", False: "new"}).values


def load_fft(df):
    fft = pd.read_csv(os.path.join(ROOT, "cache", "fft_features_full.csv"))
    fcols = [c for c in fft.columns if c != "uri"]
    fft = fft.rename(columns={c: "fft_" + c for c in fcols})
    fcols = ["fft_" + c for c in fcols]
    data = df.merge(fft, on="uri", how="inner").dropna(subset=fcols).reset_index(drop=True)
    return data, fcols


# =========================================================================== (i) artist
_SEP = re.compile(r"\s+(?:featuring|feat\.?|ft\.?|with|&|x|\+|/|,|vs\.?|and)\s+", re.I)


def _clean(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    return s


def _norm_one(s):
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def credits(name):
    """(full name, [every credited artist]) -- normalised, leading 'the' removed."""
    if not isinstance(name, str):
        return "", []
    s = _clean(name)
    parts = [_norm_one(p) for p in _SEP.split(s)]
    return _norm_one(s), [p for p in parts if p]


def artist_variants(df):
    bb = pd.read_csv(os.path.join(DATA, "hot100.csv"), usecols=["chart_week", "title", "performer"])
    bb["dstart"] = pd.to_datetime(bb["chart_week"]).dt.year // 10 * 10
    full_h = defaultdict(lambda: defaultdict(set))     # full credited string
    prim_h = defaultdict(lambda: defaultdict(set))     # primary artist only
    any_h = defaultdict(lambda: defaultdict(set))      # every credited artist
    for perf, ds, ti in zip(bb["performer"], bb["dstart"], bb["title"]):
        full, parts = credits(perf)
        ti = str(ti).lower().strip()
        if full:
            full_h[full][ds].add(ti)
        if parts:
            prim_h[parts[0]][ds].add(ti)
        for p in set(parts):
            any_h[p][ds].add(ti)

    def prior(h, key, ds):
        return sum(len(v) for d, v in h[key].items() if d < ds) if key in h else 0

    def found(h, key, ds):          # artist in the chart history up to its own decade
        return key in h and any(d <= ds for d in h[key])

    cr = [credits(a) for a in df["artist"]]
    ds = df["dstart"].values
    # strict: full credited name if it charted, else primary artist (+ leading 'the' fix)
    strict, wide, f_strict, f_wide = [], [], [], []
    for (full, parts), d in zip(cr, ds):
        prim = parts[0] if parts else ""
        if full in full_h or full in prim_h:
            k_h = full_h if full in full_h else prim_h
            strict.append(prior(k_h, full, d)); f_strict.append(found(k_h, full, d))
        else:
            strict.append(prior(prim_h, prim, d)); f_strict.append(found(prim_h, prim, d))
        # wide: every credited artist, featured appearances included; max over credits
        wide.append(max([prior(any_h, p, d) for p in parts] or [0]))
        f_wide.append(any(found(any_h, p, d) for p in parts))
    df["art_strict"] = np.log1p(strict)
    df["art_wide"] = np.log1p(wide)
    df["art_wide_raw"] = wide
    # original feature's "found" flag, same definition
    from feature_experiments import normalize_artist
    hist0 = defaultdict(set)
    for perf, d in zip(bb["performer"], bb["dstart"]):
        hist0[normalize_artist(perf)].add(d)
    df["found_orig"] = [na in hist0 and any(x <= d for x in hist0[na]) for na, d in zip(df["na"], ds)]
    df["found_strict"] = f_strict
    df["found_wide"] = f_wide
    return df


def run_artist():
    df = artist_variants(load_kaggle())
    y = df["target"].values
    facts = []
    for v in ("orig", "strict", "wide"):
        f = df[f"found_{v}"]
        facts.append({"variant": v, "found_all": round(f.mean(), 4),
                      "found_hits": round(f[y == 1].mean(), 4),
                      "found_flops": round(f[y == 0].mean(), 4)})
    nz = {"orig": (df.artist_prior_hits > 0), "strict": df.art_strict > 0, "wide": df.art_wide > 0}
    for r in facts:
        m = nz[r["variant"]]
        r.update(prior_gt0_all=round(m.mean(), 4), prior_gt0_hits=round(m[y == 1].mean(), 4),
                 prior_gt0_flops=round(m[y == 0].mean(), 4))
    facts = pd.DataFrame(facts)
    print("\n=== R8 artist resolution: share of songs found in the chart history ===")
    show(facts)
    save(facts, "r8_artist_facts.csv")
    # hits whose artist is NOT found although they are Hot 100 hits of that decade
    miss = df[(y == 1) & ~df.found_orig][["artist", "track", "decade"]].head(40)
    save(miss, "r8_artist_unmatched_hits_sample.csv")

    P = make_pipeline
    cfg = {"main (orig artist)": P(MAIN),
           "audio + decade": P(AUDIO + DEC_COLS),
           "strict matching": P(AUDIO + DEC_COLS + ["art_strict"]),
           "wide (featured credits)": P(AUDIO + DEC_COLS + ["art_wide"])}
    summ, folds = cv_eval(cfg, df, y, ref="main (orig artist)", label="R8-artist",
                          groups=groups_of(df))
    print("\n=== R8 artist resolution: 5x5 CV ===")
    show(summ)
    save(summ, "r8_artist_cv_summary.csv")
    save(folds, "r8_artist_cv_folds.csv")


# =========================================================================== (ii) genre
FAMILIES = {
    "hiphop":     ["hip hop", "rap", "trap", "drill", "grime", "crunk", "bounce"],
    "rnb_soul":   ["r&b", "soul", "funk", "motown", "disco", "new jack", "quiet storm", "doo-wop"],
    "country":    ["country", "bluegrass", "americana", "honky", "western swing"],
    "rock":       ["rock", "punk", "grunge", "emo", "britpop", "new wave", "shoegaze", "post-"],
    "metal":      ["metal", "hardcore", "deathcore", "grindcore", "thrash", "doom"],
    "electronic": ["edm", "house", "techno", "trance", "electro", "dubstep", "drum and bass",
                   "dance", "idm", "ambient", "downtempo", "breakbeat", "glitch", "synth"],
    "pop":        ["pop", "boy band", "girl group", "teen", "adult standards", "easy listening"],
    "jazz":       ["jazz", "bebop", "swing", "big band", "bop", "exotica", "lounge"],
    "classical":  ["classical", "orchestra", "baroque", "opera", "romantic era", "early music",
                   "choral", "compositional", "minimalism", "soundtrack", "score"],
    "folk":       ["folk", "singer-songwriter", "acoustic", "cantautor"],
    "blues":      ["blues"],
    "latin":      ["latin", "reggaeton", "salsa", "bachata", "cumbia", "mexican", "tango", "bossa",
                   "samba", "mpb", "sertanejo", "flamenco", "ranchera", "banda", "tropical"],
    "reggae":     ["reggae", "dancehall", "ska", "dub", "rocksteady"],
    "gospel":     ["gospel", "christian", "worship", "ccm", "spiritual"],
    "world":      ["afro", "k-pop", "j-pop", "indian", "bollywood", "arab", "turkish", "greek",
                   "celtic", "world", "chanson", "schlager", "french", "german", "italian",
                   "brazil", "polka", "nordic", "russian", "african", "hawaiian"],
}
FAM_COLS = [f"g_{k}" for k in FAMILIES]


def attach_genres(df):
    from feature_experiments import normalize_artist
    art = pd.read_csv(os.path.join(DATA, "spotify_artists.csv"),
                      usecols=["name", "genres", "followers"])
    art = art[art["genres"].astype(str).str.len() > 2].copy()
    art["na"] = art["name"].map(normalize_artist)
    art = art.sort_values("followers", ascending=False).drop_duplicates("na")
    rows = {}
    for na, g in zip(art["na"], art["genres"]):
        tags = ast.literal_eval(g)
        cnt = {k: sum(any(w in t for w in kw) for t in tags) for k, kw in FAMILIES.items()}
        dom = max(cnt, key=cnt.get) if max(cnt.values()) > 0 else "other"
        rows[na] = [float(cnt[k] > 0) for k in FAMILIES] + [dom]
    g = pd.DataFrame.from_dict(rows, orient="index", columns=FAM_COLS + ["genre_dom"])
    g.index.name = "na"
    return df.merge(g.reset_index(), on="na", how="inner")


def run_genre():
    df = load_kaggle()
    n_all = len(df)
    data, fcols = load_fft(df)
    data = attach_genres(data).reset_index(drop=True)
    y = data["target"].values
    facts = {"n_benchmark": n_all, "n_fft_and_genre": len(data),
             "hit_share": round(y.mean(), 4)}
    dist = (pd.crosstab(data["genre_dom"], data["target"], normalize="columns")
              .rename(columns={0: "share_flops", 1: "share_hits"}).round(4))
    dist["n"] = data["genre_dom"].value_counts()
    print(facts)
    print("\n=== dominant genre family: share among flops / hits ===")
    print(dist.sort_values("n", ascending=False).to_string())
    save(dist.reset_index(), "r8_genre_distribution.csv")

    # How much genre do the descriptors carry? 5-fold, predict the dominant family.
    fam = data["genre_dom"].to_numpy(dtype=object)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gacc = {}
    for name, cols in [("13 Spotify", AUDIO), ("31 FFT", fcols), ("13 Spotify + FFT", AUDIO + fcols)]:
        p = cross_val_predict(RandomForestClassifier(200, random_state=SEED, n_jobs=-1),
                              data[cols].values, fam, cv=skf)
        gacc[f"genre_acc_{name}"] = round(accuracy_score(fam, p), 4)
    gacc["genre_acc_majority"] = round(pd.Series(fam).value_counts(normalize=True).iloc[0], 4)
    print(gacc)
    facts.update(gacc)

    P = make_pipeline
    groups = groups_of(data)
    cfg = {"main": P(MAIN), "main + FFT": P(MAIN + fcols),
           "main + genre": P(MAIN + FAM_COLS), "main + genre + FFT": P(MAIN + FAM_COLS + fcols)}
    summ, folds = cv_eval(cfg, data, y, ref="main", label="R8-genre", groups=groups)
    print("\n=== R8 genre as a feature: 5x5 CV (songs with FFT and genre) ===")
    show(summ)
    save(summ, "r8_genre_cv_summary.csv")
    save(folds, "r8_genre_cv_folds.csv")

    # genre-matched sample: as many hits as flops inside every dominant family
    rng = np.random.RandomState(SEED)
    keep = []
    for g, sub in data.groupby("genre_dom"):
        h, f = sub.index[sub.target == 1], sub.index[sub.target == 0]
        k = min(len(h), len(f))
        keep += list(rng.choice(h, k, replace=False)) + list(rng.choice(f, k, replace=False))
    m = data.loc[sorted(keep)].reset_index(drop=True)
    ym = m["target"].values
    facts.update(n_matched=len(m), matched_new_artist=int((m.artist_prior_hits == 0).sum()))
    cfgm = {"main": P(MAIN), "main + FFT": P(MAIN + fcols), "audio + FFT": P(AUDIO + fcols),
            "audio": P(AUDIO)}
    summ, folds = cv_eval(cfgm, m, ym, ref="main", label="R8-matched", groups=groups_of(m))
    print("\n=== R8 genre-matched sample: 5x5 CV ===")
    show(summ)
    save(summ, "r8_genre_matched_cv_summary.csv")
    save(folds, "r8_genre_matched_cv_folds.csv")
    save(pd.DataFrame([facts]).T.reset_index().rename(columns={"index": "fact", 0: "value"}),
         "r8_genre_facts.csv")


# =========================================================================== (iii) decay
def run_decay():
    df = load_kaggle()
    data, fcols = load_fft(df)
    rows = []
    for src, frame, cfgs in [
            ("all songs", df, [("audio", AUDIO), ("audio + artist", AUDIO + [ART])]),
            ("songs with FFT", data, [("audio", AUDIO), ("audio + artist", AUDIO + [ART]),
                                      ("audio + FFT", AUDIO + fcols),
                                      ("audio + artist + FFT", AUDIO + [ART] + fcols)])]:
        y = frame["target"].values
        for i, d_tr in enumerate(DEC_ORDER[:-1]):
            tr = (frame["decade"] == d_tr).values
            for d_te in DEC_ORDER[i + 1:]:
                te = (frame["decade"] == d_te).values
                for name, cols in cfgs:
                    mdl = FrozenRF().fit(frame.loc[tr, cols].values, y[tr])
                    acc = accuracy_score(y[te], mdl.predict(frame.loc[te, cols].values))
                    rows.append({"data": src, "train": d_tr, "test": d_te,
                                 "gap": DEC_ORDER.index(d_te) - i, "features": name,
                                 "n_train": int(tr.sum()), "n_test": int(te.sum()),
                                 "accuracy": round(acc, 4)})
        # same-decade reference (5-fold CV inside each decade)
        for d in DEC_ORDER:
            sub = frame[frame["decade"] == d].reset_index(drop=True)
            ys = sub["target"].values
            for name, cols in cfgs:
                p = cross_val_predict(FrozenRF(), sub[cols].values, ys,
                                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
                rows.append({"data": src, "train": d, "test": d, "gap": 0, "features": name,
                             "n_train": int(len(sub) * .8), "n_test": len(sub),
                             "accuracy": round(accuracy_score(ys, p), 4)})
        print(f"  [R8-decay] {src} done", flush=True)
    out = pd.DataFrame(rows)
    save(out, "r8_decay.csv")
    summ = (out.groupby(["data", "features", "gap"])["accuracy"].agg(["mean", "min", "max", "size"])
               .round(4).reset_index())
    print("\n=== R8 rolling origin: accuracy vs decades between train and test ===")
    show(summ)
    save(summ, "r8_decay_summary.csv")


# =========================================================================== (iv) lyrics
def run_lyrics():
    from r4_external import SPOTIFY_13, prior_by_year, rf200
    t = pd.read_csv(os.path.join(DATA, "spotify_tracks.csv"))
    alb = pd.read_csv(os.path.join(DATA, "spotify_albums.csv"), usecols=["id", "release_date"])
    art = pd.read_csv(os.path.join(DATA, "spotify_artists.csv"), usecols=["id", "name"])
    t["year"] = pd.to_datetime(t["album_id"].map(dict(zip(alb.id, alb.release_date))),
                               errors="coerce", format="mixed").dt.year
    amap = dict(zip(art.id.astype(str), art.name.astype(str)))
    first = t["artists_id"].astype(str).str.strip("[]").str.split(",").str[0].str.strip(" '\"")
    t["artist_norm"] = first.map(amap).fillna(first).str.lower().str.strip()
    t["target"] = (pd.to_numeric(t["popularity"], errors="coerce") >= 55).astype(int)
    lyr = pd.read_csv(os.path.join(DATA, "lyrics_features.csv")).drop(columns=["Unnamed: 0"],
                                                                     errors="ignore")
    lcols = [c for c in lyr.columns if c != "track_id"]
    lyr = lyr.drop_duplicates("track_id").rename(columns={c: "lyr_" + c for c in lcols})
    lcols = ["lyr_" + c for c in lcols]
    data = t.dropna(subset=SPOTIFY_13 + ["year"]).copy()
    data["year"] = data["year"].astype(int)
    data["decade_n"] = (data["year"] // 10 * 10).astype(float)
    data = data.reset_index(drop=True)
    data["artist_prior_log"] = prior_by_year(data, "artist_norm", "year", "target")
    data = data.merge(lyr, left_on="id", right_on="track_id", how="inner")
    n_merged = len(data)
    data = data[data["lyr_n_words"] > 0].reset_index(drop=True)     # -1 = no lyrics found
    y = data["target"].values
    facts = {"n_tracks_with_lyrics_row": n_merged, "n_with_lyrics": len(data),
             "hit_share": round(y.mean(), 4)}
    print(facts)
    save(pd.DataFrame([facts]).T.reset_index().rename(columns={"index": "fact", 0: "value"}),
         "r8_lyrics_facts.csv")
    P = lambda cols: make_pipeline(cols, model=rf200())
    groups = np.where(data["artist_prior_log"] > 0, "known", "new")
    cfg = {"13 Spotify + decade + artist": P(SPOTIFY_13 + ["decade_n", "artist_prior_log"]),
           "13 Spotify": P(SPOTIFY_13),
           "13 Spotify + lyrics": P(SPOTIFY_13 + lcols),
           "+ decade + artist + lyrics": P(SPOTIFY_13 + ["decade_n", "artist_prior_log"] + lcols)}
    summ, folds = cv_eval(cfg, data, y, ref="13 Spotify + decade + artist", label="R8-lyrics",
                          n_repeats=1, groups=groups)
    print("\n=== R8 lyrics on SpotGenTrack: 5x1 CV ===")
    show(summ)
    save(summ, "r8_lyrics_cv_summary.csv")
    save(folds, "r8_lyrics_cv_folds.csv")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["decay", "artist", "genre", "lyrics"]
    for p in parts:
        {"artist": run_artist, "genre": run_genre, "decay": run_decay, "lyrics": run_lyrics}[p]()
