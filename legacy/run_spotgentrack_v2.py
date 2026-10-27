"""
run_spotgentrack_v2.py
============================================================================
SpotGenTrack has PRE-EXTRACTED audio features (low_level_audio_features.csv),
so NO audio download is needed. This does the head-to-head:

   their 13 Spotify features   vs   + our artist feature   vs   their rich
   low-level audio             vs   everything

Put this file next to:
   spotify_tracks.csv, spotify_artists.csv, low_level_audio_features.csv
Then:  py run_spotgentrack_v2.py

pip install numpy pandas scikit-learn
============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

SEED = 42
POP_THRESHOLD = 55          # a track is a "hit" if popularity >= this
N_PCA = 40

SPOTIFY_13 = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
              "acousticness", "instrumentalness", "liveness", "valence", "tempo",
              "duration_ms", "time_signature"]


def find(df, opts):
    low = {c.lower(): c for c in df.columns}
    for o in opts:
        if o in low:
            return low[o]
    return None


def main():
    print("loading files...")
    tracks = pd.read_csv("spotify_tracks.csv")
    tracks.columns = [c.strip() for c in tracks.columns]
    print(f"  tracks: {len(tracks)} rows, columns include: {list(tracks.columns)[:12]}...")

    # ---- id / artist / popularity / year ----
    # pull release_date from spotify_albums.csv (SpotGenTrack stores it there)
    album_id_col = find(tracks, ["album_id", "id_album"])
    if album_id_col:
        try:
            alb = pd.read_csv("spotify_albums.csv")
            alb.columns = [c.strip() for c in alb.columns]
            aid_ = find(alb, ["id", "album_id"])
            adate = find(alb, ["release_date", "date", "year"])
            if aid_ and adate:
                amap_date = dict(zip(alb[aid_].astype(str), alb[adate].astype(str)))
                tracks["_albdate"] = tracks[album_id_col].astype(str).map(amap_date)
                print("  got release dates from spotify_albums.csv")
        except Exception as e:
            print("  (albums date merge skipped:", e, ")")
    tid = find(tracks, ["id", "track_id", "song_id"])
    pop = find(tracks, ["popularity", "track_popularity"])
    art_ids = find(tracks, ["id_artists", "artist_id", "artists_id"])
    date = find(tracks, ["release_date", "date", "year"])

    # ---- artist names from spotify_artists.csv ----
    try:
        artists = pd.read_csv("spotify_artists.csv")
        artists.columns = [c.strip() for c in artists.columns]
        aid = find(artists, ["id", "artist_id"])
        aname = find(artists, ["name", "artist_name"])
        amap = dict(zip(artists[aid].astype(str), artists[aname].astype(str)))
    except Exception as e:
        print("  (could not load artists file:", e, ") - using id as name")
        amap = {}

    def first_artist(x):
        s = str(x).strip("[]'\" ")
        first = s.split(",")[0].strip("[]'\" ")
        return amap.get(first, first)

    tracks["_artist"] = tracks[art_ids].apply(first_artist) if art_ids else "unknown"
    tracks["_pop"] = pd.to_numeric(tracks[pop], errors="coerce")
    tracks["_target"] = (tracks["_pop"] >= POP_THRESHOLD).astype(int)
    if "_albdate" in tracks.columns:
        tracks["_year"] = pd.to_datetime(tracks["_albdate"], errors="coerce", format="mixed").dt.year
    elif date:
        tracks["_year"] = pd.to_datetime(tracks[date], errors="coerce", format="mixed").dt.year
    else:
        tracks["_year"] = np.nan
    tracks["_id"] = tracks[tid].astype(str)

    print(f"  hit rate at popularity>={POP_THRESHOLD}: {tracks['_target'].mean()*100:.1f}%")

    # ---- their rich low-level audio features ----
    print("loading low_level_audio_features.csv (big file, be patient)...")
    low = pd.read_csv("low_level_audio_features.csv")
    low.columns = [c.strip() for c in low.columns]
    lid = find(low, ["id", "track_id", "song_id"])
    low["_id"] = low[lid].astype(str)
    rich_cols = [c for c in low.columns if c not in (lid, "_id") and low[c].dtype != object and not c.lower().startswith("unnamed")]
    print(f"  rich audio features available: {len(rich_cols)}")

    # ---- merge everything ----
    data = tracks.merge(low[["_id"] + rich_cols], on="_id", how="inner")
    data = data.dropna(subset=["_target"]).reset_index(drop=True)
    print(f"  merged tracks with audio: {len(data)}")

    # ---- our leakage-free artist feature ----
    data["_an"] = data["_artist"].astype(str).str.lower().str.strip()
    if data["_year"].notna().mean() > 0.5:
        data = data.sort_values("_year").reset_index(drop=True)
        prior, out = {}, []
        for y in sorted(data["_year"].dropna().unique()):
            rows = data[data["_year"] == y]
            for i, r in rows.iterrows():
                out.append((i, prior.get(r["_an"], 0)))
            for i, r in rows[rows["_target"] == 1].iterrows():
                prior[r["_an"]] = prior.get(r["_an"], 0) + 1
        data["_prior"] = pd.Series(dict(out)).reindex(data.index).fillna(0).values
        note = "leakage-free (earlier years only)"
    else:
        tot = data.groupby("_an")["_target"].transform("sum")
        data["_prior"] = (tot - data["_target"]).clip(lower=0)
        note = "proxy (no reliable year)"
    data["_prior_log"] = np.log1p(data["_prior"])
    print(f"  artist feature: {note}")

    # ---- feature blocks ----
    spot = [find(data, [c]) for c in SPOTIFY_13]
    spot = [c for c in spot if c is not None]
    THEIR = data[spot].apply(pd.to_numeric, errors="coerce").fillna(0).values
    ART = data["_prior_log"].values.reshape(-1, 1)
    R = data[rich_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    y = data["_target"].values
    idx = np.arange(len(data))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)

    sc = StandardScaler().fit(R[tr])
    Z = PCA(n_components=min(N_PCA, R.shape[1]), random_state=SEED).fit(
        sc.transform(R[tr])).transform(sc.transform(R))

    def run(X):
        m = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                   random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        return accuracy_score(y[te], m.predict(X[te]))

    print("\n===================== HEAD-TO-HEAD (SpotGenTrack) =====================")
    print(f"   {len(data)} songs, same 75/25 split, same Random Forest\n")
    yth = run(THEIR)
    print(f"  THEIR 13 Spotify features only          : {yth:.4f}")
    a_art = run(np.column_stack([THEIR, ART]))
    print(f"  their features + OUR artist             : {a_art:.4f}   ({a_art-yth:+.4f})")
    a_rich = run(np.column_stack([THEIR, Z]))
    print(f"  their features + rich audio (PCA)       : {a_rich:.4f}   ({a_rich-yth:+.4f})")
    a_all = run(np.column_stack([THEIR, ART, Z]))
    print(f"  their features + OUR artist + rich      : {a_all:.4f}   ({a_all-yth:+.4f})")
    print(f"\n  OUR system alone (artist + rich, no Spotify 13): {run(np.column_stack([ART, Z])):.4f}")
    print("======================================================================")
    print("If 'their + our artist + rich' > 'their features only', our features")
    print("add real value on THEIR dataset, same protocol. (Their published")
    print("DeepHits number uses lyrics + a CNN, so don't compare to that directly.)")


if __name__ == "__main__":
    main()
