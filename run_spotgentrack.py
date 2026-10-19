"""
external_rich_pipeline.py
============================================================================
Run OUR full audio pipeline (download clips + extract rich librosa features)
on an EXTERNAL dataset that only gives you artist + title (SpotGenTrack, Zhao,
or any CSV). Then test: their/base features  vs  + our artist  vs  + our rich.

This is the SAME method we used on our own dataset, pointed at someone else's
songs. It re-downloads audio, so it is the slow one (hours for thousands of
tracks) - but it lets us claim our RICH audio helps on another dataset too.

------------------------------------------------------------------------
SETUP (on your PC):
  pip install numpy pandas scikit-learn scipy librosa soundfile requests
  Set CSV_PATH, and the column names if auto-detect misses them.
  py external_rich_pipeline.py
It caches everything, so you can stop and restart - it resumes.
------------------------------------------------------------------------
"""
import os, json, time, re
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= EDIT =================
CSV_PATH   = "spotify_tracks.csv"   # <-- SpotGenTrack main file. If you have spotify_artists.csv too, artists auto-merge is handled.
POP_THRESH = 55                     # hit if popularity >= this (only if no hit column)
MAX_TRACKS = 4000                   # cap how many clips to download (time!)
SEED = 42
N_PCA = 40
AUDIO_CANDIDATES = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
# =======================================

CACHE = "ext_cache"; AUDIO_DIR = os.path.join(CACHE, "audio")
MANIFEST = os.path.join(CACHE, "manifest.json")
RICH_CSV = os.path.join(CACHE, "rich.csv")
os.makedirs(AUDIO_DIR, exist_ok=True)


# ---------- audio feature extraction (identical to our project) ----------
def extract_rich(y, sr=22050):
    import librosa
    f, n = [], []
    def add(nm, v): n.append(nm); f.append(float(v))
    def stats(pref, M):
        M = np.atleast_2d(M)
        for i in range(M.shape[0]):
            add(f"{pref}{i}_m", np.mean(M[i])); add(f"{pref}{i}_s", np.std(M[i]))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    stats("mfcc", mfcc); stats("d1", librosa.feature.delta(mfcc)); stats("d2", librosa.feature.delta(mfcc, order=2))
    stats("chr", librosa.feature.chroma_cqt(y=y, sr=sr))
    stats("ton", librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    stats("con", librosa.feature.spectral_contrast(y=y, sr=sr))
    for nm, fn in [("cent", librosa.feature.spectral_centroid), ("bw", librosa.feature.spectral_bandwidth),
                   ("roll", librosa.feature.spectral_rolloff)]:
        stats(nm, fn(y=y, sr=sr))
    stats("flat", librosa.feature.spectral_flatness(y=y))
    stats("zcr", librosa.feature.zero_crossing_rate(y)); stats("rms", librosa.feature.rms(y=y))
    oe = librosa.onset.onset_strength(y=y, sr=sr); add("onset_m", np.mean(oe)); add("onset_s", np.std(oe))
    ac = np.mean(librosa.feature.tempogram(onset_envelope=oe, sr=sr), axis=1)
    add("pulse", np.max(ac) / (np.mean(ac) + 1e-9))
    return f, n


# ---------- Deezer 30s preview download (no key needed) ----------
def deezer_preview(artist, title):
    q = requests.utils.quote(f'artist:"{artist}" track:"{title}"')
    try:
        r = requests.get(f"https://api.deezer.com/search?q={q}", timeout=10).json()
        if r.get("data"):
            return r["data"][0].get("preview") or None
    except Exception:
        return None
    return None


def find_col(df, options):
    low = {c.lower(): c for c in df.columns}
    for o in options:
        if o in low: return low[o]
    return None


def load(df):
    df.columns = [c.strip() for c in df.columns]
    art = find_col(df, ["artists", "artist", "artist_name"])
    ttl = find_col(df, ["track_name", "name", "title", "song"])
    hit = find_col(df, ["target", "hit", "is_hit"])
    yr  = find_col(df, ["year", "release_year"])
    if yr is None:
        dt = find_col(df, ["release_date", "date"])
        if dt: df["._y"] = pd.to_datetime(df[dt], errors="coerce").dt.year; yr = "._y"
    if hit is None:
        pop = find_col(df, ["popularity", "track_popularity", "song_popularity"])
        df["._t"] = (pd.to_numeric(df[pop], errors="coerce") >= POP_THRESH).astype(int); hit = "._t"
    df = df.dropna(subset=[art, ttl, hit]).reset_index(drop=True)
    df["_artist"] = df[art].astype(str).str.replace(r"[\[\]'\"]", "", regex=True).str.split(",").str[0].str.strip()
    df["_title"]  = df[ttl].astype(str).str.strip()
    df["_uri"]    = df["_artist"].str.lower() + "|" + df["_title"].str.lower()
    df["_target"] = df[hit].astype(int)
    df["_year"]   = pd.to_numeric(df[yr], errors="coerce") if yr else np.nan
    return df.drop_duplicates("_uri").reset_index(drop=True)


def artist_feature(df):
    if df["_year"].notna().mean() > 0.5:
        df = df.sort_values("_year").reset_index(drop=True)
        prior, out = {}, []
        for y in sorted(df["_year"].dropna().unique()):
            rows = df[df["_year"] == y]
            for i, r in rows.iterrows(): out.append((i, prior.get(r["_artist"].lower(), 0)))
            for i, r in rows[rows["_target"] == 1].iterrows():
                prior[r["_artist"].lower()] = prior.get(r["_artist"].lower(), 0) + 1
        df["_prior"] = pd.Series(dict(out)).reindex(df.index).fillna(0).values
    else:
        tot = df.groupby(df["_artist"].str.lower())["_target"].transform("sum")
        df["_prior"] = (tot - df["_target"]).clip(lower=0)
    df["_prior_log"] = np.log1p(df["_prior"])
    return df


def download_clips(df):
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    todo = [r for _, r in df.iterrows() if r["_uri"] not in manifest][:MAX_TRACKS]
    print(f"downloading up to {len(todo)} clips (resumable)...")
    for i, r in enumerate(todo):
        url = deezer_preview(r["_artist"], r["_title"])
        path = None
        if url:
            path = os.path.join(AUDIO_DIR, re.sub(r"[^\w]", "_", r["_uri"])[:120] + ".mp3")
            try:
                a = requests.get(url, timeout=15)
                open(path, "wb").write(a.content)
            except Exception:
                path = None
        manifest[r["_uri"]] = path
        if (i + 1) % 50 == 0:
            json.dump(manifest, open(MANIFEST, "w")); print(f"  {i+1}/{len(todo)}")
        time.sleep(0.05)
    json.dump(manifest, open(MANIFEST, "w"))
    return manifest


def build_rich(manifest):
    import librosa
    cached = pd.read_csv(RICH_CSV) if os.path.exists(RICH_CSV) else pd.DataFrame()
    done = set(cached["uri"]) if len(cached) else set()
    todo = [(u, p) for u, p in manifest.items() if p and os.path.exists(p) and u not in done]
    print(f"extracting rich features for {len(todo)} clips...")
    rows, names = [], None
    for i, (u, p) in enumerate(todo):
        try:
            y, sr = librosa.load(p, sr=22050, mono=True, duration=30)
            if len(y) < sr: continue
            v, names = extract_rich(y, sr); rows.append([u] + v)
        except Exception:
            continue
        if (i + 1) % 100 == 0 and names:
            new = pd.DataFrame(rows, columns=["uri"] + names)
            cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
            cached.to_csv(RICH_CSV, index=False); rows = []; print(f"  {i+1}/{len(todo)}")
    if rows and names:
        new = pd.DataFrame(rows, columns=["uri"] + names)
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        cached.to_csv(RICH_CSV, index=False)
    print(f"rich ready: {len(cached)}")
    return cached


def main():
    df = load(pd.read_csv(CSV_PATH))
    print(f"dataset: {len(df)} unique songs ({df._target.mean()*100:.1f}% hits)")
    df = artist_feature(df)
    manifest = download_clips(df)
    rich = build_rich(manifest)
    if len(rich) == 0:
        print("No audio features extracted."); return
    rcols = [c for c in rich.columns if c != "uri"]
    data = df.merge(rich, left_on="_uri", right_on="uri").dropna(subset=rcols).reset_index(drop=True)
    print(f"\nsongs with audio: {len(data)}")

    y = data["_target"].values; idx = np.arange(len(data))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    art = data["_prior_log"].values.reshape(-1, 1)
    R = data[rcols].values
    sc = StandardScaler().fit(R[tr])
    Z = PCA(n_components=min(N_PCA, R.shape[1]), random_state=SEED).fit(sc.transform(R[tr])).transform(sc.transform(R))

    # THEIR features: the Spotify audio columns that came with the dataset
    their_cols = [c for c in AUDIO_CANDIDATES if find_col(data, [c])]
    their_cols = [find_col(data, [c]) for c in their_cols]
    THEIR = data[their_cols].values if their_cols else None

    def run(X):
        m = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        return accuracy_score(y[te], m.predict(X[te]))

    print("\n============ HEAD-TO-HEAD: THEIR system vs OUR system ============")
    print(f"   (same {len(data)} songs, same 75/25 split, same Random Forest)\n")
    if THEIR is not None:
        yth = run(THEIR)
        print(f"  THEIR features only (the {len(their_cols)} Spotify audio features) : {yth:.4f}")
    else:
        yth = None
        print("  (no ready Spotify audio columns found in this file to represent 'their features')")
    x_ours = run(np.column_stack([art, Z]))
    print(f"  OUR system (artist + rich librosa audio)                : {x_ours:.4f}")
    if yth is not None:
        print(f"  --> difference (ours - theirs)                          : {x_ours - yth:+.4f}")
        print("      positive = OUR features beat THEIR features on THEIR dataset.")
    print("\n  extra detail:")
    print(f"    our rich audio only   : {run(Z):.4f}")
    print(f"    our artist only       : {run(art):.4f}")
    if THEIR is not None:
        print(f"    their features + ours : {run(np.column_stack([THEIR, art, Z])):.4f}   (everything combined)")
    print("=================================================================")
    print("Report it as: same dataset, same protocol, our features vs theirs.")


if __name__ == "__main__":
    main()
