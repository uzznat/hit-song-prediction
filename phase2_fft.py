"""
phase2_fft.py  —  Phase 2: rhythmic / spectral (FFT) features.
==============================================================
The audio-derived feature Lior wanted. The 13 Kaggle features are Spotify's
*summaries*; here we go back to the waveform and extract finer rhythmic and
spectral structure (tempogram / pulse clarity / onset regularity + spectral
stats + MFCC + chroma), then ask the Phase 0 harness the real question:

    does this add signal the 13 audio features don't already contain?

WHY THIS ONE RUNS ON YOUR MACHINE (not in the chat)
---------------------------------------------------
Spotify's audio endpoints are dead (403 since Nov 2024), so we source 30-second
previews from Deezer's public API. That download can't run inside the assistant's
sandbox, so this script is built to run on your PC.

PIPELINE (all cached — safe to stop and re-run, it resumes)
    1. sample up to MAX_TRACKS tracks (stratified by decade x hit/flop)
    2. Deezer search by artist+title  ->  30s preview mp3   (cache/audio/)
    3. librosa  ->  31 rhythmic/spectral features           (cache/fft_features.csv)
    4. fresh stratified split on the tracks that got audio
    5. harness: baseline (13 audio) vs baseline+FFT  ->  redundancy, McNemar, per-decade
    6. graphs: phase2_results.png , phase2_redundancy.png

RUN
    put next to phase0_harness.py, then:
        pip install numpy pandas scikit-learn scipy matplotlib librosa soundfile
        py phase2_fft.py
    (librosa needs to decode mp3; if it errors on loading, install ffmpeg and
     add it to PATH — on Windows: https://www.gyan.dev/ffmpeg/builds/ )

Start with MAX_TRACKS small (e.g. 2000) to check the plumbing, then raise it.
"""

import os, re, glob, json, time, hashlib, unicodedata, urllib.parse, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from phase0_harness import evaluate_feature, make_baseline_rf, BASE_FEATURES_A

SEED = 42
MAX_TRACKS = 4000                 # raise once the plumbing works (more audio = stronger test)
SPOTIFY_DIR = "The-Spotify-Hit-Predictor-Dataset"
SPOTIFY_URL = ("https://raw.githubusercontent.com/fortyTwo102/"
               "The-Spotify-Hit-Predictor-Dataset/master/dataset-of-{d}.csv")
CACHE = "cache"; AUDIO_DIR = os.path.join(CACHE, "audio")
FEAT_CACHE = os.path.join(CACHE, "fft_features.csv")
MANIFEST = os.path.join(CACHE, "deezer_manifest.json")
DEEZER = "https://api.deezer.com/search/track/?q={q}&limit=1"
DEC_ORDER = ["60s", "70s", "80s", "90s", "00s", "10s"]
AUDIO = BASE_FEATURES_A
GREEN, GRAY, RED = "#1DB954", "#B4B2A9", "#e24b4a"


# --------------------------------------------------------------------------- #
# 1. Spotify metadata
# --------------------------------------------------------------------------- #
def load_tracks():
    os.makedirs(SPOTIFY_DIR, exist_ok=True)
    for d in DEC_ORDER:
        p = os.path.join(SPOTIFY_DIR, f"dataset-of-{d}.csv")
        if not os.path.exists(p):
            urllib.request.urlretrieve(SPOTIFY_URL.format(d=d), p)
    frames = []
    for f in sorted(glob.glob(f"{SPOTIFY_DIR}/dataset-of-*.csv")):
        dec = re.search(r"dataset-of-(\w+)\.csv", os.path.basename(f)).group(1)
        t = pd.read_csv(f); t["decade"] = dec; frames.append(t)
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["uri"], keep="first").reset_index(drop=True))


def uid(uri):
    return hashlib.md5(uri.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 2. Deezer preview download (cached)
# --------------------------------------------------------------------------- #
def deezer_preview_url(artist, title):
    """Query Deezer's public API for a 30s preview mp3 URL (or None)."""
    def clean(s):
        return re.sub(r'["\\]', "", str(s)).strip()
    q = f'artist:"{clean(artist)}" track:"{clean(title)}"'
    url = DEEZER.format(q=urllib.parse.quote(q))
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r).get("data", [])
        if data and data[0].get("preview"):
            return data[0]["preview"]
    except Exception:
        pass
    return None


def fetch_audio(df):
    """Download previews for up to MAX_TRACKS tracks; returns set of uris with audio."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}

    # stratified sample by decade x target so every era is represented
    sample = (df.groupby(["decade", "target"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), MAX_TRACKS // 12), random_state=SEED)))
    todo = [r for r in sample.itertuples() if r.uri not in manifest]
    print(f"Deezer: {len(manifest)} cached, {len(todo)} to fetch (cap {MAX_TRACKS})")

    for i, r in enumerate(todo):
        prev = deezer_preview_url(r.artist, r.track)
        if prev:
            path = os.path.join(AUDIO_DIR, f"{uid(r.uri)}.mp3")
            try:
                urllib.request.urlretrieve(prev, path)
                manifest[r.uri] = path
            except Exception:
                manifest[r.uri] = None
        else:
            manifest[r.uri] = None
        if (i + 1) % 50 == 0:
            json.dump(manifest, open(MANIFEST, "w"))
            print(f"  {i+1}/{len(todo)}  matched so far: {sum(v is not None for v in manifest.values())}")
        time.sleep(0.12)                     # be gentle with Deezer's rate limit
    json.dump(manifest, open(MANIFEST, "w"))
    ok = {u for u, p in manifest.items() if p and os.path.exists(p)}
    print(f"Deezer done: {len(ok)} tracks with audio")
    return manifest


# --------------------------------------------------------------------------- #
# 3. librosa feature extraction (cached)
# --------------------------------------------------------------------------- #
def extract_rhythm_spectral(y, sr=22050):
    import librosa
    feats, names = [], []
    def add(n, v): names.append(n); feats.append(float(v))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        import librosa.feature.rhythm as _rhythm
        tempo_est = _rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
    except Exception:
        tempo_est = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    add("tempo_est", tempo_est)
    add("onset_env_mean", np.mean(onset_env)); add("onset_env_std", np.std(onset_env))
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    ac = np.mean(tg, axis=1)
    add("pulse_clarity", np.max(ac) / (np.mean(ac) + 1e-9))
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    if len(onsets) > 2:
        ioi = np.diff(onsets)
        add("onset_ioi_mean", np.mean(ioi)); add("onset_ioi_cv", np.std(ioi) / (np.mean(ioi) + 1e-9))
    else:
        add("onset_ioi_mean", 0.0); add("onset_ioi_cv", 0.0)
    for fn, f in [("centroid", librosa.feature.spectral_centroid),
                  ("bandwidth", librosa.feature.spectral_bandwidth),
                  ("rolloff", librosa.feature.spectral_rolloff)]:
        v = f(y=y, sr=sr); add(f"spec_{fn}_mean", np.mean(v)); add(f"spec_{fn}_std", np.std(v))
    fl = librosa.feature.spectral_flatness(y=y)
    add("spec_flatness_mean", np.mean(fl)); add("spec_flatness_std", np.std(fl))
    con = librosa.feature.spectral_contrast(y=y, sr=sr)
    add("spec_contrast_mean", np.mean(con)); add("spec_contrast_std", np.std(con))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13): add(f"mfcc{i+1}_mean", np.mean(mfcc[i]))
    ch = librosa.feature.chroma_stft(y=y, sr=sr)
    add("chroma_mean", np.mean(ch)); add("chroma_std", np.std(ch))
    return feats, names


def build_fft_features(manifest):
    import librosa
    if os.path.exists(FEAT_CACHE):
        cached = pd.read_csv(FEAT_CACHE)
        print(f"FFT features cached: {len(cached)} tracks")
        done = set(cached["uri"])
    else:
        cached, done = pd.DataFrame(), set()
    rows, names = [], None
    todo = [(u, p) for u, p in manifest.items() if p and u not in done and os.path.exists(p)]
    print(f"extracting librosa features for {len(todo)} clips ...")

    def flush():
        nonlocal cached, rows
        if not rows or names is None:
            return
        new = pd.DataFrame(rows, columns=["uri"] + names)
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        cached.to_csv(FEAT_CACHE, index=False)   # CSV = no extra library, saves progress
        rows = []

    for i, (u, p) in enumerate(todo):
        try:
            y, sr = librosa.load(p, sr=22050, mono=True, duration=30)
            if len(y) < sr:                       # too short / corrupt
                continue
            vec, names = extract_rhythm_spectral(y, sr)
            rows.append([u] + vec)
        except Exception:
            continue
        if (i + 1) % 200 == 0:                    # save every 200 so nothing is lost
            flush()
            print(f"  {i+1}/{len(todo)}  (saved)")
    flush()
    print(f"FFT features ready: {len(cached)} tracks")
    return cached


# --------------------------------------------------------------------------- #
# 4-5. Evaluate through the harness (fresh split on the audio subset)
# --------------------------------------------------------------------------- #
def evaluate(df, fft):
    fcols = [c for c in fft.columns if c != "uri"]
    data = df.merge(fft, on="uri", how="inner").dropna(subset=fcols).reset_index(drop=True)
    print(f"\nusable tracks (audio + features): {len(data)}")
    if len(data) < 800:
        print("  (few tracks — raise MAX_TRACKS for a stronger test)")

    y = data["target"].values
    idx = np.arange(len(data))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    tr, val = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=SEED)

    base = {"X_tr": data[AUDIO].values[tr], "X_val": data[AUDIO].values[val],
            "X_te": data[AUDIO].values[te], "y_tr": y[tr], "y_val": y[val], "y_te": y[te],
            "feature_names": AUDIO}
    col = lambda ix: data[fcols].values[ix]
    res = evaluate_feature(base, col(tr), col(val), col(te),
                           feature_names=fcols, groups_te=data["decade"].values[te])
    res["_fcols"] = fcols
    return res


# --------------------------------------------------------------------------- #
# 6. Graphs
# --------------------------------------------------------------------------- #
def plots(res):
    # redundancy: max|corr| of each FFT feature with the 13 audio features
    corr = [r["max_abs_corr"] for r in res["redundancy"]]
    names = res["feature_names"]
    order = np.argsort(corr)
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = [RED if corr[i] >= 0.7 else GREEN for i in order]
    ax.barh(range(len(order)), [corr[i] for i in order], color=colors)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.axvline(0.7, ls="--", lw=1, color="#888")
    ax.set_xlabel("max |correlation| with the 13 audio features")
    ax.set_title("FFT features: orthogonal (green) vs redundant (red)", fontsize=11, weight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("phase2_redundancy.png", dpi=140, bbox_inches="tight")
    print("saved phase2_redundancy.png")

    # result: accuracy base vs base+FFT + per-decade
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.bar(["audio", "audio + FFT"], [res["acc_base"], res["acc_aug"]], color=[GRAY, GREEN], width=0.5)
    for i, v in enumerate([res["acc_base"], res["acc_aug"]]):
        a1.text(i, v + 0.002, f"{v:.3f}", ha="center", weight="bold")
    p = res["mcnemar"]["pvalue"]
    a1.set_title(f"Accuracy on audio subset  (McNemar p={p:.2g})", fontsize=11, weight="bold")
    a1.set_ylabel("test accuracy")
    for s in ("top", "right"): a1.spines[s].set_visible(False)

    sg = res["subgroup"]
    a2.bar(sg["group"], sg["delta"] * 100, color=GREEN)
    a2.axhline(0, lw=0.8, color="#ccc")
    a2.set_ylabel("accuracy gain (points)")
    a2.set_title("FFT gain by decade", fontsize=11, weight="bold")
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("phase2_results.png", dpi=140, bbox_inches="tight")
    print("saved phase2_results.png")


def main():
    df = load_tracks()
    manifest = fetch_audio(df)
    fft = build_fft_features(manifest)
    if len(fft) == 0:
        print("No audio features yet — check Deezer/ffmpeg, then re-run.")
        return
    res = evaluate(df, fft)
    n_orth = sum(1 for r in res["redundancy"] if r["max_abs_corr"] < 0.7)
    sig = "SIGNIFICANT" if res["mcnemar"]["pvalue"] < 0.05 else "not significant"
    print("\n" + "=" * 60)
    print(f"FFT block: {res['n_new_features']} features, {n_orth} orthogonal to audio")
    print(f"accuracy  {res['acc_base']:.4f} -> {res['acc_aug']:.4f}  (delta {res['delta_acc']:+.4f})")
    print(f"McNemar   p={res['mcnemar']['pvalue']:.3g}  [{sig}]")
    print("=" * 60)
    plots(res)
    print("\ndone.")


if __name__ == "__main__":
    main()
