"""
Audio for the WHOLE Kaggle benchmark (the original phase2_fft.py stopped at a
4,000-song sample).  Two stages, both resumable:

    py repro/fetch_audio_full.py download   # Deezer 30 s previews -> cache/audio/
    py repro/fetch_audio_full.py extract    # 31 FFT descriptors   -> cache/fft_features_full.csv

Kept separate from cache/deezer_manifest.json and cache/fft_features.csv so the
3,562-song subset used by R3 stays exactly as published.

Deezer's fielded search (artist:"..." track:"...") used by phase2_fft.py no longer
returns results, so we use a free-text query and accept a result only if its
main artist and its title match the Kaggle row after normalisation. Every
decision is logged in cache/deezer_matches_full.csv for auditing.
"""
import os
# One BLAS/numba thread per worker: with 16 processes x N threads the run thrashed
# (~6 s CPU per clip instead of 0.5 s). Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import re
import socket
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
import multiprocessing as mp
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed, wait)

import pandas as pd

from common import ROOT, load_kaggle

CACHE = os.path.join(ROOT, "cache")
AUDIO_DIR = os.path.join(CACHE, "audio")
OLD_MANIFEST = os.path.join(CACHE, "deezer_manifest.json")
MANIFEST = os.path.join(CACHE, "deezer_manifest_full.json")
MATCHES = os.path.join(CACHE, "deezer_matches_full.csv")
FEATS = os.path.join(CACHE, "fft_features_full.csv")
SKIPPED = os.path.join(CACHE, "fft_skipped_full.txt")   # clips that hung the decoder
STALL = 180                                            # s without any finished clip
API = "https://api.deezer.com/search/track/?q={q}&limit=10"
RATE = 8.0                      # API requests per second (Deezer allows 50 per 5 s)
socket.setdefaulttimeout(30)    # urlretrieve has no timeout of its own; a stalled mp3 hung a thread


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)                       # (Remastered), [Live] ...
    s = re.split(r"\s+-\s+", s)[0]                                # "Title - 2011 Remaster"
    s = s.replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def main_artist(a):
    a = re.split(r"\s+(?:featuring|feat\.?|ft\.?|with|x|vs\.?)\s+|,|\s&\s|\s/\s",
                 str(a), flags=re.I)[0]
    return norm(a)


def same(a, b):
    return a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a))


class Limiter:
    def __init__(self, rate):
        self.gap, self.next, self.lock = 1.0 / rate, time.monotonic(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


LIMIT = Limiter(RATE)


def get_json(url, tries=4):
    for k in range(tries):
        LIMIT.wait()
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.load(r)
            if "error" in d:                                    # quota: back off
                time.sleep(5 * (k + 1))
                continue
            return d
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def find(uri, artist, track):
    from phase2_fft import uid
    ta, tt = main_artist(artist), norm(track)
    d = get_json(API.format(q=urllib.parse.quote(f"{artist} {track}")))
    if d is None:
        return {"uri": uri, "status": "api_error"}
    hits = [h for h in d.get("data", []) if h.get("preview")
            and same(ta, norm(h.get("artist", {}).get("name", "")))
            and same(tt, norm(h.get("title", "")))]
    # prefer a version whose "live"-ness agrees with the Kaggle title
    live = "live" in str(track).lower()
    hits.sort(key=lambda h: ("live" in h["title"].lower()) != live)
    if hits:
        h = hits[0]
        path = os.path.join(AUDIO_DIR, f"{uid(uri)}.mp3")
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(h["preview"], path)
            return {"uri": uri, "status": "ok", "path": path, "deezer_id": h["id"],
                    "deezer_artist": h["artist"]["name"], "deezer_title": h["title"]}
        except Exception:
            return {"uri": uri, "status": "download_error"}
    return {"uri": uri, "status": "not_found", "n_results": len(d.get("data", []))}


def download():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    df = load_kaggle()
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    if not manifest and os.path.exists(OLD_MANIFEST):           # reuse the 3,563 clips
        manifest = {u: p for u, p in json.load(open(OLD_MANIFEST)).items() if p}
    log = [] if not os.path.exists(MATCHES) else pd.read_csv(MATCHES).to_dict("records")
    tried = set(manifest) | {r["uri"] for r in log if r["status"] == "not_found"}
    todo = [r for r in df[["uri", "artist", "track"]].itertuples(index=False)
            if r.uri not in tried]
    print(f"Deezer: {len(manifest)} clips already, {len(todo)} songs to query", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(find, *r) for r in todo]
        for f in as_completed(futs):
            r = f.result()
            log.append(r)
            if r["status"] == "ok":
                manifest[r["uri"]] = r["path"]
            done += 1
            if done % 500 == 0 or done == len(todo):
                json.dump(manifest, open(MANIFEST, "w"))
                pd.DataFrame(log).to_csv(MATCHES, index=False)
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)}  clips: {len(manifest)}  "
                      f"({rate:.1f}/s, ~{(len(todo) - done) / rate / 60:.0f} min left)",
                      flush=True)
    print(f"done: {len(manifest)} songs with a clip", flush=True)


def _extract(item):
    import librosa
    from phase2_fft import extract_rhythm_spectral
    u, p = item
    try:
        y, sr = librosa.load(p, sr=22050, mono=True, duration=30)
        if len(y) < sr:
            return None
        vec, names = extract_rhythm_spectral(y, sr)
        return u, vec, names
    except Exception:
        return None


def extract():
    manifest = json.load(open(MANIFEST))
    old = os.path.join(CACHE, "fft_features.csv")
    cached = pd.read_csv(FEATS) if os.path.exists(FEATS) else pd.read_csv(old)
    skipped = set(open(SKIPPED).read().split()) if os.path.exists(SKIPPED) else set()
    done = set(cached.uri)
    todo = [(u, p) for u, p in manifest.items()
            if u not in done and u not in skipped and os.path.exists(p)]
    print(f"FFT: {len(done)} cached, {len(skipped)} skipped, {len(todo)} clips to process",
          flush=True)
    rows, names, t0, n = [], None, time.time(), 0

    def flush():
        nonlocal cached, rows
        if rows:
            cached = pd.concat([cached, pd.DataFrame(rows, columns=["uri"] + names)],
                               ignore_index=True)
            cached.to_csv(FEATS, index=False)
            rows = []

    # One undecodable mp3 once hung a worker and, with the ordered ex.map, froze the
    # whole run. Now: unordered, and if nothing finishes for STALL s, the pending clips
    # are written to SKIPPED and the process exits (rerun to resume without them).
    ex = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 4))
    futs = {ex.submit(_extract, it): it[0] for it in todo}
    pending = set(futs)
    while pending:
        fin, pending = wait(pending, timeout=STALL, return_when=FIRST_COMPLETED)
        if not fin:
            flush()
            running = [futs[f] for f in pending if f.running()]
            with open(SKIPPED, "a") as fh:
                fh.write("\n".join(running) + "\n")
            print(f"stalled {STALL}s: skipping {len(running)} clips, rerun to resume",
                  flush=True)
            for p in mp.active_children():
                p.kill()
            os._exit(3)
        for f in fin:
            r = f.result()
            n += 1
            if r:
                rows.append([r[0]] + r[1]); names = r[2]
            if n % 1000 == 0 or n == len(todo):
                flush()
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  ({rate:.1f}/s, "
                      f"~{(len(todo) - n) / rate / 60:.0f} min left)", flush=True)
    flush()
    ex.shutdown()
    print(f"FFT features: {len(cached)} songs", flush=True)


if __name__ == "__main__":
    {"download": download, "extract": extract}[sys.argv[1]]()
