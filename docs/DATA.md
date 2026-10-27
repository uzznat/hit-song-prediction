# Data — where to get it and where to put it

No raw data is in this repository: the full working tree is **≈53 GB / 1.7 M files**
(third-party licences, and GitHub could not hold it anyway). Every raw dataset goes in
**`data/`**: the "Place as" paths below are relative to `data/`. The derived caches
(`cache/`, `ext_cache/`) stay at the repository root.

Datasets 1 and 2 download themselves on first run. 3 and 4 are manual.

| # | Dataset | Used by | Source | Place as |
|---|---------|---------|--------|----------|
| 1 | **The Spotify Hit Predictor Dataset (1960–2019)** — 6 decade CSVs, 41,106 rows → **40,560** after de-duplicating `uri`. 13 Spotify descriptors + `chorus_hit`, `sections`, artist, title, binary `target`. GPL-3.0. | R1, R2, R3, R5 — the whole main benchmark | auto-downloaded by `repro/feature_experiments.py` from <https://github.com/fortyTwo102/The-Spotify-Hit-Predictor-Dataset> (also on Kaggle as *theoverman/the-spotify-hit-predictor-dataset*) | `The-Spotify-Hit-Predictor-Dataset/dataset-of-{60s,70s,80s,90s,00s,10s}.csv` |
| 2 | **Billboard Hot 100, full weekly history** (`chart_week, title, performer, peak_pos, …`) — the reference hit set 𝓗 behind the artist-history feature. | the `artist_prior_hits_log` feature on dataset 1 | auto-downloaded from <https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/data-out/hot-100-current.csv> | `hot100.csv` (≈19 MB) |
| 3 | **ISMIR 2019 — Zangerle et al.**, Essentia low/high-level features for Million Song Dataset tracks. The honest external benchmark. | R4 (protocols A and B), R6 | <https://zenodo.org/records/3258042> — take `msd_bb_matches.csv`, `msd_bb_non_matches.csv`, `msd_audio_features.tar.gz` (18 GB) | the two CSVs in `data/`; extract the archive in `data/` so `features_tracks_a/ … features_tracks_z/` appear |
| 4 | **SpotGenTrack Popularity Dataset** (Martín-Gutiérrez et al. 2020, IEEE Access) — 101,909 tracks, 13 Spotify descriptors + 207 pre-extracted low-level features + popularity. | R4 (SpotGenTrack) | <https://data.mendeley.com/datasets/4m2x4zngny> | `spotify_tracks.csv`, `spotify_artists.csv`, `spotify_albums.csv`, `low_level_audio_features.csv`, `lyrics_features.csv` (≈740 MB) |
| 5 | **Billboard year-end 2000–2023 + Spotify + lyrics** ("Zhao-style") | **nothing — discarded.** See below. | obtained from a data.world / Kaggle mirror; exact source not recorded | `billboard_24years_lyrics_spotify.csv` |
| 6 | **30-second audio previews (Deezer API)** — the 3,562-song audio subset. | R3 (FFT 31, rich librosa 185) | fetched automatically, `api.deezer.com/search` | written to `cache/audio/` + `cache/deezer_manifest.json` |

## Caches — rebuilt, not downloaded

| Path | Size | Built by | Note |
|------|------|----------|------|
| `cache/` | ≈1.6 GB, 3,566 files | `repro/phase2_fft.py`, then `repro/phase5_richaudio.py`; the whole-benchmark clips and `fft_features_full.csv` by `repro/fetch_audio_full.py` | Deezer clips + `fft_features.csv` + the rich librosa matrix. Hours to build; both scripts resume if interrupted. |
| `ext_cache/ismir_essentia.pkl` | 227 MB | first run of `repro/run_ismir_v2.py` | the 592 flattened Essentia descriptors parsed out of the 18 GB of JSON. **Build this once before any R4/R6 work** — afterwards every ISMIR experiment loads in seconds instead of hours. |

## Why dataset 5 is unused

2,911 of its 3,397 rows have **no** Spotify audio descriptors. An early version of the
presentation reported 0.5906 → 0.5871 "on 3,397 songs"; that was computed on zero-filled
missing features and is invalid. On the 461 rows that do have features the result is
0.506 vs 0.510 (p = 0.82) — chance. The paper mentions it only as a discarded candidate.
