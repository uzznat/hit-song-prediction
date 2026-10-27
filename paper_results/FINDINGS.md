# Step 1 findings — what the paper can and cannot claim (2026-09-14)

All numbers below come from `repro/*.py` → `paper_results/*.csv`.
CV = repeated stratified 5-fold (5x5 unless stated), corrected resampled t-test
(Nadeau & Bengio 2003). "canonical" = the original single split (seed 42).

## R1 — main result, Kaggle Spotify Hit Predictor (40,560 songs)
- Canonical split reproduced EXACTLY: 0.7730 / +decade 0.7955 / +artist 0.8000 /
  +both 0.8215 (McNemar 177/80, p=2.1e-9) — identical to the deck.
- 5x5 CV: audio 0.7817 [0.7796, 0.7838] → +decade 0.8021 → +artist 0.8103 →
  **+decade+artist 0.8321 [0.8303, 0.8339], +5.04 pts, 25/25 folds, p=7e-17**;
  AUC 0.859 → 0.918.
- Rejected: title length +0.2 (p=0.28); chorus_hit+sections −0.03 (p=0.82);
  **song title (TF-IDF, out-of-fold) +6.7 pts (p=4e-20) = dataset-construction artifact**.
- Facts: every decade file is ~50 % hits (0.483–0.499) → decade alone carries no
  marginal information; artist match rate 52.1 %; mean prior hits 3.21 (hits) vs
  0.037 (flops); hit rate by prior hits 0 → 39.3 % (n=33,438), 1-2 → 91.0 %,
  3-5 → 97.2 %, 6-10 → 99.6 %, 11+ → 98.9 %.
- Per-decade CV accuracy, audio → +decade+artist: 60s .743→.788, 70s .750→.796,
  80s .770→.836, 90s .806→.864, 00s .836→.873, 10s .816→.868.

## R3 — audio subset (3,562 songs with a 30 s Deezer clip)
CV overturns several single-split claims of the deck:
- artist +3.8 (p=2e-6, 25/25) ✅ ; decade+artist +5.1 (p=1e-8) ✅
- **rich librosa (185→PCA40) +0.9, p=0.15 — NOT significant** (canonical said +3.8, p=0.002)
- FFT (31) +2.3, p=0.007 — modest but significant; artist+FFT 0.821 best mean
- FFT block (2026-10-25, `r3_fft_*.csv`): FFT on top of artist +2.0 (p=0.008, 23/25);
  artist+FFT+decade 0.824 = best, but decade adds +0.3 n.s. (p=0.50); FFT+decade 0.785 = FFT
  0.786 (FFT already carries the era); artist on top of FFT +3.5 (p=4e-5). Paper Fig. 4.
- decade +1.0 (p=0.026); decade×audio +2.6 (p=0.001)
- PCA 5…100 all within noise (185 raw −1.2, p=0.11) → "PCA 40 is the peak" not supported
- rich family knock-outs all within ±0.1 pt → no family matters
- block ablation from full model: only **artist** is essential (−3.3, p=1e-5);
  removing rich +0.45 (n.s.), audio −0.9 (n.s.)

## R7 — FFT on the whole Kaggle benchmark (2026-10-26, `repro/r7_fft_full.py`)
- Deezer clips for 35,370 / 40,560 songs (87.2 %); hits 94.0 % vs flops 80.6 % covered →
  subset is 53.2 % hits, so all comparisons use the SAME 35,370 songs.
- 5x5 CV: main model (audio+decade+artist) .825 → +FFT .839 (+1.4, 25/25, p=4e-9; AUC .911→.921).
- Gain is all on NEW artists: .789 → .807 (+1.8, 25/25, p=6e-9); known artists .977 → .976 (n.s.).
- artist+FFT without decade .829 (+0.4 vs main, p=0.04 — indicative); decade still useful here,
  unlike on the 3.5k R3 subset. FFT alone on top of audio .804.
- Caveat: flops are non-mainstream genres by construction; spectral descriptors can detect genre,
  so the new-artist FFT gain may be partly benchmark-inflated too. -> CONFIRMED by R8 genre: it is a genre shortcut.

## R8 — the Future Work items we could run (2026-10-27, `repro/r8_future_work.py`, paper Sec. V-F)
- **Genre control (the FFT caveat) — FFT gain is a genre shortcut.** Spotify artist genres
  (SpotGenTrack `spotify_artists.csv`, by normalised name; 20,610 songs with clip + genre, 16 families).
  Latin/classical/jazz/world/metal = 50 % of flops, 1.9 % of hits. Genre recoverable from audio:
  13 Spotify 47.4 %, FFT 49.9 %, both 53.3 % (majority 26.1 %).
  main .828 → +FFT .841 (+1.3, p=1e-4) → +genre .928 (+10.1, p=8e-23); FFT on top of genre −1.1
  (0/25, p=5e-7; new artists −1.4). Genre-matched sample (7,554): main+FFT vs main +0.1 (p=.84,
  new artists +0.1 p=.80); audio+FFT vs audio +2.5 (p=5e-4); main beats audio+FFT by 4.3 (p=2e-8).
  Genre is retrospective (Spotify tags) → used only as a control.
- **Artist resolution — nothing to gain.** 99.99 % of HITS are found in the chart history; the 52 %
  "match rate" = flops by never-charting artists (2.8 % of flops matched). Full-name-first matching
  −0.5 (p=4e-4); featured credits +0.1 n.s. The old Limitations sentence ("zero for many
  established artists") was WRONG and is corrected.
- **Rolling origin (train on one decade, test k later, songs with clip):** audio .809/.765/.728/.689/
  .666/.664 for k=0..5; artist advantage grows with k (+2.8 → +5.1 at k=3); FFT helps at k=0–1 and
  hurts from k=3 (era/genre-specific). No significance test (1–6 pairs per cell).
- **Lyrics (SpotGenTrack lexical, 79,420 tracks with lyrics, 5x1 CV):** 13 Spotify AUC .678 → .729
  (p=3e-4); on top of decade+artist acc .808 → .814 (5/5, p=.002), AUC .774 → .797 (p=2e-5).
  Indicative under Bonferroni for accuracy.

## R5 — Tilburg head-to-head (Du 2022), their exact protocol, 50 seeds of 70/30
- Our re-implementation matches their published numbers (RF 00s 0.846 vs 0.84;
  10s 0.829 vs 0.82; LR 00s 0.821 vs 0.82; 10s 0.807 vs 0.79).
- Same model + our artist feature: RF 00s **0.867**, 10s **0.861**; LR 00s 0.848,
  10s 0.835. Gain on 50/50 seeds in every cell; corrected p between 6e-6 and 9e-8.
  Beats their published number on 100 % of seeds for all four.

## R4 — external datasets
- SpotGenTrack: `run_spotgentrack_v2.py` re-run reproduces the deck EXACTLY
  (0.8115 / +artist 0.8226 / +artist+rich 0.8238 / "our system alone" 0.8222).
  ⚠️ deck mislabels 0.8222 as "our 2 features alone": it is artist + low-level audio.
  ⚠️ label = popularity ≥ 55 → only 19.4 % positives → majority class = 80.6 %:
  accuracy is near-uninformative, use AUC. R4 CV (5-fold): AUC .690 → +artist .762 →
  +decade+artist .777; repro canonical split .8103 → .8219 (slightly different from the old script).
- Billboard year-end 2000-2023 ("Zhao-style"): ⚠️ **2,911 / 3,397 rows have NO
  Spotify features** → only 461 usable songs. The deck's 0.5906 → 0.5871 on
  "3,397 songs" was computed on mostly missing (zero-filled) audio → INVALID.
  On the 461 real songs: 0.506 vs 0.510 (p=0.82), chance level. Drop from the
  validation claims; mention as a limitation at most.
- ISMIR 2019 (Zangerle), R4 5x2 CV, 11,662 songs (5,831 hits with audio):
  A random: audio .708, +year .739, +year+artist **.763** (+2.4, p=7e-8) > published 75.04 %;
  B year-matched: year useless (p=.66), +artist .708 (**+2.6, p=3e-5**, 10/10).
  (canonical single split said +3.1; CV value +2.6 is the one used in the paper.)

## R6 — artist-grouped CV + cold start (2026-09-15, `repro/r6_robustness.py`)
Grouped = StratifiedGroupKFold by normalised artist (no test artist in training).
- Kaggle 5x5: audio .782→.772; +decade+artist .832→.822; gain +5.0 unchanged (25/25, p=1e-13).
- ISMIR A 5x2: +year+artist .763→.759 [.752,.766], still > 75.04 %; gains −0.5 at most.
- ISMIR B 5x2: +artist +2.6 → **+2.9 (p=9e-7, 10/10)**; year still useless.
- Cold start (grouped): Kaggle new artists (33,438, 39 % hits): artist +0.4 n.s., decade +2.2
  (p=2e-9); known artists (7,122, **96.6 % hits**): artist +15.0 → the Kaggle artist gain is
  "known artist ⇒ hit". ISMIR B new artists: artist +1.4 (p=.002); known (72 % hits) +5.3.
  ISMIR A new: year +3.4, artist +0.8 n.s.

## ⚠️ Kaggle label construction (dataset README.txt, GPL-3.0, verbatim)
"'1' implies that this song has featured in the weekly list (Issued by Billboards)
of Hot-100 tracks in that decade at least once". A 'flop' must satisfy ALL of:
- "The track must not appear in the 'hit' list of that decade."
- "The track's artist must not appear in the 'hit' list of that decade."
- "The track must belong to a genre that could be considered non-mainstream and / or avant-garde."
- "The track's genre must not have a song in the 'hit' list."
Consequences for the paper:
- artist history (prior decades only) is still leakage-free in time, but its gain on
  Kaggle is INFLATED by the artist-exclusion rule (mean prior hits 3.21 vs 0.037);
  Kaggle numbers = benchmark-specific upper bound.
- the genre rule explains instrumentalness / "catalogue filler" separability.
- the honest estimate of the artist feature is ISMIR 2019 (negatives sampled by
  title dissimilarity only, no artist rule): +2.6 pts in the year-matched protocol (CV).
- the Tilburg head-to-head remains a valid "beats prior work ON THIS BENCHMARK" claim.

## Deck statements to correct in the paper
- "paper 0.877" dashed line = Middlebrook & Sheik 2019, a different dataset.
- Tilburg comparison: now a true head-to-head (R5), no longer "close comparison".
- Rich audio / PCA 40 / family ablation claims do not survive CV (R3).
- Billboard external validation invalid (missing features).
- SpotGenTrack "our 2 alone" label wrong.

## R9 — hit-class metrics and literature gaps (2026-10-27)

`repro/r9_metrics.py`, same folds as R1 / R4.
- Kaggle, frozen RF, 5x5: audio precision .758 / recall .819 / F1 .787 → + decade + artist .828 / .833 / **.831** (+4.3 F1, 25/25, corrected p = 7e-15). The accuracy gain is not a precision/recall trade.
- SpotGenTrack, RF-200, 5x1: the 13 descriptors find 8.3 % of hits (F1 .147); + decade + artist recall 25.2 % at the same precision (.62), F1 **.358** (5/5, p = 7e-7).

Paper changes from a comparison with the field: Interiano et al. (2018, R. Soc. Open Sci.) already used a "superstar"
(recent chart presence) indicator with a Random Forest → now credited in Intro and Related Work, and our novelty stated
as significance + same-data head-to-head + sampling-artefact controls. Added Sturm (2014, "horse"), Flexer (2007, artist
filter), Salganik et al. (2006, cumulative advantage); hyperparameters-not-tuned rationale; no-deep-learning limitation;
explicit answer on audio content in the Discussion.
