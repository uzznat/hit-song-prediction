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
- decade +1.0 (p=0.026); decade×audio +2.6 (p=0.001)
- PCA 5…100 all within noise (185 raw −1.2, p=0.11) → "PCA 40 is the peak" not supported
- rich family knock-outs all within ±0.1 pt → no family matters
- block ablation from full model: only **artist** is essential (−3.3, p=1e-5);
  removing rich +0.45 (n.s.), audio −0.9 (n.s.)

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
