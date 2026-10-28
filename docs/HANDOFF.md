# Handoff — hit song prediction project

Written 2026-10-19. For whoever picks this up next.
Everything below is verified against the files in this repository.

---

## 1. State in one line

**The research is finished and written up.** The paper (`paper/main.pdf`, 7 pages, IEEE
conference format) and the 23-slide deck (`presentation/hit_prediction_deck.pdf`) are
complete and internally consistent with the audit in `paper_results/FINDINGS.md`. What is
*not* done is everything outside the research: no public code release, no submission, no
artifact/DOI. See §7.

---

## 2. What the project set out to do, and how it changed

It started as "can we beat the published hit-prediction numbers with better features?" and
ended as something more defensible: **"which of the gains people report are real?"**

That shift came from one discovery, and it is the thing to understand before touching
anything else. The Kaggle *Spotify Hit Predictor* benchmark — on which essentially all the
headline numbers in this literature rest — builds its negative class with a rule (quoted
verbatim in `paper_results/FINDINGS.md`) requiring that a non-hit's **artist** must not
appear in that decade's hit list, and that its **genre** be "non-mainstream and/or
avant-garde" with no song in the hit list.

So a feature counting an artist's earlier hits does not merely predict success on that
benchmark — it partly reconstructs how the benchmark was assembled. Flops average 0.037
earlier hits; hits average 3.21. The +5.0 pts on Kaggle is real but *benchmark-specific*.
The paper's transferable number is the **+2.6 pts** from the ISMIR 2019 year-matched
protocol, whose negatives are drawn by title dissimilarity only.

The second shift: an earlier round of analysis used single 80/20 splits with McNemar's
test. Re-running everything under repeated stratified CV with the corrected resampled
*t*-test (Nadeau & Bengio 2003) **overturned six of our own earlier claims**. They are
listed at the end of `FINDINGS.md` and in the deck ("Six claims that did not survive
re-checking"). This is why the two-layer code structure below exists.

---

## 3. The two layers of code — this matters

### Layer A — `repro/` : canonical. Trust this.

Every number in the paper is produced here and written to `paper_results/*.csv`.

| Script | Output | Headline |
|---|---|---|
| `common.py` | — | Shared loaders, sklearn pipelines, both evaluation modes. `canonical_eval()` = single split seed 42 + McNemar; `cv_eval()` = repeated stratified K-fold + corrected resampled *t*-test. All scaling / PCA / TF-IDF live **inside** the pipeline, so they refit on every fold. |
| `r1_kaggle_core.py` | `r1_*.csv` | Main result. Audio 0.7817 → +decade 0.8021 → +artist 0.8103 → **+both 0.8321** [0.8303, 0.8339]: +5.04 pts, 25/25 folds, *p* = 7e-17; AUC 0.859 → 0.918. The canonical split reproduces the old deck exactly (0.7730 → 0.8215). |
| `r2_models_time.py` | `r2_*.csv` | The same +5 pts across RF, XGBoost, LogReg, kNN and SVM-RBF — it is the features, not the model. Plus the temporal hold-out (train on past decades, test on an unseen one) and SHAP. |
| `r3_audio_subset.py` | `r3_*.csv` | The 3,562-song Deezer-clip subset, and **where most of the demolition happened.** Rich librosa (185 → PCA 40): +0.9, *p* = 0.15, *not significant* — the single split had said +3.8, *p* = 0.002. "PCA 40 is the peak" unsupported (5…100 all within noise). No librosa family matters (knock-outs within ±0.1). FFT (31) +2.3, *p* = 0.007 survives, modestly. Block ablation from the full model: only **artist** is essential (−3.3, *p* = 1e-5). |
| `r4_external.py` | `r4_*.csv` | ISMIR 2019 protocols A (random negatives) and B (year-matched), SpotGenTrack, and the Billboard file. A: audio .708 → +year .739 → **+year+artist .763** (+2.4, *p* = 7e-8), above the published .7504. **B: release year becomes useless (*p* = .66) while artist history still adds +2.6 (*p* = 3e-5, 10/10 folds).** That contrast is the paper's strongest single piece of evidence. |
| `r5_tilburg.py` | `r5_*.csv` | Head-to-head with Du (2022, Tilburg MSc) — their exact models, inputs and protocol, 50 seeds of 70/30. Our re-implementation matches their published numbers (RF 00s .846 vs .84; 10s .829 vs .82; LR 00s .821 vs .82; 10s .807 vs .79). Adding our feature: RF 00s **.867**, 10s **.861**. Better on **50/50 seeds in all four cells**, corrected *p* between 6e-6 and 9e-8. |
| `r6_robustness.py` | `r6_*.csv` | `StratifiedGroupKFold` by normalised artist — no test artist ever appears in training. The gains survive (Kaggle +5.0 unchanged, 25/25, *p* = 1e-13). Plus the cold-start split: on Kaggle, **known artists are 96.6 % hits**, so artist history there largely means "known artist ⇒ hit" (+15.0), and it does nothing for the 33,438 new artists (+0.4, n.s.) where era helps instead (+2.2). On ISMIR B it does help new artists (+1.4, *p* = .002). |
| `fetch_audio_full.py` + `r7_fft_full.py` | `r7_*.csv` | Deezer clips for 35,370 / 40,560 songs; FFT on top of the main model .825 → .839 (+1.4, 25/25, *p* = 4e-9), all of it on new artists (+1.8). |
| `r8_future_work.py` | `r8_*.csv` | The Future Work items that can be run: artist matching (99.99 % of hits already found; featured credits +0.1 n.s.), genre control for FFT, rolling-origin decay, lyrics on SpotGenTrack. See FINDINGS.md. |
| `r9_metrics.py` | `r9_*.csv` | Precision / recall / F1 of the hit class. Kaggle F1 .787 → .831 (+4.3, 25/25, *p* = 7e-15); SpotGenTrack recall .083 → .252, F1 .147 → .358 (5/5, *p* = 7e-7). |
| `make_figures.py` | `paper/fig_*.pdf` | The paper figures. |

### Layer B — `legacy/` : exploratory. Historical only.

`external_validation_ismir.py`, `phase5_test.py`, `run_ismir.py`, `run_phase0.py`,
`run_spotgentrack.py` / `_v2`, `run_zhao.py`, plus their figures and result files.
Do **not** quote their printed results — several are exactly the claims CV overturned.
They are not maintained (written for the old flat layout).

The five old modules that the pipeline still needs were moved **into `repro/`** during the
2026-10-27 cleanup, so `repro/` is now self-contained:

```
repro/feature_experiments.py  Kaggle loader + artist-history feature (used by common, r1)
repro/phase0_harness.py       BASE_FEATURES_A, mcnemar_test (used by common)
repro/run_ismir_v2.py         ISMIR loader (used by r4, r6)
repro/phase2_fft.py           builds cache/fft_features.csv (3,562-song subset; R3)
repro/phase5_richaudio.py     builds cache/rich_features.csv (R3)
```

`repro/common.py` still calls `os.chdir(ROOT)` on import (the older loaders use paths
relative to the root, now `data/...`). You can launch the scripts from anywhere, but
expect that side effect.

---

## 4. Data

See [`DATA.md`](DATA.md) for download links and exact placement. Short version: the
working tree is **~53 GB / 1.7 M files** and none of it is in git. Datasets 1 (Kaggle
decade files) and 2 (Hot 100 weekly history) download themselves on first run; ISMIR
(18 GB, Zenodo) and SpotGenTrack (740 MB, Mendeley) are manual.

**Before any ISMIR work, build `ext_cache/ismir_essentia.pkl`** by running
`repro/run_ismir_v2.py` once: it parses 592 Essentia descriptors out of 18 GB of JSON, after
which every ISMIR experiment loads in seconds instead of hours.

---

## 5. Numbers worth memorising

| | |
|---|---|
| Kaggle main result | 0.782 → **0.832** (+5.0 pts, 25/25 folds, *p* = 7e-17); AUC 0.859 → 0.918 |
| Transferable estimate | **+2.6 pts** — ISMIR 2019 protocol B, year-matched |
| Beats published | ISMIR A **0.763** vs Zangerle et al.'s 0.7504 |
| Head-to-head | better than Du (2022) on **50/50** seeds, all four cells |
| Artist-grouped CV | gains unchanged (Kaggle +5.0, 25/25, *p* = 1e-13) |
| Artist match rate | 52.1 % of benchmark songs — but **99.99 % of hits**; the unmatched songs are flops by artists who never charted (R8) |
| Hit rate by prior hits | 0 → 39.3 % (n = 33,438) · 1–2 → 91.0 % · 3–5 → 97.2 % · 6–10 → 99.6 % · 11+ → 98.9 % |
| Prior hits, hits vs flops | 3.21 vs 0.037 |
| Multiplicity | ~60 *p*-values; every load-bearing result survives Bonferroni (*p* < 8e-4) |

**Known-wrong numbers — do not reintroduce.** The Billboard external validation
(0.5906 → 0.5871 "on 3,397 songs" — computed on zero-filled missing features); the
"paper 0.877" dashed line in the early slides (that is Middlebrook & Sheik 2019, on a
*different* dataset); the SpotGenTrack 0.8222 labelled "our 2 features alone" (it is
artist + low-level audio); the rich-librosa +3.8 single-split gain; the claim that
PCA 40 is an optimum. All are already corrected in the current paper and deck — this
list exists only so they do not creep back in.

---

## 6. Reproducing

```bash
pip install numpy pandas scikit-learn scipy xgboost matplotlib librosa soundfile requests

python repro/r1_kaggle_core.py     # main benchmark result
python repro/r2_models_time.py     # 5 classifiers + temporal hold-out + SHAP
python repro/r3_audio_subset.py    # audio subset: FFT / rich librosa / PCA / ablations
python repro/r4_external.py        # ISMIR 2019 A+B, SpotGenTrack, Billboard
python repro/r5_tilburg.py         # head-to-head vs Du (2022), 50 seeds
python repro/r6_robustness.py      # artist-grouped CV + cold start
python repro/make_figures.py       # the three paper figures

cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

R1 / R2 / R5 run in minutes. R3 needs `cache/` (hours to build the first time: Deezer
downloads plus librosa extraction; it resumes if interrupted). R4 / R6 need
`ext_cache/ismir_essentia.pkl`. Seeds are fixed at 42 unless a script says otherwise.
Logs land in `paper_results/logs/`.

The deck is a single self-contained HTML file: open
`presentation/hit_prediction_deck.html` in a browser and print to PDF to regenerate it.

---

## 7. What is left to do

**Housekeeping — hours, not research.**

1. **Publish the code.** The paper's Reproducibility section promises "the scripts in
   `repro/`" but gives no URL. A public repo, with that URL added to the section, closes
   the gap. This repository is already prepared for it — `.gitignore` excludes all 53 GB
   of data — but it has never been pushed anywhere.
2. ~~Prune Layer B~~ — done 2026-10-27: load-bearing modules moved into `repro/`, the rest
   into `legacy/`, raw data into `data/`, logs into `paper_results/logs/`.
3. **Pick a venue and format.** Currently IEEE conference style, 10 pages (no page limit) — fine for a
   workshop; a journal version would want the items below.
4. **Licensing.** `LICENSE.md` exists (MIT code; datasets keep their own terms, Kaggle is GPL-3.0).

**Research.** R8 (2026-10-27) ran the cheap Future Work items — see FINDINGS.md:
artist matching cannot be improved (99.99 % of hits already matched); the whole-benchmark
FFT gain is a **genre shortcut** (vanishes on a genre-matched sample); rolling-origin decay;
lexical lyrics help a little on SpotGenTrack. What remains needs new data:

5. **Cold start** with non-audio signals (label, collaborators, prior streams, social),
   always with a genre-matched control.
6. **Semantic / emotional lyrics and learned audio embeddings** (no PyTorch/TF on this
   machine today).
7. **A benchmark whose negatives are sampled independently of artist, genre and date.**

---

## 8. Reading order for a newcomer

1. `paper/main.pdf` — the Discussion section first, then Results.
2. `paper_results/FINDINGS.md` — the honest ledger: every claim, its *p*-value, and the
   ones that died. Shorter than the paper and more useful for deciding what to trust.
3. `repro/common.py` — understand `cv_eval()` and `make_pipeline()`; everything else is a
   thin caller.
4. `repro/r4_external.py` — protocol B is the argumentative core of the paper.
5. `presentation/hit_prediction_deck.pdf` — the ten-minute version, for context.
