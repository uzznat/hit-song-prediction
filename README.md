# What Really Predicts a Hit?
### Artist History, Era Context and Shortcut Features in Hit Song Prediction

Lior Shalon, Nathanael Uzzan — Technion, *Project in Artificial Intelligence*

**Deliverables**
* Paper (IEEE conference format, 7 pages): [`paper/main.pdf`](paper/main.pdf) ← source `paper/main.tex`
* Slides (23): [`presentation/hit_prediction_deck.pdf`](presentation/hit_prediction_deck.pdf) ← source `hit_prediction_deck.html`
* Audit of every claim: [`paper_results/FINDINGS.md`](paper_results/FINDINGS.md)
* Handoff / where to continue: [`docs/HANDOFF.md`](docs/HANDOFF.md)
* Getting the data: [`docs/DATA.md`](docs/DATA.md)
* Licensing: [`LICENSE.md`](LICENSE.md) — MIT for the code; the datasets keep their own terms (the Kaggle benchmark is GPL-3.0 and is not redistributed here)

---

## The claim in one paragraph

Two cheap, non-audio signals are added to the 13 Spotify audio descriptors: a
**leakage-free artist history** feature — `log(1 + #Hot-100 songs the artist placed in
strictly earlier periods)` — and the song's **era**. On the 40,560-song Spotify Hit
Predictor benchmark they take Random-Forest accuracy from 0.782 to **0.832**
(+5.0 pts, 25/25 CV folds, corrected *p* = 7·10⁻¹⁷). A faithful re-implementation of
the only academic study on that benchmark is improved on **50 out of 50** splits. Because
that benchmark excludes the artists of hits from its negative class, the feature is
re-validated on the peer-reviewed ISMIR 2019 dataset: **0.763** (above the published
0.750), and still **+2.6 pts** when hits and non-hits are matched year by year — a control
under which release year becomes useless. The same audit shows which gains are *not*
real: song titles (+6.7 pts) encode how the benchmark was assembled, and a
185-dimensional librosa representation that looked significant on one split is not
significant under repeated CV.

## Reproducing

```bash
pip install numpy pandas scikit-learn scipy xgboost matplotlib librosa soundfile requests
# then, after placing the data as described in docs/DATA.md:
python repro/r1_kaggle_core.py      # main benchmark result
python repro/r2_models_time.py      # 5 classifiers + temporal hold-out
python repro/r3_audio_subset.py     # audio subset, FFT / rich librosa / PCA
python repro/r4_external.py         # ISMIR 2019, SpotGenTrack, Billboard
python repro/r5_tilburg.py          # head-to-head vs Du (2022), 50 seeds
python repro/r6_robustness.py       # artist-grouped CV + cold start
python repro/make_figures.py        # the three paper figures
```

Each script writes CSVs into `paper_results/` and a log into `paper_results_r<N>.log`.
Seeds are fixed (42 unless stated). `repro/common.py` holds the shared loaders, the
sklearn pipelines and the two evaluation modes (canonical single split + repeated
stratified CV with the Nadeau–Bengio corrected resampled *t*-test).

## Layout

```
repro/            the canonical pipeline — every number in the paper comes from here
paper_results/    its output (CSVs) + FINDINGS.md, the claim-by-claim audit
paper/            main.tex, refs.bib, the 3 figures, main.pdf
presentation/     hit_prediction_deck.html → .pdf (23 slides)
docs/             DATA.md, HANDOFF.md, and the upstream datasets' own READMEs
*.py (root)       the exploratory phase; repro/ still imports 4 of them (see HANDOFF)
```

**Warning:** the loose `*.py` scripts in the root are the earlier exploratory phase.
Several of their conclusions were overturned by `repro/`. Trust `repro/` +
`paper_results/FINDINGS.md`, never a root script's own printout.
