# Literature facts verified for the paper (2026-09-14)

Checked against full texts (Tilburg PDF, ISMIR 2019 PDF, arXiv 1908.08609) and Crossref DOIs.

## "paper 0.877" in the deck = Middlebrook & Sheik 2019 (arXiv:1908.08609)
- NOT the Kaggle dataset: their own "SpotifyBillboard" set, 1985-2018, ~12k hits + 12k random non-hits,
  20k/2k/2k split. RF: 80 estimators, max_features=8, min_samples_split=2, gini -> test 0.877, val 0.887.
- Their 27 listed track fields include weeks / rank / score (chart-derived) -> leakage risk; inputs not stated.
- Our "frozen baseline RF" = their configuration applied to Kaggle. The deck's dashed 0.877 line compares different data.
- Feature sets "A (13 audio) / B (+chorus_hit, sections)" are our own naming, not from a publication.

## Tilburg thesis = Du, J. L. (2022) "Spotify between decades: Hit or Miss predictor", MSc DSS, Tilburg
- http://arno.uvt.nl/show.cgi?fid=160898 ; supervisor A. Alishahi.
- 2000s (5,872 rows) and 2010s (6,398) only, one model per decade, R.
- 70/30 split caTools::sample.split, set.seed(1); no CV for LR/RF (LVQ: 10-fold x3).
- Inputs: "all the numeric values" (13 audio + chorus_hit + sections per appendix).
- RF: ntree=450, mtry=2, maxnodes=30. LR cutoff 0.5.
- Table 10 accuracy: LR 00s 82 %, 10s 79 %; RF 00s 84 %, 10s 82 %; LVQ 77 / 74 %.
- Internal inconsistencies (prose vs table precision/recall), "base model 89 %" taken from Middlebrook (other data).

## Zangerle, Huber, Voetter, Yang (ISMIR 2019), pp. 319-326 (author order: Zangerle, Huber, Voetter, Yang)
- MSD songs with year + Essentia; hits = exact match to Hot 100 1954-2018 -> 5,832; non-hit pool 89,235
  (fuzzy title ratio <= 40 to every hit); 5,832 non-hits sampled at random -> 11,664 balanced.
- Task = REGRESSION on peak position (non-hit = 150); accuracy = predicted rank < 100. 5-fold CV.
- Wide-and-deep net; wide part includes release YEAR (year improves every experiment by 12-13 %).
- Best: HL+LL+chords+rhythm RMSE 55.45, MAE 43.84, accuracy 75.04 %. HL 72.08, LL 66.94, best linear 73.91.

## Others (BibTeX in the paper's refs.bib)
- Dimolitsas, Kantarelis, Fouka 2023 SpotHitPy arXiv:2301.07978 (~18.5k songs, ~86 %).
- Zhao, Harvey, Cameron, Hopfgartner, Gillet, iConference 2023, LNCS pp. 303-311, doi 10.1007/978-3-031-28035-1_21
  (Top-10 prediction, RF 89.1 % / AUC 0.91).
- Rompolas et al., AIAI 2024 workshops, IFIP AICT, pp. 314-324, doi 10.1007/978-3-031-63227-3_22.
- Nofer, Nimani, Hinz 2026 DeepHits, MAKE 8(3):58, doi 10.3390/make8030058 (92,517 SpotGenTrack tracks).
- SpotGenTrack = Martin-Gutierrez et al., IEEE Access 8:39361-39374, 2020, doi 10.1109/ACCESS.2020.2976033 (NOT Stanford).
- Yang et al. ICASSP 2017 doi 10.1109/ICASSP.2017.7952230; Herremans et al. JNMR 2014 doi 10.1080/09298215.2014.881888;
  Pachet & Roy ISMIR 2008 pp. 355-360; Ni et al. MML@NIPS 2011; Dietterich 1998 doi 10.1162/089976698300017197;
  Breiman 2001 doi 10.1023/A:1010933404324; Chen & Guestrin KDD 2016 doi 10.1145/2939672.2939785;
  McFee et al. SciPy 2015 doi 10.25080/Majora-7b98e3ed-003; Bogdanov et al. ISMIR 2013 pp. 493-498;
  Bertin-Mahieux et al. ISMIR 2011 pp. 591-596; Kaufman et al. TKDD 2012 doi 10.1145/2382577.2382579;
  Geirhos et al. Nat. Mach. Intell. 2020 doi 10.1038/s42256-020-00257-z.
- Unconfirmed: Zangerle arXiv version, Zhao year range / LNCS volume, DeepHits split, Ni et al. pages.
