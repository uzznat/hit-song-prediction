# Speaker notes

Deck: `presentation/hit_prediction_deck.html` (43 slides, arrow keys), PDF alongside.

The deck is **one story** from start to finish: plain-words slides say what we did, and the detailed slides
(charts, tables) appear where they happen in the story. Every slide ends with a blue arrow "→ ...": read it aloud
to move to the next slide ("so next we did this").

Every number comes from the paper (`paper/main.pdf`) and `paper_results/`.

---

## The story in one minute

We wanted to know whether a computer can guess that a song will be a hit. We started with **the sound**: 78 %
correct. Then we added **who sings it** (has the artist had hits before?) and **when** (the decade): 83 %. We
tried to break that result and it held. Then we found a **trap** in the data: the non-hits had been picked from
artists with no hit. So we tested on other data, with a fair test: **+2.6 points** remain, our honest number. Then
we tried to add more sound features; every time the gain was a **trap** (titles, luck, genre). Finally our
supervisor had us study **time series**: they do not help before release, but after release the first three
weeks predict a top-10 hit very well.

---

## Slide by slide

### The question (slides 1–3)

**1 · Title.** "We'll tell you the story of our project: what we tried, what worked, and what was a trap."

**2 · The question.** A hit = a song that entered the Billboard Hot 100. The computer learns from songs whose
answer we know, then is tested on songs it has never seen. Random guessing: 50 %.

**3 · The four datasets** (table). Kaggle (40,560 songs, our main set), ISMIR 2019 (peer-reviewed, used as a
check), SpotGenTrack (100,000 tracks, few hits), and the Billboard year-end file, dropped because 2,911 of 3,397
rows had no audio features.

### Chapter 1 · The sound (slides 4–8)

**4 · We started with the sound.** 13 Spotify numbers per song → **78 %**. One song in five still wrong.

**5 · How the computer learns.** Random Forest = 80 small decision trees that ask questions and vote. We test
25 times with different hidden songs.

**6 · The classifiers** (table). Random Forest is the main model: same configuration as the reference paper,
never tuned. XGBoost is the best (84.5 %) but not significantly better. SVM, k-NN and logistic regression show
that the gain does not depend on the model.

**7 · The other algorithms.** To measure sound (librosa, FFT, PCA, Essentia), for text and explanation (TF-IDF,
SHAP), and for testing (McNemar, corrected t-test, Bonferroni).

**8 · How every gain is tested.** The old single split, and above all 25 splits (5×5 cross-validation) with a
corrected t-test, because the plain t-test is over-confident when splits overlap.

### Chapter 2 · Who and when (slides 9–11)

**9 · Our idea.** *Who*: has the artist had hits **before** (earlier decades only)? *When*: the decade.

**10 · Result** (chart). 78 → 83 % (+5 points), better in 25 of 25 tests.

**11 · Why "who" works** (chart). No earlier hit: 39 % hits. One or two earlier hits: 91 %.

### Chapter 3 · Is it luck? (slides 12–16)

**12 · We tried to break the result**: overview of the four checks.

**13 · Every decade** (chart): +3.7 to +6.6.

**14 · Every classifier** (chart): +4.6 to +5.7. We keep the Random Forest. SHAP: artist history is feature #1.

**15 · Predicting the future** (chart): trained on older decades, tested on an unseen decade: +1.9 to +3.4.

**16 · Against the only other study (Du, 2022)** (chart): we redo their method, match their numbers, and with our
feature win 50 times out of 50.

### Chapter 4 · A trap in the data (slides 17–25)

**17 · The trap.** A non-hit must come from an artist with **no hit** that decade, in a niche genre. 97 % of songs
by already-known artists are hits here: our +5 is too optimistic.

**18 · Another dataset (ISMIR).** 76.3 %, above the best published 75.0 %. But the year alone already gives 70 %:
hits there are older songs.

**19 · ISMIR numbers** (chart): year alone 70.2 %, audio 70.8 %, audio + year 73.9 %, audio + year + artist 76.3 %.

**20 · The fair test.** Every hit against a non-hit from the same year: the year becomes useless, the artist still
adds **+2.6** — our honest number.

**21 · Fair-test numbers** (chart): +2.6, p = 3·10⁻⁵, 10 of 10 tests.

**22 · SpotGenTrack** (charts). 81 % non-hits, so we look at AUC: .690 → .777. We find 25 % of hits instead of 8 %.

**23 · New artists**: overview.

**24 · Artists never seen**: the gain stays (+5.0; +2.9 on fair ISMIR; 75.9 % still above the published result).

**25 · Brand-new artists** (table): artist history cannot help (+0.4); the era helps (+2.2).

### Chapter 5 · Trying more, catching false gains (slides 26–32)

**26 · More clues**: titles (artefact), detailed sound (luck), small details (nothing). All removed.

**27 · The numbers** (chart): one test versus 25 tests. Detailed sound drops from +3.8 to +0.9.

**28 · The FFT**: it looked good (+1.4), but it was recognising the genre.

**29 · The genre test** (chart): with genre known, FFT lowers accuracy; with genres balanced, +0.1. Genre itself is
not used as a feature: Spotify assigns it afterwards.

**30 · Models get old** (table): audio alone falls from .809 to .689 three decades later; artist history ages
well, FFT ages badly.

**31 · Three more checks**: 99.99 % of hits found (no matching problem); lyrics help a little (promising); F1
confirms the gains.

**32 · Every feature we tried** (table): kept, removed, control only or promising, and why. **Keep this slide in
mind for questions.**

### Chapter 6 · Time series, our supervisor's reading plan (slides 33–36)

**33 · Reading the chart as a signal**: trend, seasonality, residual; ARIMA forecasts best.

**34 · Details** (table): new songs per week = mostly trend; Christmas songs = pure seasonality, the FFT finds the
52-week cycle. ARIMA, ARIMA + Fourier, Holt-Winters kept; SARIMA and machine learning removed.

**35 · Can time series predict a hit?** **Before release: no** (+0.1, the trend only knows the decade). **After
release: yes** (the first three weeks predict a top-10 hit, AUC .89). Two different questions; both answers hold.

**36 · Details of both experiments** (table): AUC .59 without the weeks; .75 / .85 / .89 / .92 with 1 / 2 / 3 / 4
weeks; .87 on songs not yet in the top 10; .85 when testing after 2000.

### The end (slides 37–43)

**37 · Statistics**: ~60 tests, Bonferroni; every key result stays solid.

**38 · What we corrected**: 8 claims from our earlier slides were wrong.

**39 · What is new**: artist history existed (Interiano 2018, Middlebrook 2019); we add the tests, the head-to-head,
the external validation and the audit of traps.

**40 · Scoreboard: how much we improved each dataset** (table). Spotify Hit Predictor 78.2 → 83.2 %;
Du's method 84.6 → 86.7 % (2000s) and 82.9 → 86.1 % (2010s), above his published 84 / 82 %; ISMIR 73.9 → 76.3 %,
above the best published 75.04 %; ISMIR fair test 68.2 → 70.8 %; SpotGenTrack AUC .690 → .777 (hits found
8 % → 25 %); Hot 100 after release AUC .59 → .89 with the first three weeks.

**41 · What we proved.** Closing line: a hit is mostly **who sings it and when** — and, once it is out, **how its
first weeks go**.

**42 · Limitations and next steps**: new artists, deep learning, lyrics, a dataset without traps.

**43 · Thank you, questions.**

---

## Likely questions

- **"Why no deep learning?"** We asked which *inputs* carry signal, at equal model. All five classifiers show the
  same gain (slide 14). Deep learning on raw audio is future work.
- **"Why Random Forest and not XGBoost, which is better?"** To stay comparable with the reference paper, and the
  difference is not significant (slide 6).
- **"Isn't artist history cheating?"** No: we count only *earlier* decades, known on release day. But on Kaggle the
  gain is inflated by how the dataset was built (slide 17), which is why +2.6 (slide 21) is our honest number.
- **"What about a new artist?"** Artist history cannot help; the era helps a little (slide 25). This is the main
  limitation.
- **"Why remove the FFT if it is significant?"** It recognises genre, and genre was used to pick the non-hits. With
  balanced genres it adds nothing (slide 29).
- **"What are time series good for in your problem?"** Not for guessing a hit before release; but once a song is
  out, its first weeks predict a top-10 hit (slides 35–36).
