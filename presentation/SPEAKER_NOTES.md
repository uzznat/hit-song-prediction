# Speaker notes

Deck: `presentation/hit_prediction_deck.html` (58 slides: **30 main + 28 appendix**, arrow keys), PDF alongside.
Built by `presentation/build_deck_v3.py` (reuses the look and charts of `deck_v2_source.html`).
Every number comes from the paper (`paper/main.pdf`) and `paper_results/`.

The main part is one story that someone who has never seen the project can follow. The appendix holds every
detail slide: open it only to answer a question.

---

## The story in one minute

We want to know whether a computer can tell, on release day, whether a song will enter the Hot 100. The
algorithm takes **20 numbers per song** (13 sound numbers, 6 decade boxes, 1 artist-history number) and a
**Random Forest of 80 trees** answers hit / not hit. Sound alone: 78 %. With **who** (artist history) and **when**
(decade): 83 %. The ablation shows where this comes from; training on everything gives more, but for the wrong
reasons. Then we found a **trap** in the dataset (non-hits were chosen by artist and genre), so we tested fairly on
another dataset: **+2.6**, our honest number. Every other input went through four questions; most were shortcuts or
luck. The **lyrics** survive every test (+4.4). And once a song is out, its **first 3 weeks** predict a top-10 hit.

---

## Main part (slides 1–30)

### The question (1–3)
**1 · Title.**
**2 · The question.** A hit = entered the Billboard Hot 100. Coin toss = 50 %.
**3 · Datasets.** Main: Spotify Hit Predictor (40,560 songs). Check: ISMIR 2019. Also SpotGenTrack.

### 1 · The algorithm (4–7)
**4 · What goes in, what comes out.** Read the diagram left to right: raw data → 20 numbers → Random Forest →
P(hit) → hit if ≥ 0.5. Learned from 32,000 songs with known answers, tested on 8,000 never seen.
**5 · We started with the sound.** 13 Spotify numbers alone: 78 %.
**6 · One real song step by step: *Vogue*.** (1) raw data; (2) the 20 numbers — decade 1990s = the 4th box is 1,
Madonna had 19 Hot 100 songs before 1990 → h = 3.0; (3) one tree's questions, ending in a leaf of 646 training songs,
all hits; (4) 80/80 trees vote hit → P = 1.00. *Training*: each tree learns on its own random sample, and at each
question picks the input and threshold that best separate hits from non-hits.
**7 · How every gain is tested.** 25 tests (5×5 cross-validation) + corrected t-test (+ Bonferroni later).

### 2 · Who and when (8–12)
**8 · The decade.** *What*: the release decade. *How*: the data comes as 6 files, one per decade; the decade = the
file; we make 6 yes/no boxes (one-hot) because a decade is a category. *Why it helps*: alone it is 50 % (each
decade is half hits), but what a hit sounds like changes with the era — show the chart: louder every decade,
acousticness and duration reverse. With the decade, trees learn a rule per era. +2.0.
**9 · Artist history.** Count the artist's different Hot 100 songs in *earlier* decades only; h = log(1 + count).
Madonna: *Holiday* 0, *Vogue* 3.0, *Hung Up* 3.8. New artist = 0. Computed for every artist and every decade.
**10 · Result.** 78 → 83 %, better in 25 of 25 tests.
**11 · Why "who" works.** No earlier hit: 39 % hits; one or two: 91 %.
**12 · Is it luck?** 5 algorithms, every decade, future decades, the only other study: it held.

### 3 · Where the power comes from (13–14)
**13 · Ablation.** Remove one input and retrain: artist −3.0, decade −2.2, instrumentalness −1.6, duration −1.1;
key, liveness, mode, tempo: 0. Shuffle one input: same top. Alone: sound 78, artist 67, decade 50; together 83.
**14 · Everything at once.** 88.3 %, but −4.0 when the song title is removed (artefact), and the rest is FFT and
trend, which fail the controls. PCA of the sound: components lose 1.7 points. → keep the 20 inputs.

### 4 · A trap in the data (15–18)
**15 · The trap.** Non-hits were chosen from artists with no hit, in niche genres: 97 % of songs by known artists
are hits. Our +5 is too optimistic.
**16–17 · Fair test (ISMIR, same years).** The year becomes useless; artist history still +2.6 (10/10, p = 3·10⁻⁵).
**18 · "Unseen" ≠ "new" artist.** Unseen = not in training (may have earlier hits): gain stays. New = no earlier hit
(h = 0): artist history cannot help; only the decade does. This removes the apparent contradiction.

### 5 · Which gains are real (19–21)
**19 · Four questions.** Known on release day? Gain under 25 tests? Survives the controls (same years, same genres,
unseen artists)? Survives Bonferroni? Genre and FFT looked strongest; they were recognising niche genres.
**20 · More clues.** Titles (artefact), detailed sound (luck), small details (nothing).
**21 · Survival ranking.** Three columns: survived (low p + control passed), shortcut (low p, control failed), did
not survive (high p). "Survives" is defined at the bottom.

### 6 · Lyrics and time series (22–24)
**22 · Lyrics.** On ISMIR with musiXmatch words. Traps removed: lyrics on both sides, English only, same years, no
artist in both training and test. +4.4 (p = 3·10⁻⁵). Words: love/oh/yeah vs kill/hate/crime — partly themes and
genres.
**23 · Time series.** Before release: no (+0.1, the trend only knows the decade). After release: yes.
**24 · The first 3 weeks.** "First k weeks" = the chart position in each of the first k weeks (*Vogue*: 39, 28, 12).
Future top-10s climb fast (73 → 53 → 39). The model uses the week-3 position and the climb. AUC .89 vs .59.

### The end (25–30)
**25 · Contributions:** how, before, ours. **26 · Final model.** **27 · Scoreboard** (new row: lyrics on ISMIR).
**28 · What we proved.** **29 · Code and data:** github.com/uzznat/hit-song-prediction + dataset links.
**30 · Thanks.**

---

## Appendix (31–58) — open only for questions

31 divider · 32 how the computer learns · 33 the five classifiers and why · 34 other algorithms (librosa, FFT, PCA,
TF-IDF, SHAP, McNemar, corrected t-test, Bonferroni) · 35 every decade · 36 every classifier · 37 future decades ·
38 head-to-head with Du · 39 PCA · 40–41 ISMIR · 42 SpotGenTrack · 43–45 unseen and new artists · 46 audit chart ·
47–48 FFT and the genre test · 49 model ageing · 50 three more checks · 51 every feature we tried · 52–54 time series
(components, forecasting, both experiments) · 55 statistics · 56 corrections to our earlier claims · 57 what is new
vs Interiano and Middlebrook · 58 limitations.

## Likely questions

- **"Is artist history cheating?"** No: earlier decades only (slide 9). The dataset inflates it (slide 15), hence +2.6.
- **"What is the decade, exactly?"** Slide 8: the file the song comes from, as 6 yes/no boxes.
- **"Why not train on everything?"** Slide 14: the extra comes from the title artefact and genre/era shortcuts.
- **"Why Random Forest, not XGBoost?"** Comparable with prior work, difference not significant (slide 33).
- **"Why no deep learning?"** We compare inputs at equal model; future work (slide 58).
- **"Why remove FFT if significant?"** It recognises genre (slides 47–48).
- **"New artists?"** Only the decade helps them (slides 18, 45).
