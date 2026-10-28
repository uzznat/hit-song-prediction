"""Rebuilds presentation/hit_prediction_deck.html from the previous version: keeps the slides that are
still correct, fixes the outdated ones, adds the algorithm / feature ledgers, R7-R10 and the time-series part,
and orders everything in six parts.

    py presentation/build_deck.py
"""
import os
import re

import story

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "deck_v1_source.html")      # the 23-slide version of 2026-09-15
OUT = os.path.join(HERE, "hit_prediction_deck.html")
s = open(SRC, encoding="utf-8").read()

head = s[:s.index('<div class="stage" id="stage">')]
tail = s[s.index("</div></div>\n\n<nav"):]
old = re.findall(r"<section.*?</section>", s, re.S)
assert len(old) == 23, len(old)
O = {i + 1: sec for i, sec in enumerate(old)}


def R(text, a, b):
    assert text.count(a) == 1, a[:70]
    return text.replace(a, b)


def foot(sec, label):
    return re.sub(r'data-foot="[^"]*"', f'data-foot="{label}"', sec, count=1)


P1, P2, P3, P4, P5, P6, PC = ("Part 1 · Problem & data", "Part 2 · Method", "Part 3 · Results on the benchmark",
                              "Part 4 · Is it real?", "Part 5 · What we tried and removed",
                              "Part 6 · Time series", "Conclusion")

KEEP = '<span class="pill k-both">keep</span>'
DROP = '<span class="pill k-bad">remove</span>'
CTRL = '<span class="pill k-era">control only</span>'
MAYBE = '<span class="pill k-artist">promising</span>'
BASE = '<span class="pill" style="color:var(--ink2)">baseline</span>'

# --------------------------------------------------------------------------- CSS additions
head = R(head, "</style>", """  .t.dense th { padding: 0 10px 8px 0; }
  .t.dense td { font-size: 16px; padding: 6px 10px 6px 0; line-height: 1.28; }
  .t.dense td.n { font-size: 15px; }
  .roadmap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px 28px; margin-top: 34px; }
  .roadmap .step { border-top: 1px solid var(--line); padding-top: 14px; display: flex; flex-direction: column; gap: 8px; }
  .roadmap .no { font: 800 44px/1 var(--display); color: var(--muted); }
  .roadmap .ttl { font: 700 24px/1.15 var(--display); }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chips span { font: 500 15px/1 var(--mono); padding: 8px 10px; border: 1px solid var(--line); border-radius: 4px; background: var(--raise); color: var(--ink2); }
""" + story.CSS + """</style>""")

# --------------------------------------------------------------------------- new slides
roadmap = f"""<section class="slide" data-foot="Roadmap">
  <p class="eyebrow">Roadmap</p>
  <h2>The project in six steps</h2>
  <div class="roadmap">
    <div class="step"><span class="no">1</span><span class="ttl">Problem &amp; data</span><p class="small">Why published accuracies cannot be compared; the four datasets we use.</p></div>
    <div class="step"><span class="no">2</span><span class="ttl">Method</span><p class="small">The starting features, the two signals we add, the algorithms, and how every gain is tested.</p></div>
    <div class="step"><span class="no k-both">3</span><span class="ttl">Results on the benchmark</span><p class="small">+5.0 points with every classifier, every decade, and against the only prior model.</p></div>
    <div class="step"><span class="no k-artist">4</span><span class="ttl">Is it real?</span><p class="small">How the benchmark was built, independent datasets, unseen and new artists.</p></div>
    <div class="step"><span class="no k-bad">5</span><span class="ttl">What we tried and removed</span><p class="small">Titles, rich audio, FFT, genre, lyrics: what we keep, what we drop, and why.</p></div>
    <div class="step"><span class="no k-era">6</span><span class="ttl">Time series</span><p class="small">Components and forecasting of the Hot 100, and predicting big hits from their first weeks.</p></div>
  </div>
</section>"""

start = f"""<section class="slide" data-foot="{P2}">
  <p class="eyebrow">Method · the starting point</p>
  <h2>We start from the 13 audio descriptors that Spotify computes for every song</h2>
  <div class="body cols-2" style="margin-top:34px;align-items:start">
    <div class="stack">
      <div class="chips"><span>danceability</span><span>energy</span><span>key</span><span>loudness</span><span>mode</span><span>speechiness</span><span>acousticness</span><span>instrumentalness</span><span>liveness</span><span>valence</span><span>tempo</span><span>duration</span><span>time signature</span></div>
      <p class="small rule">Each song is a row of 13 numbers; the label is <strong>1 = hit</strong> (on the weekly Hot 100 in its decade), <strong>0 = non-hit</strong>. The task is <strong>binary classification</strong>.</p>
      <p class="tiny">The dataset also has <em>chorus_hit</em> and <em>sections</em>; we test them in Part 5 and they add nothing.</p>
    </div>
    <div class="stack">
      <p class="src">BASELINE · RANDOM FOREST ON THE 13 DESCRIPTORS</p>
      <p class="big">78.2<span class="unit">%</span></p>
      <p class="small">accuracy, 5×5 cross-validation, 40,560 songs (49 % hits). Every gain in this talk is measured <strong>against this number</strong>.</p>
    </div>
  </div>
</section>"""

algos = f"""<section class="slide" data-foot="{P2}">
  <p class="eyebrow">Method · the classifiers</p>
  <h2>One main model, four others to check that the result is not tied to it</h2>
  <div class="body" style="grid-template-columns:1fr;margin-top:24px">
    <table class="t dense">
      <thead><tr><th style="width:17%">Algorithm</th><th style="width:33%">Why we use it</th><th class="n" style="width:9%">Audio</th><th class="n" style="width:11%">+ decade + artist</th><th style="width:30%">Verdict</th></tr></thead>
      <tbody>
        <tr><td><strong>Random Forest</strong><br><span class="tiny">80 trees, 8 features per split</span></td><td>Configuration of the reference paper (Middlebrook &amp; Sheik), so results compare; learns non-linear rules (decade × audio). <strong>Never tuned</strong>: a gain must come from the features.</td><td class="n">78.0</td><td class="n">83.2</td><td>{KEEP} main model for every experiment</td></tr>
        <tr><td><strong>XGBoost</strong></td><td>The strongest learner for tabular data.</td><td class="n">78.8</td><td class="n">84.5</td><td>Best average, but ties the Random Forest on the canonical split (p = .87). Reported, not needed.</td></tr>
        <tr><td><strong>SVM (RBF kernel)</strong></td><td>A kernel method, very different from trees.</td><td class="n">77.3</td><td class="n">82.8</td><td>Same gain; slower, no probabilities.</td></tr>
        <tr><td><strong>k-NN</strong> (k = 25)</td><td>Distance-based: “a song is a hit if similar songs were”.</td><td class="n">75.0</td><td class="n">79.6</td><td>Same gain; weaker overall.</td></tr>
        <tr><td><strong>Logistic regression</strong></td><td>Linear and interpretable; also one of Du’s models.</td><td class="n">72.7</td><td class="n">78.2</td><td>Weakest: the relations are not linear. Kept for the head-to-head.</td></tr>
      </tbody>
    </table>
  </div>
  <p class="small" style="margin-top:6px">All five gain <strong>+4.6 to +5.7 points</strong>: the improvement comes from the features, not the model. External datasets use a 200-tree Random Forest. <span class="src">5-fold CV · Table IV</span></p>
</section>"""

tools = f"""<section class="slide" data-foot="{P2}">
  <p class="eyebrow">Method · the other algorithms</p>
  <h2>Feature extraction, explanation and statistics</h2>
  <div class="body cols-3" style="margin-top:30px;align-items:start;gap:30px">
    <div class="stack rule">
      <p class="src">TURNING AUDIO INTO FEATURES</p>
      <p class="small"><strong>librosa</strong>: 185 descriptors from 30-second clips (timbre, rhythm, harmony).</p>
      <p class="small"><strong>FFT</strong>: 31 spectral descriptors (energy per frequency band).</p>
      <p class="small"><strong>PCA</strong>: compresses the 185 librosa descriptors to 40 components.</p>
      <p class="small"><strong>Essentia</strong>: the 592 descriptors of the ISMIR dataset.</p>
    </div>
    <div class="stack rule">
      <p class="src">TEXT AND EXPLANATION</p>
      <p class="small"><strong>TF-IDF + logistic regression</strong>: turns the song title into one probability, trained out-of-fold so the test songs are never seen.</p>
      <p class="small"><strong>TreeSHAP</strong>: shows which features the Random Forest relies on: artist history #1, decade #2, instrumentalness #3.</p>
    </div>
    <div class="stack rule">
      <p class="src">TESTING EVERY GAIN</p>
      <p class="small"><strong>McNemar</strong>: compares two models song by song on one split.</p>
      <p class="small"><strong>Corrected t-test</strong> (Nadeau–Bengio): compares two models over 25 folds without over-confidence.</p>
      <p class="small"><strong>Bonferroni</strong>: ~60 tests, so a result needs p &lt; 8·10⁻⁴ to count as solid.</p>
    </div>
  </div>
</section>"""

fft = f"""<section class="slide" data-foot="{P5}">
  <p class="eyebrow">Audit · spectral features (FFT) on the whole benchmark</p>
  <h2>FFT looked like a real gain for new artists — it was the <span class="k-bad">genre</span></h2>
  <div class="body cols-2">
    <div>
      <div class="chart" id="c-genre"></div>
      <p class="src" style="margin-top:8px">20,610 songs with a clip and a Spotify genre · 5×5 CV · paper Table X</p>
    </div>
    <div class="stack" style="gap:14px">
      <p class="small">On 35,370 songs, FFT adds <strong>+1.4</strong> (25/25 folds, p = 4·10⁻⁹), all of it on new artists.</p>
      <p class="small">But Latin, classical, jazz, world and metal artists make <strong>50 % of the non-hits and 1.9 % of the hits</strong> — by construction. Audio can recognise genre (FFT: 49.9 % vs 26.1 % by chance).</p>
      <p class="small rule">With genre known, FFT <strong class="k-bad">lowers</strong> accuracy (−1.1). On a <strong>genre-matched sample</strong> (as many hits as non-hits in every genre): <strong>+0.1, p = .84</strong>. {DROP}</p>
      <p class="tiny">Genre alone lifts accuracy to 92.8 %, but Spotify assigns genres afterwards, so we use it only as a control.</p>
    </div>
  </div>
</section>"""

decay = f"""<section class="slide" data-foot="{P5}">
  <p class="eyebrow">Further experiment · how fast does a model age?</p>
  <h2>Train on one decade, test k decades later: artist history ages well, FFT does not</h2>
  <div class="body cols-2" style="align-items:start;margin-top:28px">
    <table class="t dense">
      <thead><tr><th>k</th><th class="n">pairs</th><th class="n">audio</th><th class="n">+ artist</th><th class="n">+ FFT</th><th class="n">+ artist + FFT</th></tr></thead>
      <tbody>
        <tr><td>0 (same decade)</td><td class="n">6</td><td class="n">.809</td><td class="n">.837</td><td class="n">.831</td><td class="n"><strong>.859</strong></td></tr>
        <tr><td>1 decade later</td><td class="n">5</td><td class="n">.765</td><td class="n">.796</td><td class="n">.782</td><td class="n"><strong>.810</strong></td></tr>
        <tr><td>2</td><td class="n">4</td><td class="n">.728</td><td class="n k-artist"><strong>.775</strong></td><td class="n">.736</td><td class="n"><strong>.775</strong></td></tr>
        <tr><td>3</td><td class="n">3</td><td class="n">.689</td><td class="n k-artist"><strong>.740</strong></td><td class="n k-bad">.673</td><td class="n">.728</td></tr>
        <tr><td>4</td><td class="n">2</td><td class="n">.666</td><td class="n k-artist"><strong>.734</strong></td><td class="n k-bad">.661</td><td class="n">.718</td></tr>
        <tr><td>5</td><td class="n">1</td><td class="n">.664</td><td class="n k-artist"><strong>.754</strong></td><td class="n k-bad">.599</td><td class="n">.706</td></tr>
      </tbody>
    </table>
    <div class="stack">
      <p class="small">Audio-only accuracy falls from <strong>.809</strong> in the same decade to <strong>.689</strong> three decades later: what a hit sounds like changes.</p>
      <p class="small">The advantage of <strong class="k-artist">artist history grows</strong> with the gap (+2.8 → +5.1 points).</p>
      <p class="small"><strong class="k-bad">FFT helps only nearby</strong> and hurts from three decades on: it encodes era-specific sound and genre.</p>
      <p class="tiny rule">Random Forest, 35,370 songs with a clip. Cells average 6 to 1 decade pairs, so no significance test. Paper Table XI.</p>
    </div>
  </div>
</section>"""

checks = f"""<section class="slide" data-foot="{P5}">
  <p class="eyebrow">Further experiments · artist matching, lyrics, hit-class metrics</p>
  <h2>Three more checks</h2>
  <div class="body cols-3" style="margin-top:30px;align-items:start;gap:30px">
    <div class="stack rule">
      <p class="src">ARTIST MATCHING</p>
      <p class="mid">99.99<span class="unit">%</span></p>
      <p class="small">of hits are found in the chart history. The songs that do not match are non-hits by artists who never charted — not a matching error.</p>
      <p class="small">Stricter matching −0.5 (p = 4·10⁻⁴); counting featured artists +0.1 (n.s.). {KEEP} the original rule.</p>
    </div>
    <div class="stack rule">
      <p class="src">LYRICS · SPOTGENTRACK, 79,420 TRACKS</p>
      <p class="mid">+.023<span class="unit">AUC</span></p>
      <p class="small">Six lexical descriptors (word count, vocabulary richness, ...) on top of decade + artist: AUC .774 → .797, accuracy .808 → .814 (p = .002).</p>
      <p class="small">The only content feature that adds to artist history on data not selected by genre. {MAYBE}</p>
    </div>
    <div class="stack rule">
      <p class="src">PRECISION · RECALL · F1 OF THE HIT CLASS</p>
      <p class="mid">.787 → .831</p>
      <p class="small">F1 on the benchmark (p = 7·10⁻¹⁵): precision and recall both rise, so the gain is not a trade-off.</p>
      <p class="small">SpotGenTrack: audio finds 8 % of the hits; with decade + artist, <strong>25 %</strong> at the same precision (F1 .147 → .358).</p>
    </div>
  </div>
</section>"""

ledger = f"""<section class="slide" data-foot="{P5}">
  <p class="eyebrow">Summary · every feature we tried</p>
  <h2>What we keep, what we removed, and why</h2>
  <div class="body" style="grid-template-columns:1fr;margin-top:18px">
    <table class="t dense">
      <thead><tr><th style="width:27%">Feature</th><th style="width:27%">Result</th><th style="width:12%">Verdict</th><th>Why</th></tr></thead>
      <tbody>
        <tr><td><strong>Artist history</strong></td><td>+2.9 alone · +2.6 on ISMIR, years matched</td><td>{KEEP}</td><td>Survives every control</td></tr>
        <tr><td><strong>Era</strong> (decade / year)</td><td>+2.0 · useless once years are matched</td><td>{KEEP}</td><td>Helps new artists; on ISMIR only a date shortcut</td></tr>
        <tr><td>Song title (TF-IDF)</td><td>+6.7, p = 4·10⁻²⁰</td><td>{DROP}</td><td>Artefact of how non-hits were collected</td></tr>
        <tr><td>Title length · chorus_hit + sections</td><td>+0.2 · 0.0, not significant</td><td>{DROP}</td><td>No signal</td></tr>
        <tr><td>Rich librosa audio (185 → PCA 40)</td><td>+3.8 on one split → +0.9, n.s.</td><td>{DROP}</td><td>The single-split gain was luck</td></tr>
        <tr><td>FFT spectral (31)</td><td>+1.4 → +0.1 genre-matched</td><td>{DROP}</td><td>Genre shortcut</td></tr>
        <tr><td>Spotify artist genre</td><td>+10.1</td><td>{CTRL}</td><td>Assigned afterwards: would leak</td></tr>
        <tr><td>Low-level audio (207)</td><td>≈ 0 on top of artist history</td><td>{DROP}</td><td>Nothing beyond artist history</td></tr>
        <tr><td>Lyrics, lexical (6)</td><td>AUC +.023, accuracy +0.6 (p = .002)</td><td>{MAYBE}</td><td>Small; indicative only</td></tr>
        <tr><td>Market trend (vs last decade's hits)</td><td id="trend-res">see Part 6</td><td id="trend-verdict"></td><td id="trend-why"></td></tr>
      </tbody>
    </table>
  </div>
</section>"""

ts1 = f"""<section class="slide" data-foot="{P6}">
  <p class="eyebrow">Time series · the Hot 100 as a signal (our supervisor's reading plan)</p>
  <h2>Trend, seasonality, residual — and which forecasting algorithms work</h2>
  <div class="body cols-2e" style="margin-top:22px;align-items:start;gap:34px">
    <div class="stack" style="gap:12px">
      <p class="src">COMPONENTS · 68 YEARS OF WEEKLY CHARTS</p>
      <p class="small"><strong>Debuts per week</strong>: mostly <strong>trend</strong> (strength .72): 11 in 1959, 5.5 in 2001, 11 again with streaming. Seasonality weak (.11): January is quiet. Largest residual spike: a <strong>rule change</strong> in December 1998.</p>
      <p class="small"><strong>Holiday songs on the chart</strong>: almost pure seasonality. The <strong>FFT</strong> finds the 52-week cycle and its harmonics (26, 17.4, 13 weeks) without being told; a Fourier fit follows the streaming-era peaks (R² = .94).</p>
      <p class="tiny rule">Notebook <span class="src">timeseries/time_series_hot100.ipynb</span>: STL decomposition, ACF/PACF, ADF/KPSS, Fourier terms.</p>
    </div>
    <div>
      <p class="src" style="margin-bottom:8px">FORECASTING DEBUTS 12 MONTHS AHEAD · 11 ORIGINS · MASE (LOWER IS BETTER)</p>
      <table class="t dense">
        <tbody>
          <tr><td>ARIMA(1,1,2)</td><td class="n">1.82</td><td>{KEEP}</td></tr>
          <tr><td>ARIMA + Fourier terms</td><td class="n">1.82</td><td>{KEEP}</td></tr>
          <tr><td>Holt-Winters (damped)</td><td class="n">1.85</td><td>{KEEP}</td></tr>
          <tr><td>SARIMA (seasonal)</td><td class="n">1.89</td><td>{DROP}</td></tr>
          <tr><td>Gradient boosting / random forest</td><td class="n">1.94 / 1.99</td><td>{DROP}</td></tr>
          <tr><td>Seasonal naive</td><td class="n">2.32</td><td>{BASE}</td></tr>
        </tbody>
      </table>
      <p class="tiny" style="margin-top:8px">The three best are indistinguishable (p = .97) and beat the seasonal naive forecast at 10 of 11 origins (p = .006). SARIMA: the series has almost no season to model. Trees: 800 points are too few, and they cannot extrapolate a trend.</p>
    </div>
  </div>
</section>"""

ts2 = f"""<section class="slide" data-foot="{P6}">
  <p class="eyebrow">Time series for hit prediction</p>
  <h2 id="ts2-title">TS2_TITLE</h2>
  TS2_BODY
</section>"""

proved = f"""<section class="slide" data-foot="{PC}">
  <p class="eyebrow">Summary · what we proved</p>
  <h2>What we managed to prove</h2>
  <div class="body cols-2e" style="margin-top:22px;align-items:start;gap:36px">
    <div class="stack" style="gap:11px">
      <p class="src k-both" style="letter-spacing:.16em">PROVED · SURVIVES EVERY CONTROL</p>
      <p class="small"><strong>1 · Artist history is a real signal.</strong> +5.0 points on the benchmark (25/25 folds, p = 7·10⁻¹⁷) and <strong>+2.6 on independent data</strong> with years matched (p = 3·10⁻⁵).</p>
      <p class="small"><strong>2 · It beats published work.</strong> 76.3 % vs the best published 75.04 % on ISMIR 2019; better than Du (2022) on 50 of 50 splits.</p>
      <p class="small"><strong>3 · It is not a trick of one model or of known artists.</strong> Five classifiers gain +4.6 to +5.7; the gain stays +5.0 when test artists are never seen in training.</p>
      <p class="small"><strong>4 · For a new artist, the era helps</strong> (+2.2, p = 2·10⁻⁹) — artist history cannot.</p>
      <p class="small"><strong>5 · After release, the first 3 weeks predict a top-10 hit</strong> (AUC .89; .87 on songs not yet in the top 10).</p>
    </div>
    <div class="stack" style="gap:11px">
      <p class="src k-bad" style="letter-spacing:.16em">DISPROVED · GAINS THAT WERE NOT REAL</p>
      <p class="small"><strong>Song titles</strong> (+6.7): an artefact of how non-hits were collected.</p>
      <p class="small"><strong>Rich audio</strong> (+3.8 on one split): luck — +0.9, not significant over 25 folds.</p>
      <p class="small"><strong>FFT spectral features</strong> (+1.4): a genre shortcut — +0.1 on a genre-matched sample.</p>
      <p class="small"><strong>Release year</strong> on ISMIR (70 % alone): a date shortcut — useless once years are matched.</p>
      <p class="small"><strong>Market trends</strong> before release: +0.1, they only carry the era.</p>
      <p class="small rule"><strong>In the sound itself, nothing survives every control.</strong></p>
    </div>
  </div>
  <p class="small" style="margin-top:12px;font-size:19px;color:var(--ink)">What predicts a hit is mostly <span class="k-artist">who sings it</span> and <span class="k-era">when</span> — and, once it is out, <span class="k-both">how its first weeks go</span>.</p>
</section>"""

# --------------------------------------------------------------------------- fixes to kept slides
O[4] = R(O[4], "Artist names normalised (accents, “feat.”, “&amp;” removed); 52 % of benchmark songs match the chart history.",
         "Artist names normalised (accents, “feat.”, “&amp;” removed). 99.99 % of hits are found in the chart history.")
O[6] = R(O[6], "<p class=\"tiny rule\">Canonical split:",
         "<p class=\"tiny\">F1 of the hit class .787 → .831 (precision .758 → .828, recall .819 → .833).</p>\n      <p class=\"tiny rule\">Canonical split:")
O[7] = R(O[7], "Keep that 0.037 in mind — it comes back on slide 12.", "Keep that 0.037 in mind — it comes back in Part 4.")
O[9] = R(O[9], "<p class=\"small rule\">TreeSHAP ranks",
         "<p class=\"small rule\">We keep the <strong>Random Forest</strong> as the main model: comparable with prior work, untuned, and within noise of XGBoost.</p>\n      <p class=\"small\">TreeSHAP ranks")
O[15] = R(O[15], "The 207 low-level audio features add nothing on top of it.",
          "The 207 low-level audio features add nothing on top of it. Hits found: 8 % → 25 % (F1 .147 → .358).")
O[18] = R(O[18], "<p class=\"tiny rule\">chorus_hit + sections, often suspected of leakage, add nothing.</p>",
          "<p class=\"tiny rule\">chorus_hit + sections, often suspected of leakage, add nothing. FFT survives here — the next slide shows why it should not.</p>")
O[19] = R(O[19], "<li>Artist-grouped gains</li>", "<li>Artist-grouped gains</li>\n        <li>The genre control (p ≤ 5·10⁻⁷)</li>")
O[19] = R(O[19], "<li>ISMIR cold-start gains (p = .002, .003)</li>",
          "<li>ISMIR cold-start gains (p = .002, .003)</li>\n        <li>Lyrics on SpotGenTrack (p = .002)</li>")
O[20] = R(O[20], "Six claims from our earlier slides did not survive re-checking", "Eight claims did not survive re-checking")
O[20] = R(O[20], """        <tr><td style="color:var(--ink2)">Artist history “works on new songs too”</td>""",
          """        <tr><td style="color:var(--ink2)">FFT +1.4 for new artists (our own R7)</td><td>A <strong>genre shortcut</strong>: +0.1 on a genre-matched sample</td></tr>
        <tr><td style="color:var(--ink2)">“Artist names match for only 52 %”</td><td><strong>99.99 % of hits</strong> match; the rest are non-hits by artists who never charted</td></tr>
        <tr><td style="color:var(--ink2)">Artist history “works on new songs too”</td>""")
O[20] = R(O[20], '<div class="body" style="grid-template-columns:1fr;margin-top:26px">\n    <table class="t">',
          '<div class="body" style="grid-template-columns:1fr;margin-top:18px">\n    <table class="t dense">')
O[21] = R(O[21], "<p class=\"small\">Middlebrook &amp; Sheik (2019) list an artist past-performance field among chart-derived fields — but their final inputs are not stated and its contribution is never isolated or tested.</p>",
          "<p class=\"small\"><strong>Interiano et al. (2018)</strong>: a “superstar” indicator (did the artist chart recently?) improves a Random Forest on 500,000 UK songs.</p>\n      <p class=\"small\"><strong>Middlebrook &amp; Sheik (2019)</strong> list an artist past-performance field, but never isolate or test it.</p>\n      <p class=\"small\">Neither tests significance, compares on the same data, or checks for sampling artefacts.</p>")
O[22] = R(O[22], "<li>Artist names match for only 52 % of benchmark songs</li>",
          "<li>Artists matched by name, not by a canonical ID (homonyms)</li>\n        <li>No deep-learning model: we compare inputs at equal model</li>")
O[22] = R(O[22], """        <li>Canonical artist IDs (MusicBrainz / Spotify) to raise the match rate</li>
        <li>Features that help <strong>first-time artists</strong>: label, collaborators, social media</li>
        <li>Rolling-origin evaluation across all decades</li>
        <li>Lyrics and learned audio embeddings, under the same audit</li>""",
          """        <li>Features that help <strong>first-time artists</strong>: label, collaborators, early streams — always with a genre-matched control</li>
        <li>Semantic lyrics and learned audio embeddings, under the same audit</li>
        <li>Early-trajectory forecasting (Chapter 6) with streaming data</li>""")
O[23] = R(O[23], "Song titles, release dates and rich audio on one split looked better still — and were artefacts or chance.",
          "Song titles, release dates, FFT and rich audio on one split looked better still — and were artefacts, genre or chance.")

# --------------------------------------------------------------------------- order and part labels
# The talk is one story: plain-words slides (story.py) with the detailed slides and charts placed where they
# happen in the story. Every slide ends with the step that comes next.
O[1] = R(O[1], "Artist history, era context and shortcut features — tested on four datasets, with every gain checked for chance and for artefacts.",
         "The story of our project: what we tried, what worked, and what turned out to be a trap.")
S, E = story.STORY, story.ENDING
C0, C1, C2, C3, C4, C5, C6, CE = (story.C0, story.C1, story.C2, story.C3, story.C4, story.C5, story.C6, story.CE)
thanks = story.slide(CE, "Thank you", "Thank you — questions?",
                     '<div class="body" style="grid-template-columns:1fr;margin-top:34px"><p class="say">'
                     'Lior Shalon &amp; Nathanael Uzzan · every number comes from the scripts in '
                     '<span class="src">repro/</span> and the paper.</p></div>')
order = [
    (O[1], None, None),
    (S[0], C0, "To answer, we need songs whose answer we already know. We used four datasets."),
    (O[3], C0, "We started with the biggest one, and with the most obvious clue: the sound."),
    (S[1], C1, "How does the computer learn, and how do we know the score is not luck?"),
    (S[2], C1, "Which algorithms exactly, and why these?"),
    (algos, C1, "We also used other algorithms: to measure the sound, to explain, and to test."),
    (tools, C1, "And this is exactly how every improvement is tested."),
    (O[5], C1, "The sound alone was not enough. So we asked: what else makes a hit?"),
    (S[3], C2, "We added these two clues to the 13 sound numbers."),
    (S[4], C2, "Why does “who” work so well?"),
    (S[5], C2, "+5 points is a lot. Too much? We tried hard to prove it was luck."),
    (S[6], C3, "First check, in detail: every decade."),
    (O[8], C3, "Second check: every classifier."),
    (O[9], C3, "Third check: predicting a decade the model has never seen."),
    (O[10], C3, "Fourth check: against the only other study on this dataset."),
    (O[11], C3, "It was not luck. But then we read how the dataset had been built..."),
    (S[7], C4, "So we needed a dataset built differently."),
    (S[8], C4, "The numbers on that dataset."),
    (O[13], C4, "So we made the test fair: every hit against a non-hit from the same year."),
    (S[9], C4, "The numbers of the fair test."),
    (O[14], C4, "One more dataset: SpotGenTrack, 100,000 songs, where hits are rare."),
    (O[15], C4, "And for artists the computer has never seen?"),
    (S[10], C4, "The numbers: artists never seen in training."),
    (O[16], C4, "And the numbers for brand-new artists."),
    (O[17], C4, "Next, we tried to give the computer more musical information."),
    (S[11], C5, "All these gains, measured on one test and on 25 tests."),
    (O[18], C5, "One clue still looked good: the FFT."),
    (S[12], C5, "The numbers of the genre test."),
    (fft, C5, "We also checked how fast each model gets old."),
    (decay, C5, "And three last checks."),
    (checks, C5, "Everything we tried, in one table."),
    (ledger, C5, "Then our teacher asked us to study time series."),
    (S[13], C6, "The details: the components and the forecasting algorithms."),
    (ts1, C6, "Then the big question: can time series help predict a hit?"),
    (S[14], C6, "The details of both experiments."),
    (ts2, C6, "We ran about 60 statistical tests in total, so we checked which results are solid."),
    (O[19], CE, "On the way, we also corrected our own earlier claims."),
    (O[20], CE, "So what is really new compared with other studies?"),
    (O[21], CE, "So, what did we prove in the end?"),
    (E[0], CE, "And what we cannot do yet."),
    (O[22], CE, None),
    (thanks, CE, None),
]


def with_next(sec, text):
    sec = re.sub(r'\n  <p class="next">.*?</p>', "", sec)
    if text:
        i = sec.rindex("</section>")
        sec = sec[:i] + f'  <p class="next">→ {text}</p>\n' + sec[i:]
    return sec


slides = [with_next(foot(sec, lab) if lab else sec, nxt) for sec, lab, nxt in order]

# --------------------------------------------------------------------------- R10 numbers (filled by fill_r10)
import json
R10 = os.path.join(HERE, "r10_deck.json")
r10 = json.load(open(R10, encoding="utf-8")) if os.path.exists(R10) else None

charts = """
  dotRows("c-genre", { min: 76, max: 95, step: 2, tick: tickPct, fmt: pct, labelW: 250, rowH: 62, rows: [
    { label: "main model", value: 82.8, color: BO },
    { label: "main + FFT", value: 84.1, color: css("--bad") },
    { label: "main + genre", value: 92.8, color: E },
    { label: "main + genre + FFT", value: 91.7, color: css("--bad") },
    { label: "genre-matched: main", value: 79.0, color: BO },
    { label: "genre-matched: + FFT", value: 79.1, color: css("--bad"), bold: true } ] });
"""
tail = R(tail, 'nameB: "cross-validation", labelW: 250, rowH: 62,', 'nameB: "cross-validation", labelW: 250, rowH: 50,')
# a chart whose slide is not in this deck must not stop the script
tail = R(tail, """    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
    document.getElementById(host).appendChild(s);""", """    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
    const hostEl = document.getElementById(host);
    if (hostEl) hostEl.appendChild(s);""")
tail = R(tail, "  /* ---------- navigation ---------- */", charts + story.CHARTS + "\n  /* ---------- navigation ---------- */")
tail = re.sub(r'<span id="counter" aria-live="polite">1 / \d+</span>',
              f'<span id="counter" aria-live="polite">1 / {len(slides)}</span>', tail)

out = head + '<div class="stage" id="stage"><div class="deck" id="deck">\n\n' + \
    "\n\n".join(f"<!-- {i} -->\n{sec}" for i, sec in enumerate(slides, 1)) + "\n\n" + tail
LEDGER_TODO = '<td id="trend-res">see Part 6</td><td id="trend-verdict"></td><td id="trend-why"></td>'
if r10:
    out = out.replace("TS2_TITLE", r10["title"]).replace("TS2_BODY", r10["body"]).replace(LEDGER_TODO,
                                                                                            r10["ledger_row"])
else:
    out = out.replace("TS2_TITLE", "Results pending").replace("TS2_BODY", "")
open(OUT, "w", encoding="utf-8").write(out)
print(f"{len(slides)} slides written to {OUT}")
