"""
Deck v3 (2026-10-29, after the supervisor's review): a shorter main story that a newcomer can follow, with
the algorithm drawn input -> output, the decade and artist-history inputs explained, the new results (ablation,
everything together, PCA, lyrics, first weeks, survival ranking), and an APPENDIX holding every detail slide.

Reuses the look, charts and navigation of deck v2 (deck_v2_source.html, built by build_deck.py) and moves
its detail slides to the appendix.

    py presentation/build_deck_v3.py      ->  presentation/hit_prediction_deck.html
"""
import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "deck_v2_source.html")
OUT = os.path.join(HERE, "hit_prediction_deck.html")
IMG = os.path.join(HERE, "img")

src = open(SRC, encoding="utf-8").read()
i0 = src.index('<div class="stage" id="stage">')
i1 = src.rindex("</section>") + len("</section>")
head, tail = src[:i0], src[i1:]
parts = re.split(r"<!-- (\d+) -->\n", src[i0:i1])
OLD = {int(parts[k]): parts[k + 1].strip() for k in range(1, len(parts), 2)}
OLD[1] = parts[0][parts[0].index("<section"):].strip() if 1 not in OLD else OLD[1]


def R(text, a, b):
    assert a in text, a[:70]
    return text.replace(a, b)


def img(name, style=""):
    b64 = base64.b64encode(open(os.path.join(IMG, name + ".png"), "rb").read()).decode()
    return f'<img class="figw" alt="" src="data:image/png;base64,{b64}" style="{style}">'


def slide(foot, eyebrow, title, body, src=""):
    s = f'<p class="src">{src}</p>' if src else ""
    return f"""<section class="slide" data-foot="{foot}">
  <p class="eyebrow">{eyebrow}</p>
  <h2>{title}</h2>
  {body}
  {s}
</section>"""


def box(head_, text, cls=""):
    return f'<div class="bx {cls}"><p class="bx-h">{head_}</p><div class="bx-t">{text}</div></div>'


ARROW = '<div class="ar">→</div>'

CSS = """
<style>
  .figw { display:block; width:100%; background:#fff; border-radius:8px; padding:10px 12px; }
  .flow { display:grid; align-items:stretch; gap:0; margin-top:22px; }
  .ar { display:flex; align-items:center; justify-content:center; font:700 30px/1 var(--display); color:var(--muted); padding:0 6px; }
  .bx { background:var(--raise); border:1px solid var(--line); border-radius:8px; padding:14px 16px; display:flex; flex-direction:column; gap:8px; }
  .bx.in { border-color:var(--base); } .bx.ft { border-color:var(--era); } .bx.md { border-color:var(--ink2); }
  .bx.out { border-color:var(--both); background:color-mix(in srgb, var(--both) 14%, var(--raise)); }
  .bx-h { font:700 19px/1.15 var(--display); margin:0; }
  .bx-t { font-size:16px; line-height:1.38; color:var(--ink2); }
  .bx-t strong { color:var(--ink); }
  .q { font:500 15px/1.5 var(--mono); color:var(--ink2); }
  .q b { color:var(--ink); }
  .oh { border-collapse:collapse; margin:4px 0; }
  .oh td { border:1px solid var(--line); padding:3px 8px; text-align:center; font:500 14px/1.2 var(--mono); }
  .oh td.on { background:var(--era); color:#fff; font-weight:700; }
  .divider h2 { font-size:64px; }
  .tick { display:inline-block; width:6px; height:18px; background:var(--artist); margin-right:3px; border-radius:1px; }
</style>
"""
head = head.replace("</head>", CSS + "</head>") if "</head>" in head else CSS + head

F0, F1, F2, F3, F4, F5, F6, FE, FA = ("The question", "1 · The algorithm", "2 · Who and when",
                                      "3 · Where the power comes from", "4 · A trap in the data",
                                      "5 · Which gains are real", "6 · Lyrics and time series", "The end",
                                      "Appendix")

# --------------------------------------------------------------------------- new slides
pipeline = slide(F1, "The algorithm · input → output", "What goes in, what comes out", f"""
  <div class="flow" style="grid-template-columns:1.05fr auto 1.25fr auto 1fr auto .9fr">
    <div class="stack" style="gap:10px">
      {box("Song audio", "Spotify Web API", "in")}
      {box("Release date", "which decade", "in")}
      {box("Artist name", "+ Billboard Hot 100 since 1958", "in")}
    </div>
    {ARROW}
    <div class="stack" style="gap:10px">
      {box("13 sound numbers", "danceability, energy, loudness, acousticness, …", "ft")}
      {box("6 decade boxes", "60s 70s 80s 90s 00s 10s — one is 1", "ft")}
      {box("1 artist-history number", "log(1 + Hot 100 songs in <strong>earlier</strong> decades)", "ft")}
    </div>
    {ARROW}
    {box("Random Forest", "<strong>80 decision trees</strong>. Each asks yes/no questions about the 20 numbers and votes. 8 inputs tried per question. <strong>Never tuned.</strong>", "md")}
    {ARROW}
    {box("Output", "<strong>P(hit)</strong> between 0 and 1<br><br>≥ 0.5 → <strong>hit</strong><br>&lt; 0.5 → <strong>not a hit</strong>", "out")}
  </div>
  <p class="say-s" style="margin-top:22px"><strong>Input:</strong> one song = 20 numbers, all known on the day it comes out. <strong>Output:</strong> will it enter the Billboard Hot 100? <strong>Learned from</strong> 32,000 songs whose answer we know; <strong>tested on</strong> 8,000 songs it never saw.</p>""",
                 "Paper · Fig. 1")

example = slide(F1, "The algorithm · one real song, step by step", "<em>Vogue</em> (Madonna, 1990), a song the model never saw", f"""
  <div class="flow" style="grid-template-columns:1fr auto 1.08fr auto 1.25fr auto .78fr">
    {box("1 · Raw data", "danceability <strong>0.737</strong><br>energy <strong>0.880</strong><br>loudness <strong>−6.4 dB</strong><br>acousticness <strong>0.004</strong><br>duration <strong>5.3 min</strong> … (13 values)<br><br>file: <strong>1990s</strong><br>artist: <strong>Madonna</strong>", "in")}
    {ARROW}
    {box("2 · 20 inputs", "13 sound values, unchanged<br><table class='oh'><tr><td>60s</td><td>70s</td><td>80s</td><td class='on'>90s</td><td>00s</td><td>10s</td></tr><tr><td>0</td><td>0</td><td>0</td><td class='on'>1</td><td>0</td><td>0</td></tr></table>Madonna: <strong>19</strong> Hot 100 songs before 1990 → h = log(1+19) = <strong>3.0</strong>", "ft")}
    {ARROW}
    {box("3 · Each tree asks", "<div class='q'>danceability ≤ 0.49? <b>no</b><br>acousticness ≤ 0.76? <b>yes</b><br>duration ≤ 5.7 min? <b>yes</b><br>artist history ≤ 0.35? <b>no</b><br>loudness ≤ −6.9 dB? <b>no</b><br>… 4 more …<br>→ leaf: 646 songs, <b>100 % hits</b></div>", "md")}
    {ARROW}
    {box("4 · Vote", "<strong>80 / 80</strong> trees say hit<br><br>P(hit) = <strong>1.00</strong><br><br>→ <strong>HIT</strong><br><span class='small'>(true: no. 1)</span>", "out")}
  </div>
  <p class="say-s" style="margin-top:18px">Every song goes through the same 4 steps. On the 8,113 test songs of this split, the answer is right <strong>83.3 %</strong> of the time. Training: each tree learns on its own random sample of the 32,447 training songs and, at each question, picks the input and threshold that best separate hits from non-hits.</p>""",
                "Paper · Fig. 2 · tree 1 of the trained forest")

decade = slide(F2, "Input · the decade (WHEN)", "What is the decade input, and how do we build it?", f"""
  <div class="body cols-2" style="grid-template-columns:.82fr 1.18fr;margin-top:20px;align-items:start;gap:30px">
    <div class="stack" style="gap:14px">
      {box("What it is", "The decade in which the song came out: 1960s, 70s, 80s, 90s, 2000s or 2010s.", "ft")}
      {box("How we build it", "The dataset comes as <strong>6 files</strong>, one per decade (<code>dataset-of-60s.csv</code> … <code>dataset-of-10s.csv</code>). The decade = the file. We turn it into <strong>6 yes/no boxes</strong>, exactly one is 1 (one-hot), because a decade is a category, not a number.", "ft")}
      {box("Why it helps", "Each decade is half hits, so alone it is a coin toss (50.3 %). But <strong>what a hit sounds like changes with the era</strong> (right): with the decade, the trees learn one rule per era. <strong>+2.0 points</strong>.", "ft")}
    </div>
    <div class="stack" style="gap:8px">{img("decade")}
      <p class="small">Median of three sound numbers for hits (blue) and non-hits (gray). A 2000s non-hit is louder than any 1960s–80s hit; hits were less acoustic than non-hits until the 1990s, slightly more after; hits were longer in the 1970s–90s, shorter after 2000.</p>
    </div>
  </div>""", "Paper · Sec. IV-B, Fig. 3")

ticks80 = "".join('<span class="tick"></span>' for _ in range(19))
history = slide(F2, "Input · artist history (WHO)", "What is artist history, and how do we compute it?", f"""
  <div class="body cols-2" style="grid-template-columns:1fr 1fr;margin-top:22px;align-items:start;gap:30px">
    <div class="stack" style="gap:14px">
      {box("Rule", "For a song, take its artist and count their <strong>different songs on the Billboard Hot 100 in earlier decades</strong>. Never the same decade, never the future, so it is known on release day.", "ft")}
      {box("Formula", "h = log(1 + count). The log squeezes big counts: 40 earlier hits is not twice 20.", "ft")}
      {box("For every artist", "Names are cleaned (lower case, accents, “feat.” removed) and matched to the full Hot 100 history (1958–2026). Computed separately for every artist and every decade.", "ft")}
    </div>
    <div class="stack" style="gap:16px">
      <p class="src">EXAMPLE · MADONNA</p>
      <p class="say-s"><strong>1980s</strong>: 19 Hot 100 songs<br>{ticks80}</p>
      <table class="t">
        <thead><tr><th>Song</th><th>Decade</th><th class="n">Earlier songs</th><th class="n">h</th></tr></thead>
        <tbody>
          <tr><td><em>Holiday</em></td><td>1980s</td><td class="n">0</td><td class="n">0.0</td></tr>
          <tr><td><em>Vogue</em></td><td>1990s</td><td class="n">19</td><td class="n k-artist"><strong>3.0</strong></td></tr>
          <tr><td><em>Hung Up</em></td><td>2000s</td><td class="n">44</td><td class="n">3.8</td></tr>
        </tbody>
      </table>
      <p class="say-s">A brand-new artist always gets <strong>h = 0</strong>.</p>
    </div>
  </div>""", "Paper · Eq. (1), Fig. 4")

ablation = slide(F3, "Ablation · remove one input at a time", "Where does the accuracy come from?", f"""
  <div class="stack" style="gap:10px;margin-top:14px">{img("ablation", "max-height:340px;width:auto;margin:0 auto")}
  <div class="body cols-3" style="gap:24px;align-items:start">
    <p class="say-s"><strong>Remove</strong> and retrain: artist history <strong>−3.0</strong>, decade <strong>−2.2</strong>, instrumentalness −1.6, duration −1.1 (all significant). Key, liveness, mode, tempo: 0.</p>
    <p class="say-s"><strong>Shuffle</strong> in the trained model: same top. Duration matters only through the era.</p>
    <p class="say-s"><strong>Alone</strong>: sound 78.2 %, artist 67.0 %, decade 50.3 %. <strong>Together 83.2 %</strong>: they complete each other.</p>
  </div></div>""", "Paper · Sec. V-B, Fig. 7")

together = slide(F3, "Train on everything at once?", "Putting every input in one model: higher score, wrong reasons", """
  <div class="body cols-2" style="grid-template-columns:1.05fr .95fr;margin-top:22px;align-items:start;gap:34px">
    <table class="t dense">
      <thead><tr><th>Model (35,370 songs)</th><th class="n">Accuracy</th></tr></thead>
      <tbody>
        <tr><td>Final model (20 inputs)</td><td class="n">82.5 %</td></tr>
        <tr><td><strong>Everything</strong> (8 blocks)</td><td class="n"><strong>88.3 %</strong></td></tr>
        <tr><td>&nbsp;&nbsp;− song title</td><td class="n k-bad">84.3 % (−4.0)</td></tr>
        <tr><td>&nbsp;&nbsp;− artist history</td><td class="n">86.4 % (−1.9)</td></tr>
        <tr><td>&nbsp;&nbsp;− 13 sound numbers</td><td class="n">87.1 % (−1.2)</td></tr>
        <tr><td>&nbsp;&nbsp;− FFT / − market trend</td><td class="n">87.9 % (−0.4 / −0.3)</td></tr>
        <tr><td>&nbsp;&nbsp;− decade</td><td class="n">88.1 % (−0.2, n.s.)</td></tr>
        <tr><td>&nbsp;&nbsp;− title length / − chorus_hit</td><td class="n">88.3 % (0)</td></tr>
      </tbody>
    </table>
    <div class="stack" style="gap:16px">
      <p class="say">Most of the extra comes from the <strong class="k-bad">song title</strong>, which only reveals how the non-hits were collected.</p>
      <p class="say-s">The rest comes from FFT and market trend, which fail the genre and era tests (chapter 5). With them in, the decade is redundant: all three describe the era.</p>
      <p class="say-s"><strong>PCA</strong> of the 13 sound numbers: replacing them by components only loses accuracy (−1.7), so we keep the raw numbers (appendix).</p>
      <p class="say k-both">→ We keep the 20 inputs.</p>
    </div>
  </div>""", "Paper · Table VI, Fig. 11")

artists = slide(F4, "Two different questions", "“Unseen” artist ≠ “new” artist", f"""
  <div class="body cols-2e" style="margin-top:26px;align-items:start;gap:34px">
    {box("Unseen artist — not in training", "The model never trained on this artist's songs, but the artist may have <strong>earlier hits</strong> (h &gt; 0). We test it by putting each artist only in training <em>or</em> only in test. <br><br>Result: the gain <strong>stays</strong> — +5.0 on the big dataset, +2.9 on the fair ISMIR test. So the model does not just recognise artists.", "ft")}
    {box("New artist — no earlier hit", "h = 0 by definition, so artist history <strong>cannot help</strong> (+0.4, not significant). Only the <strong>decade</strong> helps them (+2.2).<br><br>On the big dataset, the whole gain of artist history comes from the 18 % of songs by known artists — 97 % of them are hits.", "")}
  </div>""", "Paper · Sec. IV-D and V-E")

selection = slide(F5, "How we decided what to keep", "Every input went through the same four questions", f"""
  <div class="flow" style="grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto .9fr;margin-top:30px">
    {box("1 · Known on release day?", "No future, no own period.<br><span class='k-bad'>✗ genre tags</span> (assigned later)", "")}
    {ARROW}
    {box("2 · Gain under 25 tests?", "corrected p &lt; 0.05<br><span class='k-bad'>✗ title length, chorus_hit, librosa, trend</span>", "")}
    {ARROW}
    {box("3 · Survives the controls?", "same years · same genres · unseen artists<br><span class='k-bad'>✗ titles, FFT, release year</span>", "")}
    {ARROW}
    {box("4 · Survives Bonferroni?", "p &lt; 0.0008 over ~60 tests<br><span class='k-artist'>~ small lyrics test (SpotGenTrack)</span>", "")}
    {ARROW}
    {box("Kept", "decade<br>artist history<br>lyrics (where they exist)", "out")}
  </div>
  <p class="say-s" style="margin-top:24px">Genre and FFT were the strongest-looking inputs after the titles. We kept testing them until we understood why: they recognise the niche genres from which the dataset took its non-hits.</p>""",
                   "Paper · Fig. 5")

survival = slide(F5, "Survival ranking · every p-value", "What survived, what was a shortcut, what did not survive", """
  <div class="body cols-3" style="margin-top:20px;align-items:start;gap:26px">
    <div class="stack rule" style="gap:8px"><p class="src k-both">SURVIVED · LOW p + CONTROL PASSED</p>
      <p class="small">Decade + artist, big dataset: +5.0 · p = 7·10⁻¹⁷</p>
      <p class="small">Every classifier: +4.6 to +5.7 · p ≤ 4·10⁻⁵</p>
      <p class="small">Du's models + artist: 50/50 splits · p ≤ 6·10⁻⁶</p>
      <p class="small">Future decades: +1.9 to +3.4 · p ≤ 10⁻⁷</p>
      <p class="small">Artist, fair ISMIR test: +2.6 · p = 3·10⁻⁵</p>
      <p class="small">Lyrics, ISMIR, all controls: +4.4 · p = 3·10⁻⁵</p>
      <p class="small">First 3 weeks + context: AUC +.047 · p &lt; 10⁻⁶</p></div>
    <div class="stack rule" style="gap:8px"><p class="src" style="color:var(--artist-ink)">SHORTCUT · LOW p, CONTROL FAILED</p>
      <p class="small">Song title: +6.7 · p = 4·10⁻²⁰ → artefact</p>
      <p class="small">Artist genre: +10.1 · p = 8·10⁻²³ → assigned later</p>
      <p class="small">FFT: +1.4 · p = 4·10⁻⁹ → +0.1 when genres match</p>
      <p class="small">Release year, ISMIR: +3.1 · p = 2·10⁻⁴ → −0.1 when years match</p></div>
    <div class="stack rule" style="gap:8px"><p class="src k-bad">DID NOT SURVIVE · HIGH p</p>
      <p class="small">Lyrics, SpotGenTrack: p = .002 (indicative)</p>
      <p class="small">Market trend, per feature: p = .006</p>
      <p class="small">FFT on top of artist, subset: p = .008</p>
      <p class="small">Decade, small subset: p = .03</p>
      <p class="small">Rich librosa: p = .15 (one split said .002)</p>
      <p class="small">Artist history for new artists: p = .17</p>
      <p class="small">Title length .28 · trend .66 · chorus_hit .82</p></div>
  </div>
  <p class="small" style="margin-top:14px"><strong>Survives</strong> = significant under 25 tests, still significant after Bonferroni (p &lt; 8·10⁻⁴ over ~60 tests), and passes the control that removes its dataset's known shortcut.</p>""",
                  "Paper · Table VII")

lyrics = slide(F6, "Lyrics · ISMIR 2019, musiXmatch", "The lyrics: the only content input that survives every test", f"""
  <div class="stack" style="gap:10px;margin-top:12px">{img("lyrics", "max-height:295px;width:auto;margin:0 auto")}
  <div class="body cols-3" style="gap:24px;align-items:start">
    <p class="say-s"><strong>Inputs</strong>: 8 word statistics (length, richness, “I”, “you”, “love”…) + a bag-of-words model trained inside each fold.</p>
    <p class="say-s"><strong>Traps removed</strong>: only songs with lyrics on both sides; English-only; same years; no artist in both training and test.</p>
    <p class="say-s"><strong>+4.4 points</strong> after all controls (10/10, p = 3·10⁻⁵). Hits sing <em>love, oh, yeah</em>; non-hits <em>kill, hate, crime</em>: partly mainstream themes.</p>
  </div></div>""", "Paper · Sec. V-G, Fig. 9")

weeks = slide(F6, "After release · the first weeks on the chart", "What happens in the first 3 weeks decides a top-10 hit", f"""
  <div class="stack" style="gap:10px;margin-top:12px">{img("traj", "max-height:300px;width:auto;margin:0 auto")}
  <div class="body cols-3" style="gap:24px;align-items:start">
    <p class="say-s"><strong>First k weeks</strong> = the song's chart position in each of its first k weeks (101 = dropped off). <em>Vogue</em>: 39 → 28 → 12, then no. 1.</p>
    <p class="say-s">Future top-10s climb fast: median 73 → 53 → 39. The others stay near 80. Debut 61–80 and falling: <strong>3 %</strong>; climbing 30+ places: <strong>55 %</strong>.</p>
    <p class="say-s">The model uses the <strong>week-3 position</strong> and the <strong>climb</strong>, not the debut. AUC <strong>.89</strong> with 3 weeks, .59 without.</p>
  </div></div>""", "Paper · Sec. V-H, Fig. 10")

contrib = slide(FE, "Contributions", "What we did, how, and what changed compared with before", """
  <div class="body" style="grid-template-columns:1fr;margin-top:14px">
    <table class="t dense">
      <thead><tr><th style="width:17%">Contribution</th><th style="width:29%">How we do it</th><th style="width:27%">Before</th><th>Ours</th></tr></thead>
      <tbody>
        <tr><td><strong>Artist history + era</strong></td><td>earlier-decade Hot 100 count; decade one-hot; 25 tests</td><td>superstar flag, no test (Interiano 2018); sound 78.2 %</td><td class="k-both"><strong>83.2 %</strong> (+5.0, 25/25)</td></tr>
        <tr><td><strong>Head-to-head</strong></td><td>Du's exact method + our input, 50 splits</td><td>Du 2022: 84 % / 82 %</td><td class="k-both"><strong>86.7 % / 86.1 %</strong>, 50/50</td></tr>
        <tr><td><strong>External check</strong></td><td>ISMIR 2019, then same-year non-hits</td><td>best published 75.04 %</td><td class="k-both"><strong>76.3 %</strong>; fair test +2.6</td></tr>
        <tr><td><strong>Unseen / new artists</strong></td><td>artist-grouped tests; split by earlier hit</td><td>not tested</td><td>gain kept (+5.0, +2.9); new artists: era only</td></tr>
        <tr><td><strong>Ablation</strong></td><td>remove / shuffle one input; everything at once</td><td>not reported</td><td>artist −3.0, decade −2.2; “everything” = title artefact</td></tr>
        <tr><td><strong>Audit of shortcuts</strong></td><td>25 tests, genre- and year-matched controls</td><td>one split, no control</td><td>titles, FFT, year, librosa rejected</td></tr>
        <tr><td><strong>Lyrics</strong></td><td>musiXmatch words, all controls</td><td>lyrics in models on other data</td><td class="k-both"><strong>+4.4</strong> (p = 3·10⁻⁵)</td></tr>
        <tr><td><strong>First weeks</strong></td><td>first 3 chart positions + climb</td><td>early views predict online content</td><td class="k-both">AUC <strong>.89</strong> (.59 without)</td></tr>
      </tbody>
    </table>
  </div>""", "Paper · Table I")

links = slide(FE, "Code and data", "Everything is public", """
  <div class="body cols-2e" style="margin-top:30px;align-items:start;gap:36px">
    <div class="stack" style="gap:14px">
      <p class="src">CODE · EVERY NUMBER, ONE SCRIPT EACH</p>
      <p class="say"><strong>github.com/uzznat/hit-song-prediction</strong></p>
      <p class="say-s"><code>repro/</code> scripts → <code>paper_results/</code> tables → <code>paper/main.pdf</code>. Seeds fixed. <code>docs/DATA.md</code> says where each dataset goes.</p>
    </div>
    <div class="stack" style="gap:8px">
      <p class="src">DATA</p>
      <p class="small"><strong>Spotify Hit Predictor</strong> · github.com/fortyTwo102/The-Spotify-Hit-Predictor-Dataset (Kaggle: theoverman)</p>
      <p class="small"><strong>Billboard Hot 100</strong> · github.com/utdata/rwd-billboard-data</p>
      <p class="small"><strong>ISMIR 2019</strong> · zenodo.org/records/3258042</p>
      <p class="small"><strong>SpotGenTrack</strong> · data.mendeley.com/datasets/4m2x4zngny</p>
      <p class="small"><strong>musiXmatch lyrics</strong> · millionsongdataset.com/musixmatch</p>
      <p class="small"><strong>Audio previews</strong> · Deezer public API</p>
    </div>
  </div>""")

pca = slide(FA, "Appendix · PCA of the 13 sound numbers", "Which sound numbers matter? Variance is not what separates hits", f"""
  <div class="stack" style="gap:10px;margin-top:12px">{img("pca", "max-height:290px;width:auto;margin:0 auto")}
  <div class="body cols-3" style="gap:24px;align-items:start">
    <p class="say-s">PC1 = <strong>intensity</strong> (energy, loudness vs acousticness), 24 % of the variance, AUC .70. PC2 = <strong>mood</strong> (danceability, valence), AUC .62.</p>
    <p class="say-s">PC3–PC6 hold 33 % of the variance but separate hits like a coin (AUC .50–.52).</p>
    <p class="say-s">Best single numbers: instrumentalness (.74), danceability (.70). Components instead of raw numbers: <strong>−1.7</strong> → keep the raw numbers.</p>
  </div></div>""", "Paper · Appendix E, Fig. 11")

divider = """<section class="slide title-slide divider" data-foot="Appendix">
  <div><p class="eyebrow">Appendix</p><h2>The details behind every step</h2>
  <p class="lede" style="font-size:24px;max-width:46ch;margin-top:20px">Algorithms and tests used, every check of the main result, the trap in the data, the external datasets, the audit, the time-series study, the statistics and the corrections.</p></div>
</section>"""

# --------------------------------------------------------------------------- updates to reused slides
proved = OLD[42]
proved = R(proved, """      <p class="say-s">5 · <strong>After</strong> a song comes out, its first 3 weeks predict a top-10 hit (AUC .89). <strong>Before</strong>, time series add nothing.</p>""",
           """      <p class="say-s">5 · <strong>Lyrics</strong> add <strong>+4.4</strong> on a fair test, with unseen artists.</p>
      <p class="say-s">6 · <strong>After</strong> a song comes out, its first 3 weeks predict a top-10 hit (AUC .89). <strong>Before</strong>, time series add nothing.</p>""")
proved = R(proved, "<strong>In the sound itself, nothing survived every test.</strong>",
           "<strong>In the sound itself, nothing survived every test — the lyrics did.</strong>")
proved = R(proved, "A hit is mostly <span class=\"k-artist\">who sings it</span> and <span class=\"k-era\">when</span> — and, once it is out, <span class=\"k-both\">how its first weeks go</span>.",
           "A hit is mostly <span class=\"k-artist\">who sings it</span>, <span class=\"k-era\">when</span> and what it is about — and, once it is out, <span class=\"k-both\">how its first weeks go</span>.")
model = R(OLD[40], "Titles, detailed sound, FFT, genre, market trend, chorus_hit / sections: traps or noise. Lyrics: promising, one dataset only.",
          "Titles, detailed sound, FFT, genre, market trend, chorus_hit / sections: traps or noise. <strong>Lyrics</strong>: added where they exist (ISMIR: 77.5 %).")
score = R(OLD[41], """        <tr><td><strong>SpotGenTrack</strong>""",
          """        <tr><td><strong>ISMIR 2019</strong> · + lyrics, fair test, unseen artists</td><td>accuracy</td><td class="n">65.6 %</td><td class="n k-both"><strong>70.0 %</strong></td><td class="n">+4.4</td><td>songs with lyrics, English only</td></tr>
        <tr><td><strong>SpotGenTrack</strong>""")
checks = R(OLD[31], "The only content feature that adds to artist history on data not selected by genre. <span class=\"pill k-artist\">promising</span>",
           "Confirmed later on ISMIR with all controls (+4.4). <span class=\"pill k-both\">keep</span>")
ledger = R(OLD[32], """<tr><td>Lyrics, lexical (6)</td><td>AUC +.023, accuracy +0.6 (p = .002)</td><td><span class="pill k-artist">promising</span></td><td>Small; indicative only</td></tr>""",
           """<tr><td>Lyrics (ISMIR: words + 8 statistics)</td><td>+4.4, fair test, unseen artists (p = 3·10⁻⁵)</td><td><span class="pill k-both">keep</span></td><td>Survives every control</td></tr>""")
limits = R(OLD[43], "<li>Semantic lyrics and learned audio embeddings, under the same audit</li>",
           "<li>Lyrics for the main dataset; meaning of lyrics; learned audio embeddings — under the same audit</li>")

# --------------------------------------------------------------------------- order
main = [
    (OLD[1], None), (OLD[2], F0), (OLD[3], F0),
    (pipeline, F1), (OLD[4], F1), (example, F1), (OLD[8], F1),
    (decade, F2), (history, F2), (OLD[10], F2), (OLD[11], F2), (OLD[12], F2),
    (ablation, F3), (together, F3),
    (OLD[17], F4), (OLD[20], F4), (OLD[21], F4), (artists, F4),
    (selection, F5), (OLD[26], F5), (survival, F5),
    (lyrics, F6), (OLD[35], F6), (weeks, F6),
    (contrib, FE), (model, FE), (score, FE), (proved, FE), (links, FE), (OLD[44], FE),
]
appendix = [(divider, FA)] + [(s, FA) for s in [
    OLD[5], OLD[6], OLD[7], OLD[13], OLD[14], OLD[15], OLD[16], pca, OLD[18], OLD[19], OLD[22], OLD[23],
    OLD[24], OLD[25], OLD[27], OLD[28], OLD[29], OLD[30], checks, ledger, OLD[33], OLD[34], OLD[36],
    OLD[37], OLD[38], OLD[39], limits]]


def prep(sec, foot):
    sec = re.sub(r'\n\s*<p class="next">.*?</p>', "", sec)
    if foot is not None:
        sec = re.sub(r'data-foot="[^"]*"', f'data-foot="{foot}"', sec, count=1)
    return sec


slides = [prep(s, f) for s, f in main + appendix]
tail = re.sub(r'<span id="counter" aria-live="polite">1 / \d+</span>',
              f'<span id="counter" aria-live="polite">1 / {len(slides)}</span>', tail)
out = head + '<div class="stage" id="stage"><div class="deck" id="deck">\n\n' + \
    "\n\n".join(f"<!-- {i} -->\n{s}" for i, s in enumerate(slides, 1)) + "\n\n" + tail
open(OUT, "w", encoding="utf-8").write(out)
print(f"{len(main)} main + {len(appendix)} appendix = {len(slides)} slides -> {OUT}")
