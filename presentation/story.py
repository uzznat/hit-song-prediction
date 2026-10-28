"""The story part of the deck: what we did, step by step, in plain words. Used by build_deck.py."""

OK = '<span class="pill k-both">✓ kept</span>'
NO = '<span class="pill k-bad">✗ removed</span>'


def slide(foot, eyebrow, title, body, nxt=None):
    n = f'\n  <p class="next">→ {nxt}</p>' if nxt else ""
    return f"""<section class="slide" data-foot="{foot}">
  <p class="eyebrow">{eyebrow}</p>
  <h2>{title}</h2>
  {body}{n}
</section>"""


def card(head, text, cls=""):
    return f'<div class="stack rule" style="gap:10px"><p class="say-h {cls}">{head}</p><p class="say-s">{text}</p></div>'


C0, C1, C2, C3, C4, C5, C6, CE = ("The question", "Chapter 1 · The sound", "Chapter 2 · Who and when",
                                  "Chapter 3 · Is it luck?", "Chapter 4 · A trap in the data",
                                  "Chapter 5 · Trying more, catching false gains", "Chapter 6 · Time series",
                                  "The end")

STORY = [
    slide(C0, "The question", "Can a computer guess, before a song comes out, whether it will be a hit?",
          f"""<div class="body cols-3" style="margin-top:40px;align-items:start">
    {card("What is a hit?", "A song that entered the <strong>Billboard Hot 100</strong>, the weekly chart of the most popular songs in the USA.")}
    {card("How the computer learns", "We show it thousands of songs whose answer we already know. It finds patterns. Then we test it on songs it has <strong>never seen</strong>.")}
    {card("What is a good score?", "Guessing at random gives <strong>50 %</strong> right, like tossing a coin. Everything we do is compared to that.")}
  </div>""", "We started with the most obvious clue: the sound of the song."),

    slide(C1, "Step 1", "We started with the sound",
          """<div class="body cols-2" style="margin-top:34px;align-items:start">
    <div class="stack">
      <p class="say">Spotify describes every song with <strong>13 numbers</strong>: how danceable, how energetic, how fast, how loud, how acoustic, how happy it sounds...</p>
      <div class="chips"><span>danceability</span><span>energy</span><span>tempo</span><span>loudness</span><span>acousticness</span><span>valence (happy)</span><span>+ 7 more</span></div>
    </div>
    <div class="stack">
      <p class="big">78<span class="unit">%</span></p>
      <p class="say-s">right, on 40,560 songs (half hits, half not). Much better than a coin, but <strong>1 song in 5 is still wrong</strong>.</p>
    </div>
  </div>""", "But how does the computer learn, and how do we know the score is not luck?"),

    slide(C1, "How it works", "How the computer learns, and how we test it",
          f"""<div class="body cols-2e" style="margin-top:34px;align-items:start">
    {card("A Random Forest = 80 small decision trees", "Each tree asks simple questions — <em>is it danceable? is it long? is it acoustic?</em> — and votes “hit” or “not hit”. The majority wins. We never tune it, so any improvement comes from the clues we give it, not from tweaking.")}
    {card("We test it 25 times", "We hide 1/5 of the songs, train on the rest, and guess the hidden ones. Then again with other hidden songs: <strong>25 tests</strong> in total. An improvement counts only if it appears again and again, and passes a statistical test.")}
  </div>""", "The sound alone was not enough. So we asked: what else makes a hit?"),

    slide(C2, "Step 2 · Our idea", "A hit is not only a sound — it is also <span class=\"k-artist\">who</span> sings it and <span class=\"k-era\">when</span>",
          f"""<div class="body cols-2e" style="margin-top:34px;align-items:start">
    {card("WHO · the artist's history", "Has this artist had hits <strong>before</strong>? We count their Hot 100 songs in <strong>earlier decades only</strong>, never the future — the computer could know this on the day the song comes out.", "k-artist")}
    {card("WHEN · the decade", "A hit in 1965 does not sound like a hit in 2015. Knowing the decade lets the computer use the right “taste” for each era.", "k-era")}
  </div>""", "We added these two clues to the 13 sound numbers."),

    slide(C2, "Step 2 · Result", "From 78 % to 83 % right",
          """<div class="body cols-2">
    <div><div class="chart" id="s-main"></div></div>
    <div class="stack">
      <p class="big k-both">+5<span class="unit">points</span></p>
      <p class="say-s">Better in <strong>25 out of 25</strong> tests. Each clue helps on its own; together they help the most.</p>
    </div>
  </div>""", "Why does “who” work so well?"),

    slide(C2, "Step 2 · Why it works", "One earlier hit changes everything",
          """<div class="body cols-2">
    <div><div class="chart" id="s-curve"></div><p class="src" style="margin-top:6px">Share of songs that are hits, by the artist's number of earlier hits</p></div>
    <div class="stack">
      <p class="say">Artist with <strong>no</strong> earlier hit: <strong>39 %</strong> of their songs are hits.</p>
      <p class="say">With <strong>1 or 2</strong> earlier hits: <strong class="k-artist">91 %</strong>.</p>
    </div>
  </div>""", "+5 points is a lot. Too much? We tried hard to prove it was luck."),

    slide(C3, "Step 3", "Is it luck? We tried to break it — it held",
          f"""<div class="body" style="grid-template-columns:repeat(4,1fr);margin-top:36px;align-items:start;gap:26px">
    {card("5 different algorithms", "Random Forest, XGBoost, SVM, k-NN, logistic regression: all gain <strong>+4.6 to +5.7</strong>. It is the clues, not the algorithm.")}
    {card("Every decade", "From the 1960s to the 2010s: <strong>+3.7 to +6.6</strong> every time.")}
    {card("The future", "Trained on old decades, tested on a decade it never saw: still <strong>+1.9 to +3.4</strong>.")}
    {card("The only other study", "We redid its exact method (Du, 2022) and added our clue: <strong>better in 50 out of 50</strong> tries.")}
  </div>""", "It was not luck. But then we read how the dataset had been built..."),

    slide(C4, "Step 4 · A surprise", "A trap: how the non-hits were chosen",
          """<div class="body cols-2e" style="margin-top:34px;align-items:start">
    <div class="stack">
      <p class="say">The dataset's own rules: a non-hit must come from an artist who <strong>had no hit</strong> that decade, in a <strong>niche genre</strong> (jazz, Latin, classical...).</p>
      <p class="say-s">So “the artist had hits before” is partly the dataset's rule, not the music.</p>
    </div>
    <div class="stack">
      <p class="big k-bad">97<span class="unit">%</span></p>
      <p class="say-s">of songs by an artist with an earlier hit are hits <strong>in this dataset</strong>. Our +5 is too optimistic here.</p>
    </div>
  </div>""", "So we needed a dataset built differently."),

    slide(C4, "Step 4 · Another dataset", "We tested on another dataset, built without that rule",
          """<div class="body cols-2e" style="margin-top:34px;align-items:start">
    <div class="stack">
      <p class="say-s">ISMIR 2019 dataset (published, 11,662 songs): non-hits are <strong>random songs</strong>.</p>
      <p class="big k-both">76.3<span class="unit">%</span></p>
      <p class="say-s">with our clue — <strong>better than the best published result (75.0 %)</strong>.</p>
    </div>
    <div class="stack">
      <p class="say-h k-bad">But another trap</p>
      <p class="say">The <strong>year alone</strong> already gets <strong>70 %</strong>: the hits in this dataset are older songs. The computer can win just by looking at the date.</p>
    </div>
  </div>""", "So we made the test fair: every hit against a non-hit from the same year."),

    slide(C4, "Step 4 · The fair test", "Same year for hits and non-hits: the date is useless, the artist still helps",
          f"""<div class="body cols-2e" style="margin-top:34px;align-items:start">
    <div class="stack">
      <p class="say-h k-era">The year</p>
      <p class="mid">useless</p>
      <p class="say-s">It was only a trick of the dataset.</p>
    </div>
    <div class="stack">
      <p class="say-h k-artist">The artist's history</p>
      <p class="mid k-artist">+2.6<span class="unit">points</span></p>
      <p class="say-s">Better in 10 out of 10 tests. <strong>This is our honest number</strong>: smaller than +5, but real.</p>
    </div>
  </div>""", "And for artists the computer has never seen?"),

    slide(C4, "Step 4 · New artists", "Artists never seen, and brand-new artists",
          f"""<div class="body cols-2e" style="margin-top:34px;align-items:start">
    {card("Artists never seen in training", "We made sure no test artist appeared in training. The gain <strong>stays the same</strong> (+5.0): the computer does not just memorise names.")}
    {card("A brand-new artist (first song ever)", "No history, so “who” <strong>cannot help</strong> (+0.4, nothing). “When” helps a little (+2.2). This is the hardest case — and the main thing left to solve.")}
  </div>""", "Next, we tried to give the computer more musical information."),

    slide(C5, "Step 5 · Adding more", "We tried more clues — and checked every gain",
          f"""<div class="body cols-3" style="margin-top:34px;align-items:start">
    {card("Song titles: +6.7", f"Too good to be true. The titles reveal <strong>how the dataset was collected</strong>, not what makes a hit. {NO}", "k-bad")}
    {card("Detailed sound (185 measures)", f"+3.8 on one test... but only <strong>+0.9 over 25 tests</strong>, which is nothing. The first result was luck. {NO}", "k-bad")}
    {card("Small details", f"Length of the title, where the chorus starts, number of sections: <strong>nothing</strong>. {NO}", "k-bad")}
  </div>""", "Then one clue looked really good: splitting the sound into frequencies (FFT)."),

    slide(C5, "Step 5 · The FFT", "The FFT looked like a real gain — it was recognising the genre",
          f"""<div class="body cols-3" style="margin-top:34px;align-items:start">
    {card("It looked good", "The FFT splits the sound into frequencies (bass, middle, treble). <strong>+1.4 points</strong> on 35,000 songs, very significant.")}
    {card("Why it was a trap", "Non-hits come from niche genres (rule of the dataset), and the sound <strong>tells the genre</strong>. The computer was recognising jazz or Latin, not hits.")}
    {card("Fair test", f"Same number of hits and non-hits in every genre: <strong>+0.1, nothing</strong>. {NO}")}
  </div>""", "Lesson: a big gain can be a trap. Then our teacher asked us to study time series."),

    slide(C6, "Step 6 · Time series", "We learned to read the chart as a signal over time",
          f"""<div class="body cols-3" style="margin-top:30px;align-items:start">
    {card("3 parts of a time series", "<strong>Trend</strong>: slow change over years. <strong>Seasonality</strong>: repeats every year. <strong>Residual</strong>: what is left.")}
    {card("What we found", "New songs per week: mostly a <strong>trend</strong> (fewer until 2001, more again with streaming). Christmas songs: pure <strong>seasonality</strong> — the FFT finds the 52-week cycle by itself.")}
    {card("Forecasting next year", f"<strong>ARIMA</strong> and Holt-Winters are the best {OK}. Machine learning is not better here: 800 data points are too few.")}
  </div>""", "Then the big question: can time series help predict a hit?"),

    slide(C6, "Step 6 · Time series and hits", "Can time series predict a hit? It depends on <em>when</em> you ask",
          f"""<div class="body cols-2e" style="margin-top:30px;align-items:start">
    <div class="stack rule" style="gap:10px">
      <p class="say-h">BEFORE the song comes out <span class="tiny">(our question)</span></p>
      <p class="say-s">Does the song sound like the recent hits? <strong>+0.1, nothing</strong>: the trend only knows the decade, which we already had. {NO}</p>
      <p class="mid k-bad">No</p>
    </div>
    <div class="stack rule" style="gap:10px">
      <p class="say-h">AFTER it comes out <span class="tiny">(another question)</span></p>
      <p class="say-s">The song just entered the chart. Its <strong>first 3 weeks</strong> tell if it will reach the <strong>top 10</strong>: right 89 times out of 100 when comparing two songs (AUC .89). {OK}</p>
      <p class="mid k-both">Yes</p>
    </div>
  </div>
  <p class="say-s" style="margin-top:10px">Both are true. Time series do not help guess a hit <em>before</em> it is out — but once it is out, they are the best clue we found.</p>""",
          "So, what did we prove in the end?"),
]

ENDING = [
    slide(CE, "Summary", "What we proved",
          f"""<div class="body cols-2e" style="margin-top:24px;align-items:start;gap:36px">
    <div class="stack" style="gap:12px">
      <p class="src k-both" style="letter-spacing:.16em">TRUE — IT SURVIVED EVERY TEST</p>
      <p class="say-s">1 · <strong>Who sings it matters</strong>: +5 points on the big dataset, <strong>+2.6 on a fair test</strong>.</p>
      <p class="say-s">2 · We <strong>beat the published results</strong>: 76.3 % vs 75.0 %, and the other study 50 times out of 50.</p>
      <p class="say-s">3 · It works with <strong>any algorithm</strong> and for <strong>artists never seen</strong>.</p>
      <p class="say-s">4 · For a <strong>brand-new artist</strong>, only the era helps a little.</p>
      <p class="say-s">5 · <strong>After</strong> a song comes out, its first 3 weeks predict a top-10 hit (AUC .89). <strong>Before</strong>, time series add nothing.</p>
    </div>
    <div class="stack" style="gap:12px">
      <p class="src k-bad" style="letter-spacing:.16em">FALSE — IT WAS A TRAP</p>
      <p class="say-s"><strong>Song titles</strong>: reveal how the data was collected.</p>
      <p class="say-s"><strong>Detailed sound</strong>: the gain was luck.</p>
      <p class="say-s"><strong>FFT</strong>: it was recognising the genre.</p>
      <p class="say-s"><strong>The year</strong>: hits were just older songs.</p>
      <p class="say-s"><strong>Music trends</strong>: they only know the decade.</p>
      <p class="say-s rule"><strong>In the sound itself, nothing survived every test.</strong></p>
    </div>
  </div>
  <p class="say" style="margin-top:14px">A hit is mostly <span class="k-artist">who sings it</span> and <span class="k-era">when</span> — and, once it is out, <span class="k-both">how its first weeks go</span>.</p>"""),

    slide(CE, "What is left", "What we cannot do yet",
          f"""<div class="body cols-3" style="margin-top:36px;align-items:start">
    {card("Brand-new artists", "With no history, we have almost nothing. Next: the record label, the collaborators, the first streams.")}
    {card("Lyrics and deep learning", "Lyrics helped a little on one dataset (+0.6). Neural networks on the raw sound are still to try — with the same careful tests.")}
    {card("A dataset without traps", "Every dataset we used had a trap (artist, genre or year). A fair one would make the question much cleaner.")}
  </div>"""),

    slide(CE, "Thank you", "Thank you — questions?",
          """<div class="body" style="grid-template-columns:1fr;margin-top:34px">
    <p class="say">The slides after this one are the <strong>details</strong>: every algorithm and why, every clue we kept or removed and why, every table of results.</p>
  </div>"""),
]

APPENDIX_TITLE = """<section class="slide title-slide" data-foot="Appendix">
  <div>
    <p class="eyebrow">Appendix</p>
    <h1 style="font-size:88px">The details</h1>
    <p class="lede" style="font-size:24px;max-width:46ch;margin-top:22px">Algorithms, feature by feature results, statistics, and every table — for questions.</p>
  </div>
</section>"""

CSS = """  .say { font-size: 28px; line-height: 1.38; color: var(--ink); max-width: 48ch; }
  .say-h { font: 700 27px/1.2 var(--display); color: var(--ink); }
  .say-s { font-size: 22px; line-height: 1.45; color: var(--ink2); }
  .next { margin-top: auto; padding-top: 14px; border-top: 1px solid var(--line); font: 600 20px/1.3 var(--body); color: var(--era-ink); }
"""

CHARTS = """
  dotRows("s-main", { min: 76, max: 85, step: 1, tick: tickPct, fmt: pct, rows: [
    { label: "sound only", value: 78.2, color: B },
    { label: "+ when", value: 80.2, color: E },
    { label: "+ who", value: 81.0, color: A },
    { label: "+ who + when", value: 83.2, color: BO, bold: true } ], rowH: 64 });
  columns("s-curve", { ref: { value: 49.4, label: "all songs 49%" }, items: [
    { label: "0", value: 39.3, n: "33,438", color: B },
    { label: "1–2", value: 91.0, n: "1,838", color: A },
    { label: "3–5", value: 97.2, n: "1,752", color: A },
    { label: "6–10", value: 99.6, n: "1,495", color: A },
    { label: "11+", value: 98.9, n: "2,037", color: A } ] });
"""

SCORE = slide(CE, "Scoreboard", "How much our model improved each dataset",
              """<div class="body" style="grid-template-columns:1fr;margin-top:20px">
    <table class="t dense">
      <thead><tr><th style="width:30%">Dataset</th><th style="width:9%">Measure</th><th class="n" style="width:14%">Without our clues</th><th class="n" style="width:13%">With our model</th><th class="n" style="width:9%">Gain</th><th style="width:25%">Compared with published work</th></tr></thead>
      <tbody>
        <tr><td><strong>Spotify Hit Predictor</strong> · 40,560 songs</td><td>accuracy</td><td class="n">78.2 %</td><td class="n k-both"><strong>83.2 %</strong></td><td class="n">+5.0</td><td>F1 of hits .787 → .831</td></tr>
        <tr><td>Same data, Du (2022) method · 2000s</td><td>accuracy</td><td class="n">84.6 %</td><td class="n k-both"><strong>86.7 %</strong></td><td class="n">+2.1</td><td>Du published 84 % · we win 50/50 splits</td></tr>
        <tr><td>Same data, Du (2022) method · 2010s</td><td>accuracy</td><td class="n">82.9 %</td><td class="n k-both"><strong>86.1 %</strong></td><td class="n">+3.2</td><td>Du published 82 % · we win 50/50 splits</td></tr>
        <tr><td><strong>ISMIR 2019</strong> · as in the original study</td><td>accuracy</td><td class="n">73.9 %</td><td class="n k-both"><strong>76.3 %</strong></td><td class="n">+2.4</td><td>best published <strong>75.04 %</strong> → beaten</td></tr>
        <tr><td><strong>ISMIR 2019</strong> · fair test (same years)</td><td>accuracy</td><td class="n">68.2 %</td><td class="n k-artist"><strong>70.8 %</strong></td><td class="n">+2.6</td><td>our honest number</td></tr>
        <tr><td><strong>SpotGenTrack</strong> · 101,909 tracks (19 % hits)</td><td>AUC</td><td class="n">.690</td><td class="n k-both"><strong>.777</strong></td><td class="n">+.087</td><td>hits found 8 % → 25 %</td></tr>
        <tr><td><strong>Hot 100, after release</strong> · 31,925 songs</td><td>AUC (top 10)</td><td class="n">.59</td><td class="n k-both"><strong>.89</strong></td><td class="n">+.30</td><td>with the first 3 weeks on the chart</td></tr>
      </tbody>
    </table>
  </div>
  <p class="tiny" style="margin-top:8px">“Without our clues”: the same model on the dataset's usual inputs (13 Spotify descriptors; Du's own inputs; 592 Essentia descriptors + year on ISMIR; year, month and artist history for the Hot 100). Cross-validated means.</p>""")

MODEL = slide(CE, "Our final model", "Our final model: one Random Forest, 20 numbers per song",
              f"""<div class="body cols-2" style="margin-top:22px;align-items:start;gap:34px">
    <div class="stack" style="gap:14px">
      <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:center">
        <div class="panel stack" style="gap:6px;padding:16px"><p class="mid" style="font-size:48px">13</p><p class="say-s">Spotify sound descriptors</p></div>
        <p class="mid" style="font-size:40px;color:var(--muted)">+</p>
        <div class="panel stack" style="gap:6px;padding:16px"><p class="mid k-era" style="font-size:48px">6</p><p class="say-s">decade boxes (when)</p></div>
        <p class="mid" style="font-size:40px;color:var(--muted)">+</p>
        <div class="panel stack" style="gap:6px;padding:16px"><p class="mid k-artist" style="font-size:48px">1</p><p class="say-s">artist history (who)</p></div>
      </div>
      <p class="say-s">Artist history = log(1 + the artist's Hot 100 songs in <strong>earlier decades only</strong>).</p>
      <p class="say" style="font-size:24px">→ <strong>Random Forest</strong>: 80 trees, 8 features per question, <strong>never tuned</strong> → hit / not hit</p>
      <p class="say-s" style="font-size:19px">Same setting as the reference paper, so results compare. XGBoost is 1 point better on average, not significantly (p = .87).</p>
    </div>
    <div class="stack" style="gap:14px">
      <div class="stack rule" style="gap:8px">
        <p class="src" style="letter-spacing:.16em">OTHER DATASETS, SAME RECIPE</p>
        <p class="say-s" style="font-size:19px">Their own sound descriptors (592 Essentia on ISMIR) + year or decade + artist history; Random Forest, 200 trees.</p>
      </div>
      <div class="stack rule" style="gap:8px">
        <p class="src k-bad" style="letter-spacing:.16em">NOT IN THE MODEL</p>
        <p class="say-s" style="font-size:19px">Titles, detailed sound, FFT, genre, market trend, chorus_hit / sections: traps or noise. Lyrics: promising, one dataset only.</p>
      </div>
      <div class="stack rule" style="gap:8px">
        <p class="src k-both" style="letter-spacing:.16em">A SEPARATE MODEL, AFTER RELEASE</p>
        <p class="say-s" style="font-size:19px">Songs already on the chart: first 3 weeks (positions, best rank, slope, acceleration) + year, month, artist history → Random Forest → top 10 or not. AUC .89.</p>
      </div>
    </div>
  </div>""")
