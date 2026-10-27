"""
R7 -- FFT descriptors on (almost) the whole Kaggle benchmark.

R3 measured FFT on a 3,562-song sample only. fetch_audio_full.py extended the
Deezer previews to every song it could match (cache/fft_features_full.csv).
Here: does FFT add anything on top of the main model (audio + decade + artist)?

  * coverage facts: share of hits / flops / decades with a clip (Deezer finds
    hits more often, so every comparison below is made on the SAME songs)
  * 5x5 repeated CV, reference = main model, per-group accuracy for songs by
    new artists (no earlier hit) vs known artists

    py repro/fetch_audio_full.py download && py repro/fetch_audio_full.py extract
    py repro/r7_fft_full.py
"""
import os

import pandas as pd

from common import AUDIO, ART, DEC_COLS, ROOT, cv_eval, load_kaggle, make_pipeline, save, show


def main():
    df = load_kaggle()
    fft = pd.read_csv(os.path.join(ROOT, "cache", "fft_features_full.csv"))
    fcols = [c for c in fft.columns if c != "uri"]
    fft = fft.rename(columns={c: "fft_" + c for c in fcols})
    fcols = ["fft_" + c for c in fcols]
    df["has_fft"] = df.uri.isin(fft.uri)
    data = df.merge(fft, on="uri", how="inner").dropna(subset=fcols).reset_index(drop=True)
    y = data["target"].values

    facts = {"n_benchmark": len(df), "n_with_fft": len(data),
             "coverage": round(df.has_fft.mean(), 4),
             "coverage_hits": round(df.loc[df.target == 1, "has_fft"].mean(), 4),
             "coverage_flops": round(df.loc[df.target == 0, "has_fft"].mean(), 4),
             "hit_share_subset": round(y.mean(), 4),
             "n_new_artist": int((data.artist_prior_hits == 0).sum())}
    for d, g in df.groupby("decade"):
        facts[f"coverage_{d}"] = round(g.has_fft.mean(), 4)
    print(facts)
    save(pd.DataFrame([facts]).T.reset_index().rename(columns={"index": "fact", 0: "value"}),
         "r7_facts.csv")

    P = make_pipeline
    configs = {
        "+ decade + artist":         P(AUDIO + DEC_COLS + [ART]),        # main model
        "audio (13)":                P(AUDIO),
        "+ FFT":                     P(AUDIO + fcols),
        "+ FFT + decade":            P(AUDIO + fcols + DEC_COLS),
        "+ artist + FFT":            P(AUDIO + [ART] + fcols),
        "+ decade + artist + FFT":   P(AUDIO + DEC_COLS + [ART] + fcols),
    }
    groups = (data.artist_prior_hits > 0).map({True: "known", False: "new"}).values
    summ, folds = cv_eval(configs, data, y, ref="+ decade + artist", label="R7",
                          groups=groups)
    print("\n=== R7: 5x5 repeated CV on songs with a Deezer clip ===")
    show(summ)
    save(summ, "r7_cv_summary.csv")
    save(folds, "r7_cv_folds.csv")


if __name__ == "__main__":
    main()
