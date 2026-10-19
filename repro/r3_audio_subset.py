"""
R3 -- the audio subset (Kaggle songs with a 30 s Deezer clip).

  * combinations of blocks: audio(13) / rich (185 librosa -> PCA 40) /
    decade / artist / decade x audio, and FFT (31) vs rich
  * PCA sweep on audio + artist + rich
  * block knock-out ablation from the full model
  * rich-family knock-out (MFCC, deltas, chroma, tonnetz, spectral shape,
    ZCR+RMS, rhythm)
  canonical split (75/25, seed 42, as in phase5) + 5x5 repeated CV.

    py repro/r3_audio_subset.py
"""
import numpy as np
import pandas as pd

from common import (AUDIO, ART, DEC_COLS, DEC_ORDER, SEED, canonical_eval, cv_eval,
                    load_audio_subset, load_kaggle, make_pipeline, save, show)
from sklearn.model_selection import train_test_split

FAMILIES = {
    "MFCC":           ("mfcc",),
    "MFCC deltas":    ("d1", "d2"),
    "chroma":         ("chr",),
    "tonnetz":        ("ton",),
    "spectral shape": ("cent", "bw", "roll", "flat", "con"),
    "ZCR + RMS":      ("zcr", "rms"),
    "rhythm":         ("onset", "pulse"),
}


def family_of(col):
    stem = col.split("_")[0].rstrip("0123456789")
    for fam, prefixes in FAMILIES.items():
        if stem in prefixes:
            return fam
    return "other"


def main():
    df = load_kaggle()
    data, rcols, fcols = load_audio_subset(df)
    data = data.dropna(subset=fcols).reset_index(drop=True)
    y = data["target"].values
    # decade x audio: each audio feature multiplied by each decade indicator
    dxa = []
    for d, c in zip(DEC_ORDER, DEC_COLS):
        for a in AUDIO:
            data[f"{a}_x_{d}"] = data[a] * data[c]
            dxa.append(f"{a}_x_{d}")
    fams = pd.Series({c: family_of(c) for c in rcols})
    print(f"audio subset: {len(data)} songs ({y.sum()} hits), rich={len(rcols)}, fft={len(fcols)}")
    print("rich families:", fams.value_counts().to_dict())
    save(pd.DataFrame([{"n_songs": len(data), "n_hits": int(y.sum()), "n_rich": len(rcols),
                        "n_fft": len(fcols), **{f"n_{d}": int((data.decade == d).sum())
                                                 for d in DEC_ORDER}}]),
         "r3_subset_facts.csv")

    P = make_pipeline
    combos = {
        "audio (13)":                     P(AUDIO),
        "+ decade":                       P(AUDIO + DEC_COLS),
        "+ artist":                       P(AUDIO + [ART]),
        "+ rich":                         P(AUDIO, rcols),
        "+ FFT (31)":                     P(AUDIO + fcols),
        "+ decade x audio":               P(AUDIO + dxa),
        "+ decade + artist":              P(AUDIO + DEC_COLS + [ART]),
        "+ rich + decade":                P(AUDIO + DEC_COLS, rcols),
        "+ artist + rich":                P(AUDIO + [ART], rcols),
        "+ artist + FFT":                 P(AUDIO + [ART] + fcols),
        "+ artist + rich + FFT":          P(AUDIO + [ART] + fcols, rcols),
        "+ artist + rich + decade":       P(AUDIO + [ART] + DEC_COLS, rcols),
        "all 5 blocks":                   P(AUDIO + [ART] + DEC_COLS + dxa, rcols),
    }
    # knock-out from the full model (audio + artist + rich + decade + decade x audio)
    full = AUDIO + [ART] + DEC_COLS + dxa
    ablation = {
        "full model":            P(full, rcols),
        "- audio":               P([ART] + DEC_COLS + dxa, rcols),
        "- artist":              P(AUDIO + DEC_COLS + dxa, rcols),
        "- rich":                P(full),
        "- decade":              P(AUDIO + [ART] + dxa, rcols),
        "- decade x audio":      P(AUDIO + [ART] + DEC_COLS, rcols),
        "- decade, dxa, rich":   P(AUDIO + [ART]),
    }
    pca = {f"PCA {k}": P(AUDIO + [ART], rcols, n_pca=k) for k in (5, 10, 20, 40, 60, 100)}
    pca["no PCA (185 raw)"] = P(AUDIO + [ART] + rcols)
    pca = {"PCA 40": pca.pop("PCA 40"), **pca}
    fam = {"audio + artist + rich (all families)": P(AUDIO + [ART], rcols)}
    for f in FAMILIES:
        keep = [c for c in rcols if fams[c] != f]
        fam[f"- {f}"] = P(AUDIO + [ART], keep, n_pca=min(40, len(keep)))

    idx = np.arange(len(data))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)

    for tag, cfg, ref in [("combos", combos, "audio (13)"),
                          ("ablation", ablation, "full model"),
                          ("pca", pca, "PCA 40"),
                          ("families", fam, "audio + artist + rich (all families)")]:
        can = canonical_eval(cfg, data, y, tr, te, ref=ref)
        print(f"\n=== R3 {tag}: canonical 75/25 split ({len(tr)} train / {len(te)} test) ===")
        show(can)
        save(can, f"r3_{tag}_canonical.csv")
        summ, folds = cv_eval(cfg, data, y, ref=ref, label=f"R3 {tag}")
        print(f"\n=== R3 {tag}: 5x5 repeated CV ===")
        show(summ)
        save(summ, f"r3_{tag}_cv_summary.csv")
        save(folds, f"r3_{tag}_cv_folds.csv")


if __name__ == "__main__":
    main()
