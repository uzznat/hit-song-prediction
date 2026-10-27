"""
phase5_richaudio.py  —  richer audio features (no neural net, runs on your setup).
=================================================================================
The FFT step used 31 audio features. Here we extract ~185 finer ones (20 MFCCs +
their 1st/2nd derivatives, chroma-CQT, tonnetz, spectral contrast, ZCR, RMS
dynamics, rhythm), reduce them with PCA (fit on train only), and ask the harness:

    does a richer audio representation beat the 31-feature FFT, and does it add
    anything on top of the artist feature?

Reuses the audio clips you already downloaded (cache/audio + cache/deezer_manifest.json)
and only needs librosa — no TensorFlow/PyTorch, so it runs on Python 3.14.

RUN (next to phase0_harness.py + feature_experiments.py):
    pip install numpy pandas scikit-learn scipy matplotlib librosa soundfile
    py phase5_richaudio.py
Outputs: phase5_results.png , cache/rich_features.csv (resumable)
"""
import os, glob, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from feature_experiments import load_spotify_with_features, AUDIO, DEC_ORDER
from phase0_harness import make_baseline_rf, mcnemar_test

SEED=42
CACHE="cache"; MANIFEST=os.path.join(CACHE,"deezer_manifest.json")
RICH_CSV=os.path.join(CACHE,"rich_features.csv")
N_PCA=40
GRAY,GREEN,BLUE,AMBER="#B4B2A9","#1DB954","#2a78d6","#C97A10"


def extract_rich(y, sr=22050):
    import librosa
    f,n=[],[]
    def add(nm,v): n.append(nm); f.append(float(v))
    def stats(pref,M):
        M=np.atleast_2d(M)
        for i in range(M.shape[0]):
            add(f"{pref}{i}_m",np.mean(M[i])); add(f"{pref}{i}_s",np.std(M[i]))
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
    stats("mfcc",mfcc); stats("d1",librosa.feature.delta(mfcc)); stats("d2",librosa.feature.delta(mfcc,order=2))
    stats("chr",librosa.feature.chroma_cqt(y=y,sr=sr))
    stats("ton",librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr))
    stats("con",librosa.feature.spectral_contrast(y=y,sr=sr))
    for nm,fn in [("cent",librosa.feature.spectral_centroid),("bw",librosa.feature.spectral_bandwidth),
                  ("roll",librosa.feature.spectral_rolloff)]:
        stats(nm,fn(y=y,sr=sr))
    stats("flat",librosa.feature.spectral_flatness(y=y))
    stats("zcr",librosa.feature.zero_crossing_rate(y)); stats("rms",librosa.feature.rms(y=y))
    oe=librosa.onset.onset_strength(y=y,sr=sr); add("onset_m",np.mean(oe)); add("onset_s",np.std(oe))
    ac=np.mean(librosa.feature.tempogram(onset_envelope=oe,sr=sr),axis=1)
    add("pulse",np.max(ac)/(np.mean(ac)+1e-9))
    return f,n


def build_rich(manifest):
    import librosa
    if os.path.exists(RICH_CSV):
        cached=pd.read_csv(RICH_CSV); done=set(cached["uri"]); print(f"rich cached: {len(cached)}")
    else: cached,done=pd.DataFrame(),set()
    rows,names=[],None
    todo=[(u,p) for u,p in manifest.items() if p and u not in done and os.path.exists(p)]
    print(f"extracting rich features for {len(todo)} clips (this is the slow step)...")
    def flush():
        nonlocal cached,rows
        if not rows or names is None: return
        new=pd.DataFrame(rows,columns=["uri"]+names)
        cached=pd.concat([cached,new],ignore_index=True) if len(cached) else new
        cached.to_csv(RICH_CSV,index=False); rows=[]
    for i,(u,p) in enumerate(todo):
        try:
            y,sr=librosa.load(p,sr=22050,mono=True,duration=30)
            if len(y)<sr: continue
            v,names=extract_rich(y,sr); rows.append([u]+v)
        except Exception: continue
        if (i+1)%100==0: flush(); print(f"  {i+1}/{len(todo)} (saved)")
    flush(); print(f"rich features ready: {len(cached)}")
    return cached


def rf_pca_predict(A_tr,A_te,R_tr,R_te,ytr,use_rich):
    """RF on audio (+ optional PCA-reduced rich features). PCA fit on train only."""
    if use_rich:
        sc=StandardScaler().fit(R_tr)
        pca=PCA(n_components=min(N_PCA,R_tr.shape[1]),random_state=SEED).fit(sc.transform(R_tr))
        Xtr=np.column_stack([A_tr,pca.transform(sc.transform(R_tr))])
        Xte=np.column_stack([A_te,pca.transform(sc.transform(R_te))])
    else:
        Xtr,Xte=A_tr,A_te
    m=make_baseline_rf().fit(Xtr,ytr); return m.predict(Xte)


def main():
    if not os.path.exists(MANIFEST):
        print("No cache/deezer_manifest.json — run phase2_fft.py first to download clips."); return
    manifest=json.load(open(MANIFEST))
    rich=build_rich(manifest)
    if len(rich)==0: print("No features."); return
    rcols=[c for c in rich.columns if c!="uri"]

    df=load_spotify_with_features()
    data=df.merge(rich,on="uri",how="inner").dropna(subset=rcols).reset_index(drop=True)
    print(f"\nusable tracks: {len(data)}")
    y=data["target"].values; idx=np.arange(len(data))
    tr,te=train_test_split(idx,test_size=0.25,stratify=y,random_state=SEED)
    A=data[AUDIO].values; R=data[rcols].values
    art=data["artist_prior_hits_log"].values.reshape(-1,1)

    p_audio=rf_pca_predict(A[tr],A[te],R[tr],R[te],y[tr],use_rich=False)
    p_rich =rf_pca_predict(A[tr],A[te],R[tr],R[te],y[tr],use_rich=True)
    Aa=np.column_stack([A,art])
    p_art =rf_pca_predict(Aa[tr],Aa[te],R[tr],R[te],y[tr],use_rich=False)
    p_artrich=rf_pca_predict(Aa[tr],Aa[te],R[tr],R[te],y[tr],use_rich=True)

    acc=lambda p:accuracy_score(y[te],p)
    print("\n=== results (same audio subset) ===")
    print(f"  audio only            {acc(p_audio):.4f}")
    print(f"  audio + rich          {acc(p_rich):.4f}   McNemar vs audio p={mcnemar_test(y[te],p_audio,p_rich)['pvalue']:.3g}")
    print(f"  audio + artist        {acc(p_art):.4f}")
    print(f"  audio + artist + rich {acc(p_artrich):.4f}   McNemar vs a+artist p={mcnemar_test(y[te],p_art,p_artrich)['pvalue']:.3g}")

    labels=["audio","audio\n+ rich","audio\n+ artist","audio + artist\n+ rich"]
    vals=[acc(p_audio),acc(p_rich),acc(p_art),acc(p_artrich)]
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.bar(range(4),vals,color=[GRAY,GREEN,BLUE,AMBER],width=0.62)
    for i,v in enumerate(vals): ax.text(i,v+0.003,f"{v:.3f}",ha="center",weight="bold",fontsize=12)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=10); ax.set_ylim(0.72,0.86)
    ax.set_ylabel("test accuracy"); ax.set_title(f"Richer audio ({len(rcols)} features -> PCA {N_PCA})",fontsize=12,weight="bold")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("phase5_results.png",dpi=140,bbox_inches="tight"); print("\nsaved phase5_results.png")


if __name__=="__main__":
    main()
