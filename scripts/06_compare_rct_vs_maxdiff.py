
#!/usr/bin/env python3
import os, argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from utils import ensure_dir, bh_fdr
import math
from scipy.stats import norm

def main(out_dir, figures_dir):
    ensure_dir(out_dir); ensure_dir(figures_dir)
    effects = pd.read_csv(os.path.join(out_dir, "per_video_effects.csv"))
    md = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "maxdiff_dummy_data.csv"))

    df = effects.merge(md[["video_id","maxdiff_mean","sample_size"]], on="video_id", how="left")

    # Correlations
    pear = df[["shrunk_effect","maxdiff_mean"]].corr().iloc[0,1]
    spear = df[["shrunk_effect","maxdiff_mean"]].corr(method="spearman").iloc[0,1]

    # z-test on naive effects (or use shrunk_* if you prefer)
    z = df["naive_effect"] / df["naive_se"].replace(0, np.nan)

    # Two-sided p-values, vectorized
    pvals = 2 * norm.sf(np.abs(z))
    pvals = pd.Series(pvals).fillna(1.0)   # guard any NaNs (e.g., missing/zero SE)

    # FDR adjust
    reject, p_adj = bh_fdr(pvals.values, alpha=0.05)

    df["pval"] = pvals.values
    df["pval_fdr"] = p_adj
    df["sig_fdr_05"] = reject

    df.to_csv(os.path.join(out_dir, "te_vs_maxdiff_summary.csv"), index=False)

    # Scatter
    plt.figure(figsize=(7,5))
    plt.scatter(df["maxdiff_mean"], df["shrunk_effect"], s=np.clip(df["sample_size"], 10, 200), alpha=0.7, edgecolor="k", linewidth=0.3)
    plt.xlabel("MaxDiff mean (proportion selected)")
    plt.ylabel("Shrunken RCT effect (Δ approval vs pooled control)")
    plt.title(f"Alignment: Pearson={pear:.2f}, Spearman={spear:.2f}")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "scatter_te_vs_maxdiff.png"), dpi=160)

    # Funnel plot: |effect| vs SE
    plt.figure(figsize=(6,5))
    plt.scatter(df["naive_se"], np.abs(df["naive_effect"]), alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.xlabel("SE (naive effect)")
    plt.ylabel("|Naive effect|")
    plt.title("Funnel plot (smaller SE → more stable effects)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "funnel_effects.png"), dpi=160)

    print("Comparison complete. Figures saved to", figures_dir)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    ap.add_argument("--figures-dir", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    args = ap.parse_args()
    main(os.path.abspath(args.out_dir), os.path.abspath(args.figures_dir))
