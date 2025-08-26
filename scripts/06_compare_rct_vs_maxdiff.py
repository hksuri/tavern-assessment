#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_compare_rct_vs_maxdiff.py
Compare per-video RCT treatment effects against MaxDiff persuasion scores,
compute correlations, multiple-testing adjusted p-values, and generate figures.

Core outputs (always attempted):
  - output/te_vs_maxdiff_summary.csv
  - figures/alignment.png     (if enough overlap)
  - figures/funnel.png        (always, based on naive effects)

Optional extras (enable with --extra-plots):
  - figures/hist_naive_vs_shrunk.png
  - figures/hist_maxdiff.png
  - figures/residuals_maxdiff_minus_te.png
  - figures/rank_overlap_top10.png
  - figures/top_bottom_tables.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Utilities from our helpers
from utils import ensure_dir, bh_fdr


# --------------------------- Helpers -----------------------------------------

def norm_id_series(s: pd.Series) -> pd.Series:
    """
    Normalize IDs so '21.0' and '21' match; also trims whitespace and lowercases.
    """
    s = s.astype("string").str.strip()
    num = pd.to_numeric(s, errors="coerce")
    out = s.copy()
    m = num.notna()
    out.loc[m] = num.loc[m].astype("int64").astype(str)
    return out.str.lower()


def safe_corr(df: pd.DataFrame, a: str, b: str):
    """
    Compute Pearson and Spearman correlations on non-null overlap.
    Returns (pearson, spearman, n), where values may be np.nan if not enough rows.
    """
    sub = df[[a, b]].dropna()
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    return float(sub[a].corr(sub[b])), float(sub[a].corr(sub[b], method="spearman")), len(sub)


def save_alignment_scatter(df: pd.DataFrame, out_path: str):
    pear, spear, n = safe_corr(df, "shrunk_effect", "maxdiff_mean")
    if n < 3:
        print(f"[alignment] Not enough non-null overlap (n={n}); skipping {out_path}")
        return
    plt.figure(figsize=(8.5, 6))
    sizes = df.loc[df[["maxdiff_mean", "shrunk_effect"]].dropna().index, "sample_size"] \
                .fillna(60).clip(lower=20, upper=240)
    sub = df[["maxdiff_mean", "shrunk_effect"]].dropna()
    plt.scatter(sub["maxdiff_mean"], sub["shrunk_effect"], s=sizes.loc[sub.index], alpha=0.75,
                edgecolor="k", linewidth=0.3)
    plt.xlabel("MaxDiff mean (proportion selected)")
    plt.ylabel("Shrunken RCT effect (Δ approval vs pooled control)")
    plt.title(f"Alignment: Pearson={pear:.2f}, Spearman={spear:.2f} (n={n})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[alignment] Saved {out_path} (n={n}, r={pear:.2f}, ρ={spear:.2f})")


def save_funnel(df: pd.DataFrame, out_path: str):
    # Require naive_effect & naive_se
    sub = df[["naive_effect", "naive_se"]].dropna()
    if len(sub) == 0:
        print(f"[funnel] No data; skipping {out_path}")
        return
    plt.figure(figsize=(8.5, 6))
    plt.scatter(sub["naive_se"], np.abs(sub["naive_effect"]), alpha=0.65,
                edgecolor="k", linewidth=0.3)
    plt.xlabel("SE (naive effect)")
    plt.ylabel("|Naive effect|")
    plt.title("Funnel plot (smaller SE → more stable effects)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[funnel] Saved {out_path} (n={len(sub)})")


def save_hist_naive_vs_shrunk(df: pd.DataFrame, out_path: str):
    sub = df[["naive_effect", "shrunk_effect"]].dropna()
    if len(sub) == 0:
        print(f"[hist_naive_vs_shrunk] No data; skipping {out_path}")
        return
    # Use a shared binning to make comparison meaningful
    lo = float(min(sub["naive_effect"].min(), sub["shrunk_effect"].min()))
    hi = float(max(sub["naive_effect"].max(), sub["shrunk_effect"].max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        print(f"[hist_naive_vs_shrunk] Degenerate range; skipping {out_path}")
        return
    bins = np.linspace(lo, hi, 25)

    plt.figure(figsize=(8.5, 6))
    plt.hist(sub["naive_effect"], bins=bins, alpha=0.55, density=True, label="Naive")
    plt.hist(sub["shrunk_effect"], bins=bins, alpha=0.55, density=True, label="Shrunken")
    plt.axvline(0, linestyle="--", linewidth=1, color="k")
    plt.legend()
    plt.xlabel("Effect (Δ approval)")
    plt.ylabel("Density")
    plt.title("Distribution of Effects: Naive vs Shrunken")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[hist_naive_vs_shrunk] Saved {out_path}")


def save_hist_maxdiff(df: pd.DataFrame, out_path: str):
    x = df["maxdiff_mean"].dropna()
    if len(x) == 0:
        print(f"[hist_maxdiff] No maxdiff_mean; skipping {out_path}")
        return
    # Align weights to x's index
    w = df.loc[x.index, "sample_size"].fillna(1).clip(lower=1)
    plt.figure(figsize=(8.5, 6))
    plt.hist(x, bins=20, weights=w, alpha=0.8)
    plt.xlabel("MaxDiff mean (proportion selected)")
    plt.ylabel("Weighted count")
    plt.title("Distribution of MaxDiff Scores (weighted by sample size)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[hist_maxdiff] Saved {out_path} (n={len(x)})")


def save_residuals(df: pd.DataFrame, out_path: str):
    sub = df[["maxdiff_mean", "shrunk_effect"]].dropna()
    if len(sub) < 3:
        print(f"[residuals] Not enough non-null overlap (n={len(sub)}); skipping {out_path}")
        return
    te = sub["shrunk_effect"].to_numpy()
    te_min, te_max = np.nanmin(te), np.nanmax(te)
    if not np.isfinite(te_min) or not np.isfinite(te_max) or te_max <= te_min:
        print("[residuals] Degenerate TE range; skipping residual plot.")
        return
    te_scaled = (te - te_min) / (te_max - te_min)
    resid = sub["maxdiff_mean"].to_numpy() - te_scaled
    resid = resid[np.isfinite(resid)]

    if len(resid) == 0:
        print(f"[residuals] All residuals non-finite; skipping {out_path}")
        return

    plt.figure(figsize=(8.5, 6))
    plt.hist(resid, bins=25, density=True, alpha=0.85)
    mu = float(np.nanmean(resid))
    if np.isfinite(mu):
        plt.axvline(mu, color="k", linestyle="--", linewidth=1, label=f"mean={mu:.3f}")
        plt.legend()
    plt.xlabel("Residual = MaxDiff − scaled TE")
    plt.ylabel("Density")
    plt.title("Residuals: MaxDiff vs Scaled RCT Effect")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[residuals] Saved {out_path} (n={len(resid)})")


def save_rank_overlap(df: pd.DataFrame, out_png: str, out_csv: str):
    # Need both series
    sub = df[["video_id_display", "maxdiff_mean", "shrunk_effect"]].dropna()
    if len(sub) < 10:
        print(f"[rank_overlap] Not enough rows with both metrics (n={len(sub)}); skipping.")
        return
    top10_md = sub.sort_values("maxdiff_mean", ascending=False).head(10)["video_id_display"].astype(str).tolist()
    top10_te = sub.sort_values("shrunk_effect", ascending=False).head(10)["video_id_display"].astype(str).tolist()
    overlap = len(set(top10_md) & set(top10_te))

    plt.figure(figsize=(7.5, 6))
    xs = ["Top10 MaxDiff only", "Overlap", "Top10 TE only"]
    vals = [10 - overlap, overlap, 10 - overlap]
    plt.bar(xs, vals)
    plt.title(f"Top-10 Overlap (n={overlap})")
    plt.ylabel("Count of videos")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[rank_overlap] Saved {out_png} (overlap={overlap})")

    # Save top/bottom tables to CSV with sections
    tb = {
        "top10_by_maxdiff": sub.sort_values("maxdiff_mean", ascending=False).head(10)[
            ["video_id_display", "maxdiff_mean"]],
        "bottom10_by_maxdiff": sub.sort_values("maxdiff_mean", ascending=True).head(10)[
            ["video_id_display", "maxdiff_mean"]],
        "top10_by_te": sub.sort_values("shrunk_effect", ascending=False).head(10)[
            ["video_id_display", "shrunk_effect"]],
        "bottom10_by_te": sub.sort_values("shrunk_effect", ascending=True).head(10)[
            ["video_id_display", "shrunk_effect"]],
    }
    with open(out_csv, "w") as f:
        for name, frame in tb.items():
            f.write(f"## {name}\n")
            frame.to_csv(f, index=False)
            f.write("\n")
    print(f"[rank_overlap] Saved tables to {out_csv}")


# --------------------------- Main --------------------------------------------

def main(out_dir: str, figures_dir: str, extra_plots: bool):
    ensure_dir(out_dir)
    ensure_dir(figures_dir)

    # Load effects and MaxDiff
    effects = pd.read_csv(os.path.join(out_dir, "per_video_effects.csv"))
    md_path = os.path.join(os.path.dirname(__file__), "..", "data", "maxdiff_dummy_data.csv")
    md = pd.read_csv(md_path)

    # Normalize IDs and merge
    effects["video_id_norm"] = norm_id_series(effects["video_id"])
    md["video_id_norm"] = norm_id_series(md["video_id"])

    df = effects.merge(
        md[["video_id_norm", "video_id", "maxdiff_mean", "sample_size"]],
        on="video_id_norm",
        how="left"
    )

    # Choose a single display ID column for plotting/tables
    if "video_id_x" in df.columns or "video_id_y" in df.columns:
        vid_x = df["video_id_x"] if "video_id_x" in df.columns else pd.Series(index=df.index, dtype="object")
        vid_y = df["video_id_y"] if "video_id_y" in df.columns else pd.Series(index=df.index, dtype="object")
        df["video_id_display"] = vid_y.fillna(vid_x)
    elif "video_id" in df.columns:
        df["video_id_display"] = df["video_id"]
    else:
        # fallback to normalized ID if neither present
        df["video_id_display"] = df["video_id_norm"]

    # Two-sided z-test p-values for naive effects (vectorized)
    if "naive_effect" in df.columns and "naive_se" in df.columns:
        z = df["naive_effect"] / df["naive_se"].replace(0, np.nan)
        pvals = 2 * norm.sf(np.abs(z))
        pvals = pd.Series(pvals).fillna(1.0)
        reject, p_adj = bh_fdr(pvals.values, alpha=0.05)
        df["pval"] = pvals.values
        df["pval_fdr"] = p_adj
        df["sig_fdr_05"] = reject
    else:
        print("[warn] Missing naive_effect/naive_se; p-values not computed.")

    # Save merged summary
    out_csv = os.path.join(out_dir, "te_vs_maxdiff_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[summary] Wrote {out_csv} (rows={len(df)})")

    # Core figures
    save_alignment_scatter(df, os.path.join(figures_dir, "alignment.png"))
    save_funnel(df, os.path.join(figures_dir, "funnel.png"))

    # Optional extras (each guards its own data needs)
    if extra_plots:
        save_hist_naive_vs_shrunk(df, os.path.join(figures_dir, "hist_naive_vs_shrunk.png"))
        save_hist_maxdiff(df, os.path.join(figures_dir, "hist_maxdiff.png"))
        save_residuals(df, os.path.join(figures_dir, "residuals_maxdiff_minus_te.png"))
        save_rank_overlap(
            df,
            os.path.join(figures_dir, "rank_overlap_top10.png"),
            os.path.join(figures_dir, "top_bottom_tables.csv"),
        )

    print("[done] Comparison complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    ap.add_argument("--figures-dir", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    ap.add_argument("--extra-plots", action="store_true", help="Generate extra exploratory plots")
    args = ap.parse_args()
    main(os.path.abspath(args.out_dir), os.path.abspath(args.figures_dir), args.extra_plots)