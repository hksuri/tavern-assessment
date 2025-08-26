#!/usr/bin/env python3
import os, argparse, pandas as pd, numpy as np
from utils import ensure_dir, standardized_mean_diff

def main(data_dir, out_dir):
    ensure_dir(out_dir)
    rct_path = os.path.join(data_dir, "rct_dummy_data.csv")
    md_path  = os.path.join(data_dir, "maxdiff_dummy_data.csv")
    rct = pd.read_csv(rct_path)
    md  = pd.read_csv(md_path)

    # Basic integrity
    assert {"treated","trump_approval","video_id"}.issubset(rct.columns), "RCT columns missing"
    assert {"video_id","text","maxdiff_mean","sample_size"}.issubset(md.columns), "MaxDiff columns missing"

    # Drop obvious duplicates on transcripts (keep first, but log)
    dup_vids = md[md.duplicated("video_id", keep=False)].sort_values("video_id")
    if len(dup_vids) > 0:
        dup_vids.to_csv(os.path.join(out_dir, "transcript_duplicates.csv"), index=False)
        md = md.drop_duplicates("video_id", keep="first")

    # Save a clean transcript map
    transcripts = md[["video_id","text"]].copy()
    transcripts.to_csv(os.path.join(out_dir, "transcripts.csv"), index=False)

    # Balance check: numeric SMDs + categorical proportion diffs
    treated = rct[rct["treated"]==1]
    control = rct[rct["treated"]==0]

    covars = [
        "partisanship","democratic_party_fav","republican_party_fav",
        "trump_fav","biden_fav","vote_pres_2020","vote_pres_2024"
    ]
    covars = [c for c in covars if c in rct.columns]

    rows = []
    for col in covars:
        s = rct[col]
        if pd.api.types.is_numeric_dtype(s):
            # numeric: SMD
            smd = standardized_mean_diff(
                treated[col].astype(float).values,
                control[col].astype(float).values
            )
            rows.append({
                "var": col,
                "type": "numeric",
                "metric": "std_mean_diff",
                "value": smd,
                "treated_mean": treated[col].mean(),
                "control_mean": control[col].mean()
            })
        else:
            # categorical: per-level proportion difference (treated - control)
            levels = s.dropna().unique().tolist()
            for lvl in levels:
                p_t = (treated[col] == lvl).mean()
                p_c = (control[col] == lvl).mean()
                rows.append({
                    "var": f"{col}=={lvl}",
                    "type": "categorical",
                    "metric": "prop_diff",
                    "value": p_t - p_c,
                    "treated_prop": p_t,
                    "control_prop": p_c
                })

    bal_df = pd.DataFrame(rows)
    # Order: numeric first by |value|, then categorical by |value|
    bal_df = pd.concat([
        bal_df[bal_df["type"]=="numeric"].sort_values("value", key=lambda s: s.abs(), ascending=False),
        bal_df[bal_df["type"]=="categorical"].sort_values("value", key=lambda s: s.abs(), ascending=False)
    ], ignore_index=True)
    bal_df.to_csv(os.path.join(out_dir, "balance_overall.csv"), index=False)

    # Per-video counts
    vc = rct.groupby(["video_id","treated"])["trump_approval"].agg(["count","mean"]).unstack(fill_value=0)
    vc.columns = [f"{a}_{b}" for a,b in vc.columns]
    vc = vc.reset_index()
    vc.to_csv(os.path.join(out_dir, "per_video_counts.csv"), index=False)

    # Join RCT with transcripts (left join)
    rct_with_text = rct.merge(transcripts, on="video_id", how="left")
    rct_with_text.to_csv(os.path.join(out_dir, "rct_with_text.csv"), index=False)

    # Small EDA summary
    summary = {
        "n_rows_rct": len(rct),
        "n_rows_maxdiff": len(md),
        "n_videos_rct": rct["video_id"].nunique(),
        "n_videos_maxdiff": md["video_id"].nunique(),
        "n_treated": int((rct["treated"]==1).sum()),
        "n_control": int((rct["treated"]==0).sum()),
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "qc_summary.csv"))
    print("QC & join complete. Outputs written to:", out_dir)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = ap.parse_args()
    main(os.path.abspath(args.data_dir), os.path.abspath(args.out_dir))