
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

    # Quick balance check overall
    treated = rct[rct["treated"]==1]
    control = rct[rct["treated"]==0]
    balance_rows = []
    num_cols = ["partisanship","democratic_party_fav","republican_party_fav","trump_fav","biden_fav"]
    for col in num_cols:
        if col in rct.columns:
            smd = standardized_mean_diff(treated[col].values, control[col].values)
            balance_rows.append({"var": col, "std_mean_diff": smd, "treated_mean": treated[col].mean(), "control_mean": control[col].mean()})
    bal_df = pd.DataFrame(balance_rows).sort_values("std_mean_diff", key=lambda s: s.abs(), ascending=False)
    bal_df.to_csv(os.path.join(out_dir, "balance_overall.csv"), index=False)

    # Per-video counts
    vc = rct.groupby(["video_id","treated"])["trump_approval"].agg(["count","mean"]).unstack(fill_value=0)
    vc.columns = [f"{a}_{b}" for a,b in vc.columns]
    vc = vc.reset_index()
    vc.to_csv(os.path.join(out_dir, "per_video_counts.csv"), index=False)

    # Join RCT with transcripts (inner join on video_id for treated rows)
    rct_with_text = rct.merge(transcripts, on="video_id", how="left")
    rct_with_text.to_csv(os.path.join(out_dir, "rct_with_text.csv"), index=False)

    # Save a tiny EDA summary
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
