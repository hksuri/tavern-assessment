#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from utils import ensure_dir, two_prop_effect_and_se, eb_shrinkage, prob_ci

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Candidate covariates mentioned in spec
CAND_NUM = ["partisanship","democratic_party_fav","republican_party_fav","trump_fav","biden_fav"]
CAND_CAT = ["vote_pres_2020","vote_pres_2024","partisanship"]  # treat partisanship as cat if non-numeric

def naive_effects(df: pd.DataFrame):
    """Compute per-video naive difference vs pooled controls."""
    controls = df[df["treated"]==0]
    n0 = len(controls)
    y0 = controls["trump_approval"].sum()
    p0 = y0 / max(n0, 1)

    rows = []
    g = df[df["treated"]==1].groupby("video_id")
    for vid, d in g:
        n1 = len(d)
        y1 = d["trump_approval"].sum()
        eff, se = two_prop_effect_and_se(y1, n1, y0, n0)
        lo, hi = prob_ci(eff, se)
        rows.append({
            "video_id": vid,
            "naive_effect": eff,
            "naive_se": se,
            "naive_ci_lo": lo,
            "naive_ci_hi": hi,
            "treated_n": n1,
            "treated_rate": y1/max(n1,1),
            "control_n": n0,
            "control_rate": p0
        })
    return pd.DataFrame(rows)

def adjusted_logit_effects_sk(df: pd.DataFrame):
    """
    Penalized logistic regression with vectorized per-video treated dummies:
      D_vid = 1 if treated==1 and video_id==vid else 0
    Target: trump_approval (0/1).
    """
    df = df.copy()

    # Build DVID matrix from the FULL column so rows align even if controls have NaN video_id
    vid_series_full = df["video_id"].astype(str).fillna("__NA__")
    dvid = pd.get_dummies(vid_series_full, prefix="DVID", dtype=int)

    # Zero out all control rows in one shot (rows align with df by construction)
    treated_mask = df["treated"].astype(int).to_numpy()[:, None]
    dvid = dvid.to_numpy() * treated_mask  # use numpy for speed/broadcast
    dvid = pd.DataFrame(dvid, columns=pd.get_dummies(vid_series_full, prefix="DVID", dtype=int).columns, index=df.index)

    # Infer covariate types from the actual data
    present_cols = set(df.columns)
    cand_num = ["partisanship","democratic_party_fav","republican_party_fav","trump_fav","biden_fav"]
    cand_cat = ["vote_pres_2020","vote_pres_2024","partisanship"]  # treat as categorical if non-numeric

    num_covs = [c for c in cand_num if c in present_cols and pd.api.types.is_numeric_dtype(df[c])]
    cat_covs = [c for c in cand_cat if c in present_cols and not pd.api.types.is_numeric_dtype(df[c])]

    # Minimal imputation
    for c in num_covs:
        df[c] = df[c].astype(float).fillna(df[c].astype(float).median())
    for c in cat_covs:
        df[c] = df[c].astype(str).fillna("Missing")

    # Assemble base design matrix: [DVID_* | numeric covs]
    X_parts = [dvid]
    if num_covs:
        X_parts.append(df[num_covs])
    X_base = pd.concat(X_parts, axis=1)

    y = df["trump_approval"].astype(int).values

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    cat_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(X_base.columns)),
            ("cat", cat_transformer, cat_covs),
        ],
        remainder="drop"
    )

    logit = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced"  # helps if approval is imbalanced
    )

    # Fit on combined numeric block + raw categorical block (preprocessor splits them correctly)
    X_fit = pd.concat([X_base, df[cat_covs]], axis=1)
    pipe = Pipeline([("prep", pre), ("clf", logit)])
    pipe.fit(X_fit, y)

    # Recover coefficients for DVID_* (they come first in the numeric block)
    clf = pipe.named_steps["clf"]
    coef = clf.coef_.ravel()

    k_dvid = dvid.shape[1]
    dvid_coefs = coef[:k_dvid]
    dvid_cols = list(dvid.columns)

    out_rows = []
    for col, beta in zip(dvid_cols, dvid_coefs):
        vid = col.replace("DVID_", "")
        if vid == "__NA__":
            # This corresponds to controls (no specific video); skip reporting an effect for it
            continue
        out_rows.append({
            "video_id": vid,
            "adj_logit_effect": float(beta),
            "adj_or": float(np.exp(beta)) if np.isfinite(beta) else np.nan
        })
    return pd.DataFrame(out_rows)

def main(data_dir, out_dir):
    ensure_dir(out_dir)
    rct_path = os.path.join(data_dir, "rct_dummy_data.csv")
    df = pd.read_csv(rct_path)

    # NAIVE effects (Δ proportion vs pooled controls)
    naive = naive_effects(df)

    # EB SHRINKAGE on naive effects
    post, post_se, tau2, mu = eb_shrinkage(naive["naive_effect"].values, naive["naive_se"].values)
    naive["shrunk_effect"] = post
    naive["shrunk_se"] = post_se
    lo, hi = [], []
    for e, s in zip(post, post_se):
        l, h = prob_ci(e, s)
        lo.append(l); hi.append(h)
    naive["shrunk_ci_lo"] = lo
    naive["shrunk_ci_hi"] = hi
    naive["tau2_overall"] = tau2
    naive["mu_overall"] = mu

    # ADJUSTED (penalized logit) using vectorized per-video treated dummies
    adj = adjusted_logit_effects_sk(df)

    # Join (coerce both video_id to string to avoid mismatched types)
    naive["video_id"] = naive["video_id"].astype(str)
    adj["video_id"] = adj["video_id"].astype(str)

    out = naive.merge(adj, on="video_id", how="left")
    out = out.sort_values("shrunk_effect", ascending=False)
    out.to_csv(os.path.join(out_dir, "per_video_effects.csv"), index=False)
    print("Per-video effects written to:", os.path.join(out_dir, "per_video_effects.csv"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = ap.parse_args()
    main(os.path.abspath(args.data_dir), os.path.abspath(args.out_dir))