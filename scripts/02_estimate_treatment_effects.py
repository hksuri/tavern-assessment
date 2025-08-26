
#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils import ensure_dir, two_prop_effect_and_se, eb_shrinkage, prob_ci

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

def adjusted_logit_effects(df: pd.DataFrame):
    """Logit with treated * C(video_id) + covariates; extract per-video treated effect (log-odds)."""
    covs = ["partisanship","democratic_party_fav","republican_party_fav","trump_fav","biden_fav","C(vote_pres_2020)","C(vote_pres_2024)"]
    covs_present = []
    for c in ["partisanship","democratic_party_fav","republican_party_fav","trump_fav","biden_fav"]:
        if c in df.columns:
            covs_present.append(c)
    if "vote_pres_2020" in df.columns:
        covs_present.append("C(vote_pres_2020)")
    if "vote_pres_2024" in df.columns:
        covs_present.append("C(vote_pres_2024)")

    formula = "trump_approval ~ treated + C(video_id) + treated:C(video_id)"
    if len(covs_present)>0:
        formula += " + " + " + ".join(covs_present)
    model = smf.logit(formula, data=df).fit(disp=False)
    rob = model.get_robustcov_results(cov_type="HC1")
    params = rob.params
    cov = rob.cov_params()

    vids = sorted(df["video_id"].unique())
    # Determine baseline category for video_id in statsmodels
    # It is alphabetically first by default; statsmodels encodes as C(video_id)[T.x] for non-baseline
    baseline = min(vids)
    rows = []
    for vid in vids:
        if vid == baseline:
            coef = params.get("treated", np.nan)
            var = cov.loc["treated","treated"] if "treated" in cov.index else np.nan
        else:
            iname = f"treated:C(video_id)[T.{vid}]"
            base = "treated"
            coef = params.get(base, 0.0) + params.get(iname, 0.0)
            # var of sum = var(a)+var(b)+2cov(a,b)
            v = 0.0
            if base in cov.index and base in cov.columns:
                v += cov.loc[base, base]
            if iname in cov.index and iname in cov.columns:
                v += cov.loc[iname, iname]
            if base in cov.index and iname in cov.columns:
                v += 2*cov.loc[base, iname]
            var = v
        se = np.sqrt(max(var, 0.0))
        rows.append({
            "video_id": vid,
            "adj_logit_effect": coef,
            "adj_logit_se": se,
            "adj_or": np.exp(coef) if np.isfinite(coef) else np.nan
        })
    return pd.DataFrame(rows)

def main(data_dir, out_dir):
    ensure_dir(out_dir)
    rct_path = os.path.join(data_dir, "rct_dummy_data.csv")
    df = pd.read_csv(rct_path)

    # NAIVE
    naive = naive_effects(df)

    # SHRINKAGE (EB) on naive effects
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

    # ADJUSTED (logit)
    adj = adjusted_logit_effects(df)

    out = naive.merge(adj, on="video_id", how="left")
    out = out.sort_values("shrunk_effect", ascending=False)
    out.to_csv(os.path.join(out_dir, "per_video_effects.csv"), index=False)
    print("Per-video effects written to:", os.path.join(out_dir, "per_video_effects.csv"))

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = ap.parse_args()
    main(os.path.abspath(args.data_dir), os.path.abspath(args.out_dir))
