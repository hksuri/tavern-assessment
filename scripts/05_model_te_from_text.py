#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import joblib

def norm_id_series(s: pd.Series) -> pd.Series:
    """Robust ID normalizer:
       - strip whitespace
       - try numeric → cast to int → back to string (so '21.0' and '21' match)
       - lowercase fallback for non-numeric IDs
    """
    s = s.astype("string").str.strip()
    num = pd.to_numeric(s, errors="coerce")
    out = s.copy()
    mask = num.notna()
    out.loc[mask] = num.loc[mask].astype("int64").astype(str)
    return out.str.lower()

def cv_eval_regression(X, y, model, max_splits=5):
    from scipy import sparse as sp
    n = X.shape[0]
    n_splits = min(max_splits, n) if n > 1 else 1
    if n_splits < 2:
        m = model.fit(X, y) if n > 0 else model
        preds = m.predict(X) if n > 0 else np.array([])
        return np.nan, np.nan, np.nan, np.nan, preds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses, maes, preds = [], [], np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = (X[tr], X[te]) if sp.issparse(X) else (X[tr,:], X[te,:])
        m = model.fit(Xtr, y[tr])
        p = m.predict(Xte)
        preds[te] = p
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(np.mean(np.abs(y[te]-p)))
    return float(np.mean(rmses)), float(np.std(rmses)), float(np.mean(maes)), float(np.std(maes)), preds

def cv_eval_auc_stratified(X, y_bin, model, max_splits=5):
    from scipy import sparse as sp
    classes, counts = np.unique(y_bin, return_counts=True)
    if len(classes) < 2:
        return None
    n_splits = min(max_splits, counts.min())
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, preds = [], np.full_like(y_bin, fill_value=np.nan, dtype=float)
    for tr, te in skf.split(np.zeros_like(y_bin), y_bin):
        if len(np.unique(y_bin[tr])) < 2 or len(np.unique(y_bin[te])) < 2:
            continue
        Xtr, Xte = (X[tr], X[te]) if sp.issparse(X) else (X[tr,:], X[te,:])
        m = model.fit(Xtr, y_bin[tr])
        p = m.predict_proba(Xte)[:,1]
        preds[te] = p
        aucs.append(roc_auc_score(y_bin[te], p))
    if len(aucs) == 0:
        return None
    return (float(np.mean(aucs)), float(np.std(aucs)), preds)

def main(out_dir, data_dir):
    # Load artifacts
    effects = pd.read_csv(os.path.join(out_dir, "per_video_effects.csv"))
    tfidf = joblib.load(os.path.join(out_dir, "text_tfidf.joblib"))
    X_full = sparse.load_npz(os.path.join(out_dir, "text_tfidf_X.npz"))
    md = pd.read_csv(os.path.join(data_dir, "maxdiff_dummy_data.csv"))

    # Normalize IDs everywhere
    effects["video_id_norm"] = norm_id_series(effects["video_id"])
    md["video_id_norm"] = norm_id_series(md["video_id"])

    # Reconstruct TF-IDF row order (the features were built from md drop_duplicates)
    md_unique = md.drop_duplicates("video_id_norm").copy()
    vid_order_norm = md_unique["video_id_norm"].reset_index(drop=True)

    # Align effects to TF-IDF order
    eff_sub = md_unique[["video_id_norm", "video_id"]].merge(
        effects[["video_id_norm","shrunk_effect"]],
        on="video_id_norm",
        how="left"
    )
    y = eff_sub["shrunk_effect"].to_numpy(dtype=float)

    # Diagnostics: write coverage
    pd.DataFrame({
        "video_id_raw": md_unique["video_id"],
        "video_id_norm": vid_order_norm,
        "has_shrunk_effect": ~np.isnan(y)
    }).to_csv(os.path.join(out_dir, "te_from_text_alignment_diagnostics.csv"), index=False)

    # Mask rows with a target
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        # Graceful exit if still no overlap
        pd.DataFrame([
            {"model":"RidgeCV","rmse_mean":np.nan,"rmse_sd":np.nan,"mae_mean":np.nan,"mae_sd":np.nan},
            {"model":"ElasticNetCV","rmse_mean":np.nan,"rmse_sd":np.nan,"mae_mean":np.nan,"mae_sd":np.nan},
            {"model":"LogitCV(binary>0)","auc_mean":np.nan,"auc_sd":np.nan},
        ]).to_csv(os.path.join(out_dir, "te_model_cv_metrics.csv"), index=False)
        pd.DataFrame({
            "video_id": md_unique["video_id"],
            "y_true": y,
            "ridge_pred": np.nan,
            "enet_pred": np.nan,
            "logit_prob_pos": np.nan
        }).to_csv(os.path.join(out_dir, "te_pred.csv"), index=False)
        print("No matched targets after ID normalization; see alignment diagnostics.")
        return

    # Subset
    X = X_full[mask]
    y_sub = y[mask]

    # Regression models
    ridge = RidgeCV(alphas=np.logspace(-3,3,15))
    en = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-3,1,12), max_iter=5000)
    r_rm, r_rs, r_mae, r_maes, r_pred_sub = cv_eval_regression(X, y_sub, ridge)
    e_rm, e_rs, e_mae, e_maes, e_pred_sub = cv_eval_regression(X, y_sub, en)

    # Reinsert predictions aligned to full list
    ridge_pred = np.full_like(y, np.nan, dtype=float)
    enet_pred  = np.full_like(y, np.nan, dtype=float)
    ridge_pred[mask] = r_pred_sub
    enet_pred[mask]  = e_pred_sub

    # Classification (optional)
    y_bin_sub = (y_sub > 0).astype(int)
    logit = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced")
    auc_result = cv_eval_auc_stratified(X, y_bin_sub, logit, max_splits=5)
    logit_pred = np.full_like(y, np.nan, dtype=float)
    metrics_rows = [
        {"model":"RidgeCV", "rmse_mean": r_rm, "rmse_sd": r_rs, "mae_mean": r_mae, "mae_sd": r_maes},
        {"model":"ElasticNetCV", "rmse_mean": e_rm, "rmse_sd": e_rs, "mae_mean": e_mae, "mae_sd": e_maes},
    ]
    if auc_result is None:
        metrics_rows.append({"model":"LogitCV(binary>0)","auc_mean":np.nan,"auc_sd":np.nan})
    else:
        a_m, a_s, a_pred_sub = auc_result
        metrics_rows.append({"model":"LogitCV(binary>0)","auc_mean":a_m,"auc_sd":a_s})
        logit_pred[mask] = a_pred_sub

    # Save
    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "te_model_cv_metrics.csv"), index=False)
    pd.DataFrame({
        "video_id": md_unique["video_id"],  # human-friendly original form from MaxDiff file
        "y_true": y,
        "ridge_pred": ridge_pred,
        "enet_pred": enet_pred,
        "logit_prob_pos": logit_pred
    }).to_csv(os.path.join(out_dir, "te_pred.csv"), index=False)
    print("TE-from-text models complete. Metrics and predictions saved.")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()
    main(os.path.abspath(args.out_dir), os.path.abspath(args.data_dir))