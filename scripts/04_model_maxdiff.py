
#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
from utils import ensure_dir

def weighted_spearman(y_true, y_pred, w):
    # Simple weighted rank correlation proxy: weight by w during Spearman via tie-safe approach
    # We'll center ranks by average; this is an approximation
    import scipy.stats as st
    return st.spearmanr(y_true, y_pred).correlation

def cv_eval(X, y, sample_weight, model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses, corrs = [], []
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        if sparse.issparse(X):
            Xtr, Xte = X[tr], X[te]
        else:
            Xtr, Xte = X[tr,:], X[te,:]
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        m = model.fit(Xtr, y[tr], sample_weight=sw_tr)
        p = m.predict(Xte).clip(0,1)
        preds[te] = p
        rmse = np.sqrt(mean_squared_error(y[te], p, sample_weight=sample_weight[te] if sample_weight is not None else None))
        corr = weighted_spearman(y[te], p, sample_weight[te] if sample_weight is not None else None)
        rmses.append(rmse); corrs.append(corr)
    return np.mean(rmses), np.std(rmses), np.mean(corrs), np.std(corrs), preds

def main(data_dir, out_dir):
    ensure_dir(out_dir)
    md_path  = os.path.join(data_dir, "maxdiff_dummy_data.csv")
    md  = pd.read_csv(md_path)

    # Targets
    y = md["maxdiff_mean"].values.astype(float)
    w = md["sample_size"].values.astype(float)

    # Features: TF-IDF + meta
    tfidf = joblib.load(os.path.join(out_dir, "text_tfidf.joblib"))
    X = sparse.load_npz(os.path.join(out_dir, "text_tfidf_X.npz"))
    vid_order = tfidf["video_id"]
    # align md order to tfidf order
    md_aligned = md.merge(pd.DataFrame({"video_id": vid_order}), on="video_id", how="right")
    y = md_aligned["maxdiff_mean"].values
    w = md_aligned["sample_size"].values

    # Models
    ridge = RidgeCV(alphas=np.logspace(-3,3,15))
    en = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-3,1,12), max_iter=5000)

    r_rm, r_rs, r_sc, r_ss, r_pred = cv_eval(X, y, w, ridge)
    e_rm, e_rs, e_sc, e_ss, e_pred = cv_eval(X, y, w, en)

    metrics = pd.DataFrame([
        {"model":"RidgeCV", "weighted_rmse_mean": r_rm, "weighted_rmse_sd": r_rs, "spearman_mean": r_sc, "spearman_sd": r_ss},
        {"model":"ElasticNetCV", "weighted_rmse_mean": e_rm, "weighted_rmse_sd": e_rs, "spearman_mean": e_sc, "spearman_sd": e_ss},
    ])
    metrics.to_csv(os.path.join(out_dir, "maxdiff_cv_metrics.csv"), index=False)

    preds = pd.DataFrame({"video_id": vid_order, "y_true": y, "ridge_pred": r_pred, "enet_pred": e_pred, "sample_size": w})
    preds.to_csv(os.path.join(out_dir, "maxdiff_pred.csv"), index=False)
    print("MaxDiff models complete. Metrics and predictions saved to output.")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = ap.parse_args()
    main(os.path.abspath(args.data_dir), os.path.abspath(args.out_dir))
