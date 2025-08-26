
#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import joblib
from utils import ensure_dir

def cv_eval_regression(X, y, model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses, maes, preds = [], [], np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = (X[tr], X[te]) if sparse.issparse(X) else (X[tr,:], X[te,:])
        m = model.fit(Xtr, y[tr])
        p = m.predict(Xte)
        preds[te] = p
        rmse = np.sqrt(mean_squared_error(y[te], p))
        mae = np.mean(np.abs(y[te]-p))
        rmses.append(rmse); maes.append(mae)
    return np.mean(rmses), np.std(rmses), np.mean(maes), np.std(maes), preds

def cv_eval_auc(X, y_bin, model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs, preds = [], np.zeros_like(y_bin, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = (X[tr], X[te]) if sparse.issparse(X) else (X[tr,:], X[te,:])
        m = model.fit(Xtr, y_bin[tr])
        p = m.predict_proba(Xte)[:,1]
        preds[te] = p
        aucs.append(roc_auc_score(y_bin[te], p))
    return np.mean(aucs), np.std(aucs), preds

def main(out_dir):
    ensure_dir(out_dir)
    effects = pd.read_csv(os.path.join(out_dir, "per_video_effects.csv"))
    tfidf = joblib.load(os.path.join(out_dir, "text_tfidf.joblib"))
    X = sparse.load_npz(os.path.join(out_dir, "text_tfidf_X.npz"))
    vid_order = tfidf["video_id"]

    # Align
    eff = effects.merge(pd.DataFrame({"video_id": vid_order}), on="video_id", how="right")
    y = eff["shrunk_effect"].values  # target on probability difference scale
    y_bin = (y > 0).astype(int)

    ridge = RidgeCV(alphas=np.logspace(-3,3,15))
    en = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-3,1,12), max_iter=5000)
    logit = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", solver="lbfgs", max_iter=1000)

    r_rm, r_rs, r_mae, r_maes, r_pred = cv_eval_regression(X, y, ridge)
    e_rm, e_rs, e_mae, e_maes, e_pred = cv_eval_regression(X, y, en)
    a_m, a_s, a_pred = cv_eval_auc(X, y_bin, logit)

    metrics = pd.DataFrame([
        {"model":"RidgeCV", "rmse_mean": r_rm, "rmse_sd": r_rs, "mae_mean": r_mae, "mae_sd": r_maes},
        {"model":"ElasticNetCV", "rmse_mean": e_rm, "rmse_sd": e_rs, "mae_mean": e_mae, "mae_sd": e_maes},
        {"model":"LogitCV(binary>0)", "auc_mean": a_m, "auc_sd": a_s},
    ])
    metrics.to_csv(os.path.join(out_dir, "te_model_cv_metrics.csv"), index=False)

    preds = pd.DataFrame({"video_id": vid_order, "y_true": y, "ridge_pred": r_pred, "enet_pred": e_pred, "logit_prob_pos": a_pred})
    preds.to_csv(os.path.join(out_dir, "te_pred.csv"), index=False)
    print("TE-from-text models complete. Metrics and predictions saved.")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = ap.parse_args()
    main(os.path.abspath(args.out_dir))
