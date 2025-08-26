
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def standardized_mean_diff(a: np.ndarray, b: np.ndarray):
    """Cohen's d variant for quick treated vs control balance check."""
    a = a.astype(float); b = b.astype(float)
    ma, mb = np.nanmean(a), np.nanmean(b)
    sa, sb = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    s_pool = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
    if s_pool == 0:
        return 0.0
    return (ma - mb) / s_pool

def two_prop_effect_and_se(success1, n1, success0, n0):
    """Difference in proportions p1 - p0 and its SE under independence."""
    p1 = success1 / max(n1, 1)
    p0 = success0 / max(n0, 1)
    var = (p1*(1-p1)/max(n1,1)) + (p0*(1-p0)/max(n0,1))
    se = np.sqrt(var)
    return p1 - p0, se

def dersimonian_laird_tau2(effects, ses):
    """Random-effects tau^2 via DerSimonian-Laird (non-negative)."""
    w = 1.0 / np.square(ses)
    ybar = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * np.square(effects - ybar))
    k = len(effects)
    c = np.sum(w) - (np.sum(np.square(w)) / np.sum(w))
    tau2 = max((Q - (k - 1)) / max(c, 1e-12), 0.0)
    return tau2

def eb_shrinkage(effects, ses):
    """Empirical Bayes shrinkage toward the precision-weighted mean using tau^2 (DL)."""
    effects = np.asarray(effects)
    ses = np.asarray(ses)
    tau2 = dersimonian_laird_tau2(effects, ses)
    vi = np.square(ses)
    # precision-weighted mean (fixed-effects)
    wi = 1.0 / np.maximum(vi, 1e-12)
    mu = np.sum(wi * effects) / np.sum(wi)
    # posterior means
    post = (effects/vi + mu/np.maximum(tau2, 1e-12 if tau2==0 else tau2)) / (1.0/vi + 1.0/np.maximum(tau2, 1e-12 if tau2==0 else tau2))
    # posterior SEs
    post_se = np.sqrt(1.0 / (1.0/vi + 1.0/np.maximum(tau2, 1e-12 if tau2==0 else tau2)))
    return post, post_se, tau2, mu

def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction; returns mask and adjusted pvals."""
    pvals = np.asarray(pvals)
    rej, p_adj = fdrcorrection(pvals, alpha=alpha, method='indep')
    return rej, p_adj

def z_to_ci(z, se, alpha=0.05):
    zcrit = stats.norm.ppf(1 - alpha/2)
    lo = z - zcrit*se
    hi = z + zcrit*se
    return lo, hi

def prob_ci(p, se, alpha=0.05):
    zcrit = stats.norm.ppf(1 - alpha/2)
    return p - zcrit*se, p + zcrit*se
