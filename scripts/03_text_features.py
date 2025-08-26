
#!/usr/bin/env python3
import os, argparse, re, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from utils import ensure_dir

class TextMeta(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        # X is iterable of texts
        feats = []
        for t in X:
            t = t if isinstance(t, str) else ""
            words = re.findall(r"\b\w+\b", t.lower())
            n_words = len(words)
            n_chars = len(t)
            avg_wlen = (sum(len(w) for w in words) / n_words) if n_words>0 else 0.0
            n_digits = sum(ch.isdigit() for ch in t)
            n_upper  = sum(ch.isupper() for ch in t)
            n_sentences = max(1, t.count(".") + t.count("!") + t.count("?"))
            type_token = (len(set(words))/n_words) if n_words>0 else 0.0
            feats.append([n_words, n_chars, avg_wlen, n_digits, n_upper, n_sentences, type_token])
        return np.array(feats)

def main(data_dir, out_dir, max_features):
    ensure_dir(out_dir)
    md_path  = os.path.join(data_dir, "maxdiff_dummy_data.csv")
    md  = pd.read_csv(md_path)
    # Deduplicate per video_id for text
    txt = md.drop_duplicates("video_id")[["video_id","text"]].copy()

    # Save meta features
    meta_extractor = TextMeta()
    meta = meta_extractor.transform(txt["text"].tolist())
    meta_cols = ["n_words","n_chars","avg_word_len","n_digits","n_upper","n_sentences","type_token_ratio"]
    meta_df = pd.DataFrame(meta, columns=meta_cols)
    meta_df.insert(0, "video_id", txt["video_id"].values)
    meta_df.to_csv(os.path.join(out_dir, "text_meta.csv"), index=False)

    # TF-IDF
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=max_features, strip_accents="unicode", lowercase=True)
    X = vec.fit_transform(txt["text"].fillna(""))
    joblib.dump({"vectorizer": vec, "video_id": txt["video_id"].values}, os.path.join(out_dir, "text_tfidf.joblib"))
    # save sparse matrix as .npz
    from scipy import sparse
    sparse.save_npz(os.path.join(out_dir, "text_tfidf_X.npz"), X)
    print("Saved TF-IDF and text meta to:", out_dir)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out-dir", default=os.path.dirname(__file__) + "/../output")
    ap.add_argument("--max-features", type=int, default=5000)
    args = ap.parse_args()
    main(os.path.abspath(args.data_dir), os.path.abspath(args.out_dir), args.max_features)
