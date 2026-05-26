"""Banking77 → tabular features.

Banking77 is a 77-class intent classification dataset published by PolyAI
(13 083 examples, single-sentence customer-banking queries). We use it as
the headline benchmark because the only inputs are raw strings, which
forces every feature on the table to be derived — exactly what we want
the evolutionary loop to do.

Pipeline:

  1. Pull `PolyAI/banking77` from HuggingFace (train + test concatenated;
     we make our own 5-fold splits anyway).
  2. Derive a base tabular feature set from the query text — lengths,
     punctuation counts, simple NER counts via spaCy (lazy-load) and a
     TF-IDF top-50 numeric block.
  3. Save the resulting frame + 5-fold stratified split indices as
     Parquet under ``data/banking77/`` (the train/test split from
     upstream is ignored — we cross-validate).

Re-running the script is idempotent. If ``data/banking77/features.parquet``
already exists it is overwritten; the split indices file is rewritten too
so seed changes are picked up.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


log = logging.getLogger("evofeat.banking77")

OUT_DIR = "data/banking77"
PARQUET_PATH = os.path.join(OUT_DIR, "features.parquet")
SPLITS_PATH = os.path.join(OUT_DIR, "splits.npz")
META_PATH = os.path.join(OUT_DIR, "metadata.json")


_QUESTION_WORDS = {"what", "when", "where", "who", "whom", "whose", "why",
                   "how", "which", "can", "could", "do", "does", "is", "am",
                   "are", "should", "would", "will"}


def _load_hf() -> pd.DataFrame:
    """Pull the dataset off HuggingFace. Concatenate train + test."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "the 'datasets' library is required for the Banking77 prep "
            "script. install with: pip install datasets"
        ) from e

    log.info("loading PolyAI/banking77 from huggingface…")
    ds = load_dataset("PolyAI/banking77")
    frames = []
    for split_name in ("train", "test"):
        if split_name in ds:
            d = ds[split_name].to_pandas()
            d["source_split"] = split_name
            frames.append(d)
    if not frames:
        raise RuntimeError("no train/test splits found on PolyAI/banking77")
    df = pd.concat(frames, axis=0, ignore_index=True)
    # column rename for consistency
    if "label" in df.columns:
        df = df.rename(columns={"label": "intent_id"})
    df = df.rename(columns={"text": "query"})
    # the dataset object holds the int → label mapping under
    # ds["train"].features["label"].names — keep that around so the meta
    # file is self-describing
    names = ds["train"].features["label"].names
    df["intent_name"] = df["intent_id"].map(lambda i: names[i])
    return df


def _lex_features(s: str) -> dict:
    s = s if isinstance(s, str) else ""
    tokens = s.split()
    n_tok = max(len(tokens), 1)
    chars = list(s)
    n_chars = max(len(chars), 1)
    digits = sum(c.isdigit() for c in s)
    punct = sum(c in string.punctuation for c in s)
    uppers = sum(c.isupper() for c in s)
    avg_word_len = sum(len(t) for t in tokens) / n_tok
    has_question = int("?" in s)
    has_money = int(bool(re.search(r"[\$£€¥]|\bdollar|\beuro|\bpound|\bgbp|\busd|\beur", s.lower())))
    has_number = int(bool(re.search(r"\d", s)))
    lower_tokens = [t.strip(string.punctuation).lower() for t in tokens]
    starts_with_q = int(bool(lower_tokens) and lower_tokens[0] in _QUESTION_WORDS)
    q_word_count = sum(1 for t in lower_tokens if t in _QUESTION_WORDS)
    # ttr — type/token ratio; cheap proxy for lexical diversity
    ttr = len(set(lower_tokens)) / n_tok
    return {
        "n_chars": n_chars,
        "n_tokens": n_tok,
        "avg_word_len": avg_word_len,
        "n_digits": digits,
        "n_punct": punct,
        "n_uppercase": uppers,
        "has_question_mark": has_question,
        "has_money_symbol": has_money,
        "has_number": has_number,
        "starts_with_qword": starts_with_q,
        "qword_count": q_word_count,
        "type_token_ratio": ttr,
    }


def _spacy_features(texts: List[str]) -> pd.DataFrame:
    try:
        import spacy
    except ImportError:
        log.warning("spaCy not installed — skipping NER counts (set to 0)")
        return pd.DataFrame({
            "ner_count": np.zeros(len(texts), dtype=int),
            "ner_money": np.zeros(len(texts), dtype=int),
            "ner_date": np.zeros(len(texts), dtype=int),
            "ner_gpe":  np.zeros(len(texts), dtype=int),
            "ner_org":  np.zeros(len(texts), dtype=int),
        })

    # try the small english model — fall back to blank if not downloaded
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except OSError:
        log.warning(
            "spaCy model en_core_web_sm not found — install it with:\n"
            "  python -m spacy download en_core_web_sm\n"
            "falling back to zero NER counts."
        )
        return pd.DataFrame({
            "ner_count": np.zeros(len(texts), dtype=int),
            "ner_money": np.zeros(len(texts), dtype=int),
            "ner_date": np.zeros(len(texts), dtype=int),
            "ner_gpe":  np.zeros(len(texts), dtype=int),
            "ner_org":  np.zeros(len(texts), dtype=int),
        })

    counts = {"ner_count": [], "ner_money": [], "ner_date": [], "ner_gpe": [], "ner_org": []}
    for doc in nlp.pipe(texts, batch_size=512):
        per = {"MONEY": 0, "DATE": 0, "GPE": 0, "ORG": 0}
        total = 0
        for ent in doc.ents:
            total += 1
            if ent.label_ in per:
                per[ent.label_] += 1
        counts["ner_count"].append(total)
        counts["ner_money"].append(per["MONEY"])
        counts["ner_date"].append(per["DATE"])
        counts["ner_gpe"].append(per["GPE"])
        counts["ner_org"].append(per["ORG"])
    return pd.DataFrame(counts)


def _tfidf_features(texts: List[str], top_k: int = 50) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=top_k, ngram_range=(1, 2),
                          stop_words="english", lowercase=True)
    m = vec.fit_transform(texts)
    cols = [f"tfidf_{t}" for t in vec.get_feature_names_out()]
    df = pd.DataFrame(m.toarray(), columns=cols)
    return df


def _stratified_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def build(n_splits: int = 5, seed: int = 42, tfidf_top_k: int = 50) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = _load_hf()
    log.info("loaded %d rows, %d intents", len(df), df["intent_name"].nunique())

    # base lexical features (vectorized-ish — small enough that .apply is fine)
    log.info("computing lexical features…")
    lex = pd.DataFrame([_lex_features(s) for s in df["query"].tolist()])

    log.info("computing spaCy NER counts…")
    ner = _spacy_features(df["query"].tolist())

    log.info("computing tfidf top-%d…", tfidf_top_k)
    tfidf = _tfidf_features(df["query"].tolist(), top_k=tfidf_top_k)

    feats = pd.concat([
        df[["query", "intent_id", "intent_name", "source_split"]].reset_index(drop=True),
        lex.reset_index(drop=True),
        ner.reset_index(drop=True),
        tfidf.reset_index(drop=True),
    ], axis=1)
    feats.to_parquet(PARQUET_PATH, index=False)
    log.info("wrote %s (%d rows, %d cols)", PARQUET_PATH, *feats.shape)

    # stratified splits — store as object array of (train_idx, test_idx) pairs
    splits = _stratified_kfold(feats["intent_id"].to_numpy(), n_splits=n_splits, seed=seed)
    np.savez(SPLITS_PATH, **{
        f"fold_{i}_train": tr for i, (tr, _) in enumerate(splits)
    }, **{
        f"fold_{i}_test":  te for i, (_, te) in enumerate(splits)
    })
    log.info("wrote %s with %d folds", SPLITS_PATH, len(splits))

    # metadata for the prompt builder + reproduction
    feature_cols = [c for c in feats.columns if c not in ("query", "intent_id", "intent_name", "source_split")]
    meta = {
        "name": "banking77",
        "task": "classification",
        "n_classes": int(feats["intent_id"].nunique()),
        "n_rows": int(len(feats)),
        "n_features": int(len(feature_cols)),
        "target_column": "intent_id",
        "label_column": "intent_name",
        "feature_columns": feature_cols,
        "splits_seed": seed,
        "n_splits": n_splits,
        "tfidf_top_k": tfidf_top_k,
        "feature_descriptions": _feature_descriptions(feature_cols),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("wrote %s", META_PATH)


def _feature_descriptions(cols: List[str]) -> dict:
    canned = {
        "n_chars": "character count of the customer query",
        "n_tokens": "whitespace-tokenized word count",
        "avg_word_len": "mean token length in characters",
        "n_digits": "count of digit characters in the query",
        "n_punct": "count of punctuation characters",
        "n_uppercase": "count of uppercase characters",
        "has_question_mark": "1 if the query ends in or contains '?'",
        "has_money_symbol": "1 if a currency symbol or word appears",
        "has_number": "1 if any digit is present",
        "starts_with_qword": "1 if the first token is a question word",
        "qword_count": "count of question words anywhere in the query",
        "type_token_ratio": "unique tokens / total tokens (lex diversity proxy)",
        "ner_count": "total named-entity count (spaCy small model)",
        "ner_money": "MONEY-typed entity count",
        "ner_date":  "DATE-typed entity count",
        "ner_gpe":   "GPE (geo-political) entity count",
        "ner_org":   "ORG (organization) entity count",
    }
    out = {}
    for c in cols:
        if c in canned:
            out[c] = canned[c]
        elif c.startswith("tfidf_"):
            out[c] = f"tf-idf weight for term '{c[len('tfidf_'):]}'"
        else:
            out[c] = c
    return out


def load() -> tuple[pd.DataFrame, np.ndarray, list, dict]:
    """Load the pre-built features + splits. Raises if build() hasn't run."""
    for p in (PARQUET_PATH, SPLITS_PATH, META_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing — run `python -m evofeat.datasets.banking77 --build` first"
            )
    feats = pd.read_parquet(PARQUET_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    raw = np.load(SPLITS_PATH)
    n_splits = meta["n_splits"]
    splits = [(raw[f"fold_{i}_train"], raw[f"fold_{i}_test"]) for i in range(n_splits)]
    y = feats[meta["target_column"]].to_numpy()
    X = feats[meta["feature_columns"]].copy()
    return X, y, splits, meta


def _cli() -> None:
    p = argparse.ArgumentParser(description="Build the Banking77 feature parquet + splits.")
    p.add_argument("--build", action="store_true", help="build from HuggingFace")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tfidf-top-k", type=int, default=50)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    if args.build:
        build(n_splits=args.n_splits, seed=args.seed, tfidf_top_k=args.tfidf_top_k)
    else:
        p.print_help()
        sys.exit(0)


if __name__ == "__main__":
    _cli()
