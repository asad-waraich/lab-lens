#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering for MIMIC-III Discharge Summaries (medium + optional bonus)
- Deterministic (pandas/regex only)
- Portable repo-root detection
- CLI configurable
- Optional section-aware micro-features: --with_sections
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ───────────────────────────── utils ─────────────────────────────

def find_repo_root(start: Path = Path.cwd()) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / "configs" / "data_config.yaml").exists():
            return cur
        cur = cur.parent
    return start

def setup_logger(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("feature_engineering")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


# ───────────────────── lexicons (medium) ─────────────────────

CHRONIC_DISEASE_TERMS = [
    "diabetes", "hypertension", "ckd", "chf", "cad", "copd", "asthma",
    "cirrhosis", "hepatitis", "hiv", "stroke", "afib", "atrial fibrillation",
    "hyperlipidemia", "hld", "hypothyroidism", "hyperthyroidism",
    "dementia", "alzheimer", "parkinson"
]
SYMPTOM_TERMS = [
    "fever", "chills", "nausea", "vomiting", "diarrhea", "dyspnea",
    "sob", "cough", "chest pain", "fatigue", "dizziness", "syncope",
    "edema", "pain", "headache", "rash"
]
MEDICATION_TERMS = [
    "insulin", "metformin", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "atorvastatin", "simvastatin", "warfarin", "heparin",
    "aspirin", "pantoprazole", "omeprazole", "gabapentin", "oxycodone",
    "duloxetine", "citalopram", "midodrine", "furosemide", "spironolactone",
    "lactulose", "thiamine", "folic acid", "acetaminophen", "naproxen"
]
MED_SUFFIXES = [
    "pril", "sartan", "olol", "dipine", "statin", "azole", "tidine", "caine",
    "cycline", "cillin", "mycin", "sone"
]


# ───────────────────── core feature helpers ─────────────────────

def sentence_count(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    parts = re.split(r"[\.!?]+", text)
    return sum(1 for p in parts if p.strip())

def count_terms(text: str, terms: List[str]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    tl = text.lower()
    total = 0
    for t in terms:
        if " " in t:
            total += len(re.findall(re.escape(t), tl))
        else:
            total += len(re.findall(rf"\b{re.escape(t)}\b", tl))
    return total

def count_med_suffixes(text: str, suffixes: List[str]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return sum(any(tok.endswith(sfx) for sfx in suffixes) for tok in tokens)

def parse_icd_list(top_diagnoses: str) -> list[str]:
    if not isinstance(top_diagnoses, str) or not top_diagnoses.strip():
        return []
    parts = [p.strip() for p in top_diagnoses.split(",") if p.strip()]
    return sorted(set(parts))

def one_hot_topk(df: pd.DataFrame, col: str, k: int, prefix: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    series = df[col].astype("string").fillna("UNKNOWN")
    topk = series.value_counts().index[:k]
    trimmed = series.where(series.isin(topk), "OTHER")
    dummies = pd.get_dummies(trimmed, prefix=prefix, dtype="int8")
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


# ─────────────── optional micro-features (sections/negations) ───────────────

SECTION_PATTERNS = {
    "has_allergies_section": [
        r"\ballerg(y|ies)\b", r"\ballergies:\b"
    ],
    "has_medications_section": [
        r"\bmedications?\b", r"\bdischarge medications?\b", r"\bmedications on admission\b"
    ],
    "has_brief_hospital_course": [
        r"\bbrief hospital course\b"
    ],
}
NEGATION_TOKENS = [" no ", " denies ", " without ", " not ", " none "]

def add_section_flags(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    if text_col not in df.columns:
        return df
    tl = df[text_col].astype("string").str.lower().fillna("")
    for flag, patterns in SECTION_PATTERNS.items():
        df[flag] = (
            tl.map(lambda s: int(any(re.search(p, s) is not None for p in patterns)))
            .astype("int8")
        )
    # negation density = (#negation tokens) / tokens
    if "text_tokens" not in df.columns:
        df["text_tokens"] = tl.str.split().map(safe_len).astype("Int64")
    def neg_count(s: str) -> int:
        if not isinstance(s, str) or not s:
            return 0
        s_pad = f" {s} "
        return sum(s_pad.count(tok) for tok in NEGATION_TOKENS)
    neg = tl.map(neg_count)
    denom = df["text_tokens"].replace({0: np.nan})
    df["negation_density"] = (neg / denom).fillna(0.0)
    return df


# ───────────────────── pipeline (engineer_features) ─────────────────────

def engineer_features(df: pd.DataFrame, logger: logging.Logger, with_sections: bool = False) -> pd.DataFrame:
    if "cleaned_text" not in df.columns:
        raise ValueError("Missing column 'cleaned_text' in input dataset.")

    # Ensure base text metrics exist
    if "text_chars" not in df.columns:
        df["text_chars"] = df["cleaned_text"].astype(str).str.len()
    if "text_tokens" not in df.columns:
        df["text_tokens"] = df["cleaned_text"].astype(str).str.split().map(safe_len)
    df["text_tokens"] = pd.to_numeric(df["text_tokens"], errors="coerce").astype("Int64")
    df["text_chars"] = pd.to_numeric(df["text_chars"], errors="coerce").astype("Int64")

    # Sentences
    logger.info("Computing sentence_count...")
    df["sentences"] = df["cleaned_text"].map(sentence_count).astype("Int64")

    # Keyword families
    logger.info("Counting disease/symptom/medication terms...")
    df["kw_chronic_disease"] = df["cleaned_text"].map(lambda s: count_terms(s, CHRONIC_DISEASE_TERMS)).astype("Int64")
    df["kw_symptoms"]        = df["cleaned_text"].map(lambda s: count_terms(s, SYMPTOM_TERMS)).astype("Int64")
    df["kw_medications"]     = df["cleaned_text"].map(lambda s: count_terms(s, MEDICATION_TERMS)).astype("Int64")
    df["kw_med_suffix_hits"] = df["cleaned_text"].map(lambda s: count_med_suffixes(s, MED_SUFFIXES)).astype("Int64")

    # Structured clinical severity
    logger.info("Deriving severity ratios...")
    if "total_labs" in df.columns and "abnormal_lab_count" in df.columns:
        denom = pd.to_numeric(df["total_labs"], errors="coerce").replace({0: np.nan})
        num   = pd.to_numeric(df["abnormal_lab_count"], errors="coerce")
        df["abnormal_lab_ratio"] = (num / denom).clip(lower=0.0, upper=1.0).fillna(0.0)
        df["total_labs"] = denom.fillna(0).astype("Int64")
        df["abnormal_lab_count"] = num.fillna(0).astype("Int64")
    else:
        df["abnormal_lab_ratio"] = 0.0

    # Diagnosis derived
    if "top_diagnoses" in df.columns:
        df["diagnosis_unique_count"] = df["top_diagnoses"].map(lambda s: len(parse_icd_list(s))).astype("Int64")
    else:
        df["diagnosis_unique_count"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "diagnosis_count" in df.columns:
        df["comorbidity_score"] = pd.to_numeric(df["diagnosis_count"], errors="coerce").astype("Int64")
    else:
        df["comorbidity_score"] = df["diagnosis_unique_count"]

    # Normalized text metrics
    logger.info("Normalizing text metrics...")
    denom_tok = df["text_tokens"].replace({0: np.nan})
    df["chars_per_token"] = (df["text_chars"] / denom_tok).fillna(0.0)
    df["long_note_flag"]  = (pd.to_numeric(df["text_tokens"], errors="coerce").fillna(0) >= 512).astype("int16")

    # Optional section flags + negation density
    if with_sections:
        logger.info("Adding section flags and negation density...")
        df = add_section_flags(df, text_col="cleaned_text")

    # One-hot encodings (top-k)
    logger.info("One-hot encoding selected demographics (top-k)...")
    for col, k, prefix in [
        ("gender", 2, "gender"),
        ("ethnicity", 6, "eth"),
        ("insurance", 5, "ins"),
        ("admission_type", 4, "adm"),
        ("language", 4, "lang"),
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("UNKNOWN")
            df = one_hot_topk(df, col, k=k, prefix=prefix)

    # Final ordering
    id_cols = [c for c in ["subject_id", "hadm_id"] if c in df.columns]
    base = [
        "text_chars", "text_tokens", "sentences", "chars_per_token", "long_note_flag",
        "kw_chronic_disease", "kw_symptoms", "kw_medications", "kw_med_suffix_hits",
        "total_labs", "abnormal_lab_count", "abnormal_lab_ratio",
        "diagnosis_count", "diagnosis_unique_count", "comorbidity_score",
    ]
    optional = [c for c in ["has_allergies_section", "has_medications_section", "has_brief_hospital_course", "negation_density"] if c in df.columns]
    # keep OHEs at end
    engineered = id_cols + [c for c in base if c in df.columns] + optional
    ohe_cols = [c for c in df.columns if c.startswith(("gender_", "eth_", "ins_", "adm_", "lang_"))]
    others = [c for c in df.columns if c not in engineered + ohe_cols + ["cleaned_text"]]  # drop text from feature table
    ordered = engineered + others + ohe_cols
    df = df[ordered].copy()

    logger.info("Feature engineering complete.")
    return df


# ───────────────────────────── CLI ─────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature engineering for MIMIC-III discharge summaries.")
    p.add_argument("--input", type=str, help="Preprocessed CSV (default: <repo>/data/processed/mimic_cleaned.csv)")
    p.add_argument("--output", type=str, help="Features CSV (default: <repo>/data/processed/mimic_features.csv)")
    p.add_argument("--log", type=str, help="Log file (default: <repo>/logs/feature_engineering.log)")
    p.add_argument("--with_sections", action="store_true", help="Add section flags + negation_density features")
    return p.parse_args()

def main() -> None:
    repo = find_repo_root()
    default_input  = repo / "data" / "processed" / "mimic_cleaned.csv"
    default_output = repo / "data" / "processed" / "mimic_features.csv"
    default_log    = repo / "logs" / "feature_engineering.log"

    args = parse_args()
    input_path  = Path(args.input)  if args.input  else default_input
    output_path = Path(args.output) if args.output else default_output
    log_path    = Path(args.log)    if args.log    else default_log

    logger = setup_logger(log_path)
    logger.info(f"Repo root: {repo}")
    logger.info(f"Input:     {input_path}")
    logger.info(f"Output:    {output_path}")
    logger.info(f"Log file:  {log_path}")
    logger.info(f"With sections: {args.with_sections}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8", dtype_backend="numpy_nullable")
    logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    df_feat = engineer_features(df, logger=logger, with_sections=args.with_sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=1,          # QUOTE_ALL
        lineterminator="\n",
        escapechar="\\",
    )
    logger.info(f"Saved features → {output_path.resolve()}")
    logger.info(f"Final shape: {df_feat.shape[0]} rows, {df_feat.shape[1]} cols")


if __name__ == "__main__":
    main()