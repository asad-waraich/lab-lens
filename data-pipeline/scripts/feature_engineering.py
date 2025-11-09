"""
Feature Engineering Pipeline for MIMIC-III Discharge Summaries
Author: Lab Lens Team
Description: Creates advanced features from preprocessed clinical text data
"""

import argparse
import logging
import re
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# Utility Functions

def find_repo_root(start: Path = Path.cwd()) -> Path:
    """
    Find the repository root by looking for data-pipeline directory
    
    Args:
        start: Starting directory for search
        
    Returns:
        Path to repository root
    """
    cur = start
    while cur != cur.parent:
        if (cur / "data-pipeline").exists() or (cur / "configs").exists():
            return cur
        cur = cur.parent
    return start


def setup_logger(log_path: Path = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logger with file and console handlers
    
    Args:
        log_path: Path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("feature_engineering")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def safe_len(x) -> int:
    """Safely get length of object, return 0 if not possible"""
    try:
        return len(x)
    except Exception:
        return 0


# Medical Term Lexicons for Feature Extraction

# Chronic disease terms for comorbidity detection
CHRONIC_DISEASE_TERMS = [
    "diabetes", "hypertension", "ckd", "chf", "cad", "copd", "asthma",
    "cirrhosis", "hepatitis", "hiv", "stroke", "afib", "atrial fibrillation",
    "hyperlipidemia", "hld", "hypothyroidism", "hyperthyroidism",
    "dementia", "alzheimer", "parkinson"
]

# Symptom terms for clinical presentation analysis
SYMPTOM_TERMS = [
    "fever", "chills", "nausea", "vomiting", "diarrhea", "dyspnea",
    "sob", "cough", "chest pain", "fatigue", "dizziness", "syncope",
    "edema", "pain", "headache", "rash"
]

# Medication terms for treatment complexity
MEDICATION_TERMS = [
    "insulin", "metformin", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "atorvastatin", "simvastatin", "warfarin", "heparin",
    "aspirin", "pantoprazole", "omeprazole", "gabapentin", "oxycodone",
    "duloxetine", "citalopram", "midodrine", "furosemide", "spironolactone",
    "lactulose", "thiamine", "folic acid", "acetaminophen", "naproxen"
]

# Common medication suffixes for drug identification
MED_SUFFIXES = [
    "pril", "sartan", "olol", "dipine", "statin", "azole", "tidine", "caine",
    "cycline", "cillin", "mycin", "sone"
]

# High-risk clinical terms for severity assessment
HIGH_RISK_TERMS = [
    "sepsis", "shock", "arrest", "failure", "critical", "icu",
    "intubat", "ventilat", "resuscitat", "unstable", "deteriorat",
    "code blue", "emergency", "urgent"
]

# Positive outcome terms for recovery tracking
POSITIVE_OUTCOME_TERMS = [
    "improv", "stable", "recover", "discharg", "ambulat", 
    "independent", "normal", "resolved", "healing", "better"
]

# Acute presentation indicators
ACUTE_TERMS = [
    "acute", "sudden", "rapid", "urgent", "emergency", "immediate", "new onset"
]

# Chronic condition indicators
CHRONIC_TERMS = [
    "chronic", "longstanding", "history of", "ongoing", "persistent", "long term"
]

# Section presence patterns for documentation quality
SECTION_PATTERNS = {
    "has_allergies_section": [r"\ballerg(y|ies)\b", r"\ballergies:\b"],
    "has_medications_section": [r"\bmedications?\b", r"\bdischarge medications?\b"],
    "has_brief_hospital_course": [r"\bbrief hospital course\b"],
}

# Negation tokens for sentiment analysis
NEGATION_TOKENS = [" no ", " denies ", " without ", " not ", " none "]


# Core Feature Extraction Functions

def sentence_count(text: str) -> int:
    """
    Count sentences in text using punctuation delimiters
    
    Args:
        text: Input text string
        
    Returns:
        Number of sentences
    """
    if not isinstance(text, str) or not text:
        return 0
    parts = re.split(r"[\.!?]+", text)
    return sum(1 for p in parts if p.strip())


def count_terms(text: str, terms: List[str]) -> int:
    """
    Count occurrences of specific terms in text
    
    Args:
        text: Input text string
        terms: List of terms to search for
        
    Returns:
        Total count of all terms found
    """
    if not isinstance(text, str) or not text:
        return 0
    tl = text.lower()
    total = 0
    for t in terms:
        if " " in t:
            # Multi-word phrases
            total += len(re.findall(re.escape(t), tl))
        else:
            # Single words with word boundaries
            total += len(re.findall(rf"\b{re.escape(t)}\b", tl))
    return total


def count_med_suffixes(text: str, suffixes: List[str]) -> int:
    """
    Count medication-like words based on common drug suffixes
    
    Args:
        text: Input text string
        suffixes: List of medication suffixes to match
        
    Returns:
        Count of words matching medication patterns
    """
    if not isinstance(text, str) or not text:
        return 0
    # Extract potential medication words
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return sum(any(tok.endswith(sfx) for sfx in suffixes) for tok in tokens)


def parse_icd_list(top_diagnoses: str) -> List[str]:
    """
    Parse comma-separated ICD codes into unique list
    
    Args:
        top_diagnoses: Comma-separated string of ICD codes
        
    Returns:
        Sorted list of unique ICD codes
    """
    if not isinstance(top_diagnoses, str) or not top_diagnoses.strip():
        return []
    parts = [p.strip() for p in top_diagnoses.split(",") if p.strip()]
    return sorted(set(parts))


def one_hot_topk(df: pd.DataFrame, col: str, k: int, prefix: str) -> pd.DataFrame:
    """
    Create one-hot encoding for top-k categories, group rest as OTHER
    
    Args:
        df: Input DataFrame
        col: Column name to encode
        k: Number of top categories to keep
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with one-hot encoded columns
    """
    if col not in df.columns:
        return df
    
    series = df[col].astype("string").fillna("UNKNOWN")
    topk = series.value_counts().index[:k]
    trimmed = series.where(series.isin(topk), "OTHER")
    dummies = pd.get_dummies(trimmed, prefix=prefix, dtype="int8")
    
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


# Readability and Text Quality Features

def calculate_readability_scores(text: str) -> Dict[str, float]:
    """
    Calculate text readability metrics including Flesch score
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with readability metrics
    """
    if not isinstance(text, str) or not text:
        return {
            'flesch_reading_ease': 0, 
            'avg_syllables_per_word': 0, 
            'vocabulary_richness': 0
        }
    
    # Count sentences, words, and syllables
    sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    words = text.split()
    word_count = len(words)
    
    if word_count == 0 or sentences == 0:
        return {
            'flesch_reading_ease': 0,
            'avg_syllables_per_word': 0,
            'vocabulary_richness': 0
        }
    
    # Simple syllable counting by vowel clusters
    syllables = sum(
        max(1, len(re.findall(r'[aeiouy]+', word.lower()))) 
        for word in words
    )
    
    # Flesch Reading Ease score: 0-100, higher is easier to read
    flesch = 206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllables / word_count)
    flesch = max(0, min(100, flesch))
    
    # Vocabulary richness: unique words / total words
    unique_words = len(set(word.lower() for word in words))
    vocabulary_richness = unique_words / word_count
    
    return {
        'flesch_reading_ease': round(flesch, 2),
        'avg_syllables_per_word': round(syllables / word_count, 2),
        'vocabulary_richness': round(vocabulary_richness, 3)
    }


# Documentation Quality Features

def calculate_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features based on missing clinical sections
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missingness features added
    """
    section_cols = [
        'discharge_diagnosis', 'discharge_medications', 
        'follow_up', 'chief_complaint', 'hospital_course'
    ]
    
    existing_sections = [col for col in section_cols if col in df.columns]
    
    if existing_sections:
        # Count missing sections per record
        missing_count = sum(
            df[col].isna() | (df[col] == '') 
            for col in existing_sections
        )
        
        df['missing_section_count'] = missing_count.astype('Int64')
        
        # Calculate completeness score (0-1)
        df['documentation_completeness'] = (
            (len(existing_sections) - missing_count) / len(existing_sections)
        ).clip(0, 1)
    else:
        df['missing_section_count'] = 0
        df['documentation_completeness'] = 1.0
    
    return df


# Clinical Risk and Severity Features

def calculate_clinical_risk_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Calculate clinical risk and outcome indicators from text
    
    Args:
        df: Input DataFrame
        text_col: Name of text column to analyze
        
    Returns:
        DataFrame with clinical risk features added
    """
    # High-risk term count
    df["high_risk_score"] = df[text_col].map(
        lambda s: count_terms(s, HIGH_RISK_TERMS)
    ).astype("Int64")
    
    # Positive outcome term count
    df["positive_outcome_score"] = df[text_col].map(
        lambda s: count_terms(s, POSITIVE_OUTCOME_TERMS)
    ).astype("Int64")
    
    # Risk to outcome ratio (higher = more concerning)
    df["risk_outcome_ratio"] = (
        df["high_risk_score"] / (df["positive_outcome_score"] + 1)
    ).clip(upper=10.0)
    
    # Acute presentation indicators
    df["acute_presentation_score"] = df[text_col].map(
        lambda s: count_terms(s, ACUTE_TERMS)
    ).astype("Int64")
    
    # Chronic condition indicators
    df["chronic_condition_score"] = df[text_col].map(
        lambda s: count_terms(s, CHRONIC_TERMS)
    ).astype("Int64")
    
    # Acute to chronic ratio (higher = more acute)
    df["acute_chronic_ratio"] = (
        df["acute_presentation_score"] / (df["chronic_condition_score"] + 1)
    ).clip(upper=5.0)
    
    return df


def calculate_medical_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate density of medical terms per sentence
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with medical density features added
    """
    denom = df["sentences"].replace({0: np.nan})
    
    # Disease mentions per sentence
    df["disease_density"] = (
        df["kw_chronic_disease"] / denom
    ).fillna(0).clip(upper=5.0)
    
    # Medication mentions per sentence
    df["medication_density"] = (
        df["kw_medications"] / denom
    ).fillna(0).clip(upper=5.0)
    
    # Symptom mentions per sentence
    df["symptom_density"] = (
        df["kw_symptoms"] / denom
    ).fillna(0).clip(upper=5.0)
    
    return df


def calculate_treatment_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate treatment complexity and intensity measures
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with treatment complexity features added
    """
    # Flag for polypharmacy (5+ medications)
    df["polypharmacy_flag"] = (df["kw_medications"] >= 5).astype("int8")
    
    # Overall treatment intensity score (weighted combination)
    med_score = df["kw_medications"] * 0.3
    lab_score = df.get("abnormal_lab_count", 0) * 0.4
    comorbid_score = df.get("comorbidity_score", 0) * 0.3
    
    df["treatment_intensity"] = (
        med_score + lab_score + comorbid_score
    ).fillna(0).clip(upper=20.0)
    
    return df


# Optional Advanced Features

def add_section_flags(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Add binary flags for presence of specific clinical sections
    
    Args:
        df: Input DataFrame
        text_col: Name of text column to analyze
        
    Returns:
        DataFrame with section flag features added
    """
    if text_col not in df.columns:
        return df
    
    tl = df[text_col].astype("string").str.lower().fillna("")
    
    # Check for each section pattern
    for flag, patterns in SECTION_PATTERNS.items():
        df[flag] = (
            tl.map(lambda s: int(any(re.search(p, s) is not None for p in patterns)))
            .astype("int8")
        )
    
    # Use existing word count if available
    text_tokens_col = 'word_count' if 'word_count' in df.columns else 'text_tokens'
    if text_tokens_col not in df.columns:
        df[text_tokens_col] = tl.str.split().map(safe_len).astype("Int64")
    
    # Count negation tokens
    def neg_count(s: str) -> int:
        if not isinstance(s, str) or not s:
            return 0
        s_pad = f" {s} "
        return sum(s_pad.count(tok) for tok in NEGATION_TOKENS)
    
    neg = tl.map(neg_count)
    denom = df[text_tokens_col].replace({0: np.nan})
    
    # Negation density: proportion of negation words
    df["negation_density"] = (neg / denom).fillna(0.0)
    
    return df


# Column Normalization

def normalize_column_names(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Map preprocessing output column names to expected feature engineering names
    
    Args:
        df: Input DataFrame
        logger: Logger instance
        
    Returns:
        DataFrame with normalized column names
    """
    column_mapping = {
        'text_length': 'text_chars',
        'word_count': 'text_tokens', 
        'abnormal_count': 'abnormal_lab_count',
    }
    
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
    if rename_dict:
        logger.info(f"Mapping columns: {rename_dict}")
        df = df.rename(columns=rename_dict)
    
    return df


# Main Feature Engineering Pipeline

def engineer_features(df: pd.DataFrame, logger: logging.Logger, with_sections: bool = False) -> pd.DataFrame:
    """
    Main feature engineering pipeline
    
    Args:
        df: Input DataFrame from preprocessing
        logger: Logger instance
        with_sections: Whether to include section-level features
        
    Returns:
        DataFrame with engineered features
    """
    if "cleaned_text" not in df.columns:
        raise ValueError("Missing column 'cleaned_text' in input dataset.")

    # Normalize column names from preprocessing
    df = normalize_column_names(df, logger)
    
    # Ensure base text metrics exist
    logger.info("Ensuring base text metrics...")
    if "text_chars" not in df.columns:
        df["text_chars"] = df["cleaned_text"].astype(str).str.len()
    if "text_tokens" not in df.columns:
        df["text_tokens"] = df["cleaned_text"].astype(str).str.split().map(safe_len)
    
    df["text_tokens"] = pd.to_numeric(df["text_tokens"], errors="coerce").astype("Int64")
    df["text_chars"] = pd.to_numeric(df["text_chars"], errors="coerce").astype("Int64")

    # Calculate sentence count
    logger.info("Computing sentence counts...")
    if "sentence_count" in df.columns:
        df["sentences"] = pd.to_numeric(df["sentence_count"], errors="coerce").astype("Int64")
    else:
        df["sentences"] = df["cleaned_text"].map(sentence_count).astype("Int64")

    # Calculate readability scores
    logger.info("Calculating readability scores...")
    readability = df["cleaned_text"].map(calculate_readability_scores)
    df["flesch_reading_ease"] = readability.map(lambda x: x['flesch_reading_ease'])
    df["avg_syllables_per_word"] = readability.map(lambda x: x['avg_syllables_per_word'])
    df["vocabulary_richness"] = readability.map(lambda x: x['vocabulary_richness'])

    # Count medical keyword families
    logger.info("Counting medical terms...")
    df["kw_chronic_disease"] = df["cleaned_text"].map(
        lambda s: count_terms(s, CHRONIC_DISEASE_TERMS)
    ).astype("Int64")
    
    df["kw_symptoms"] = df["cleaned_text"].map(
        lambda s: count_terms(s, SYMPTOM_TERMS)
    ).astype("Int64")
    
    df["kw_medications"] = df["cleaned_text"].map(
        lambda s: count_terms(s, MEDICATION_TERMS)
    ).astype("Int64")
    
    df["kw_med_suffix_hits"] = df["cleaned_text"].map(
        lambda s: count_med_suffixes(s, MED_SUFFIXES)
    ).astype("Int64")

    # Calculate clinical risk features
    logger.info("Calculating clinical risk features...")
    df = calculate_clinical_risk_features(df, "cleaned_text")

    # Calculate documentation quality
    logger.info("Calculating documentation quality...")
    df = calculate_missingness_features(df)

    # Process lab values and ratios
    logger.info("Deriving lab ratios...")
    if "total_labs" in df.columns and "abnormal_lab_count" in df.columns:
        denom = pd.to_numeric(df["total_labs"], errors="coerce").replace({0: np.nan})
        num = pd.to_numeric(df["abnormal_lab_count"], errors="coerce")
        df["abnormal_lab_ratio"] = (num / denom).clip(lower=0.0, upper=1.0).fillna(0.0)
        df["total_labs"] = denom.fillna(0).astype("Int64")
        df["abnormal_lab_count"] = num.fillna(0).astype("Int64")
    else:
        df["total_labs"] = pd.Series([0] * len(df), dtype="Int64")
        df["abnormal_lab_count"] = pd.Series([0] * len(df), dtype="Int64")
        df["abnormal_lab_ratio"] = 0.0

    # Process diagnosis information
    logger.info("Processing diagnosis information...")
    if "top_diagnoses" in df.columns:
        df["diagnosis_unique_count"] = df["top_diagnoses"].map(
            lambda s: len(parse_icd_list(s))
        ).astype("Int64")
    else:
        df["diagnosis_unique_count"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    
    if "diagnosis_count" in df.columns:
        df["comorbidity_score"] = pd.to_numeric(df["diagnosis_count"], errors="coerce").astype("Int64")
    else:
        df["comorbidity_score"] = df.get("diagnosis_unique_count", pd.Series([pd.NA] * len(df), dtype="Int64"))

    # Calculate medical density metrics
    logger.info("Calculating medical density...")
    df = calculate_medical_density(df)

    # Calculate treatment complexity
    logger.info("Calculating treatment complexity...")
    df = calculate_treatment_complexity(df)

    # Calculate normalized text metrics
    logger.info("Normalizing text metrics...")
    denom_tok = df["text_tokens"].replace({0: np.nan})
    df["chars_per_token"] = (df["text_chars"] / denom_tok).fillna(0.0)
    df["long_note_flag"] = (
        pd.to_numeric(df["text_tokens"], errors="coerce").fillna(0) >= 512
    ).astype("int16")

    # Add optional section flags
    if with_sections:
        logger.info("Adding section flags and negation density...")
        df = add_section_flags(df, text_col="cleaned_text")

    # One-hot encode demographic categories
    logger.info("One-hot encoding demographics...")
    for col, k, prefix in [
        ("gender", 2, "gender"),
        ("ethnicity_clean", 6, "eth"),
        ("insurance", 5, "ins"),
        ("admission_type", 4, "adm"),
        ("language", 4, "lang"),
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("UNKNOWN")
            df = one_hot_topk(df, col, k=k, prefix=prefix)

    # Order columns logically for output
    logger.info("Organizing output columns...")
    id_cols = [c for c in ["subject_id", "hadm_id"] if c in df.columns]
    
    base_features = [
        "text_chars", "text_tokens", "sentences", "chars_per_token", "long_note_flag",
        "flesch_reading_ease", "avg_syllables_per_word", "vocabulary_richness",
        "kw_chronic_disease", "kw_symptoms", "kw_medications", "kw_med_suffix_hits",
        "high_risk_score", "positive_outcome_score", "risk_outcome_ratio",
        "acute_presentation_score", "chronic_condition_score", "acute_chronic_ratio",
        "missing_section_count", "documentation_completeness",
        "disease_density", "medication_density", "symptom_density",
        "polypharmacy_flag", "treatment_intensity",
        "total_labs", "abnormal_lab_count", "abnormal_lab_ratio",
        "diagnosis_count", "diagnosis_unique_count", "comorbidity_score",
    ]
    
    optional = [
        c for c in ["has_allergies_section", "has_medications_section", 
                   "has_brief_hospital_course", "negation_density"] 
        if c in df.columns
    ]
    
    engineered = id_cols + [c for c in base_features if c in df.columns] + optional
    ohe_cols = [c for c in df.columns if c.startswith(("gender_", "eth_", "ins_", "adm_", "lang_"))]
    others = [c for c in df.columns if c not in engineered + ohe_cols + ["cleaned_text", "cleaned_text_final"]]
    
    ordered = engineered + others + ohe_cols
    df = df[[c for c in ordered if c in df.columns]].copy()

    logger.info(f"Feature engineering complete: {len(df.columns)} features created")
    return df


# Command Line Interface

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Feature engineering for MIMIC-III discharge summaries")
    p.add_argument("--input", type=str, help="Input CSV path (default: from config)")
    p.add_argument("--output", type=str, help="Output CSV path (default: from config)")
    p.add_argument("--log", type=str, help="Log file path (default: from config)")
    p.add_argument("--with_sections", action="store_true", help="Include section-level features")
    return p.parse_args()


def main() -> None:
    """Main execution function"""
    
    # Find repository root
    repo = find_repo_root()
    
    # Load configuration from pipeline
    config_path = repo / "data-pipeline" / "configs" / "pipeline_config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Use config paths
        default_input = repo / config['pipeline_config']['output_path'] / "processed_discharge_summaries.csv"
        default_output = repo / config['pipeline_config']['output_path'] / "mimic_features.csv"
        default_log = repo / config['pipeline_config']['logs_path'] / "feature_engineering.log"
    else:
        # Fallback paths if config not found
        default_input = repo / "data-pipeline" / "data" / "processed" / "processed_discharge_summaries.csv"
        default_output = repo / "data-pipeline" / "data" / "processed" / "mimic_features.csv"
        default_log = repo / "data-pipeline" / "logs" / "feature_engineering.log"

    # Parse command line arguments
    args = parse_args()
    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output
    log_path = Path(args.log) if args.log else default_log

    # Setup logger
    logger = setup_logger(log_path)
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*60)
    logger.info(f"Repository root: {repo}")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Section features: {args.with_sections}")

    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Run feature engineering
    df_features = engineer_features(df, logger=logger, with_sections=args.with_sections)

    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False, encoding="utf-8")
    
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total features: {len(df_features.columns)}")
    logger.info(f"Output saved: {output_path}")
    logger.info(f"Final shape: {df_features.shape[0]} rows, {df_features.shape[1]} columns")
    logger.info("="*60)
    
    # Print summary to console
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Records processed: {df_features.shape[0]}")
    print(f"Features created: {df_features.shape[1]}")
    print(f"Output file: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()