"""
Unit Tests for Feature Engineering Pipeline
Tests readability scores, clinical features, medical density, and treatment complexity
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, scripts_path)

from feature_engineering import (
    calculate_readability_scores,
    count_terms,
    count_med_suffixes,
    parse_icd_list,
    calculate_clinical_risk_features,
    calculate_medical_density,
    calculate_treatment_complexity,
    calculate_missingness_features
)


# Fixtures

@pytest.fixture
def sample_clinical_text():
    """Sample clinical text for testing"""
    return """
    Patient admitted with acute myocardial infarction. History of diabetes and hypertension.
    Started on aspirin, metoprolol, and lisinopril. Patient showed improvement during hospital stay.
    Discharged in stable condition with follow-up scheduled.
    """


@pytest.fixture
def sample_feature_data():
    """Create sample data for feature engineering"""
    data = {
        'hadm_id': [1001, 1002, 1003, 1004],
        'subject_id': [101, 102, 103, 104],
        'cleaned_text': [
            'Patient with sepsis and shock. Critical condition requiring ICU admission.',
            'Patient improved and stable. Discharged home in good condition.',
            'Chronic diabetes and hypertension. Long-standing history of heart failure.',
            'Acute chest pain. Emergency presentation requiring immediate intervention.'
        ],
        'text_length': [500, 450, 480, 470],
        'word_count': [75, 70, 72, 71],
        'sentence_count': [8, 7, 7, 6],
        'discharge_diagnosis': ['sepsis, shock', 'resolved', '', 'chest pain'],
        'discharge_medications': ['multiple medications listed here', 'aspirin', '', 'aspirin, lisinopril'],
        'follow_up': ['ICU follow-up', 'routine', '', 'cardiology'],
        'total_labs': [15, 10, 8, 12],
        'abnormal_lab_count': [8, 2, 1, 5],
        'diagnosis_count': [5, 2, 3, 2],
        'top_diagnoses': ['995.92, 785.52, 584.9', '414.01', '250.00, 401.9, 428.0', '786.50, 414.01']
    }
    
    return pd.DataFrame(data)


# Readability Score Tests

def test_calculate_readability_scores_normal_text(sample_clinical_text):
    """Test readability score calculation with normal text"""
    scores = calculate_readability_scores(sample_clinical_text)
    
    # Check all expected keys exist
    assert 'flesch_reading_ease' in scores
    assert 'avg_syllables_per_word' in scores
    assert 'vocabulary_richness' in scores
    
    # Check values are in reasonable ranges
    assert 0 <= scores['flesch_reading_ease'] <= 100
    assert scores['avg_syllables_per_word'] > 0
    assert 0 <= scores['vocabulary_richness'] <= 1


def test_calculate_readability_scores_empty_text():
    """Test readability scores with empty or None text"""
    # Test empty string
    scores = calculate_readability_scores('')
    assert scores['flesch_reading_ease'] == 0
    assert scores['avg_syllables_per_word'] == 0
    assert scores['vocabulary_richness'] == 0
    
    # Test None
    scores = calculate_readability_scores(None)
    assert scores['flesch_reading_ease'] == 0


def test_calculate_readability_scores_short_text():
    """Test readability with very short text"""
    short_text = "Patient stable."
    scores = calculate_readability_scores(short_text)
    
    # Should handle short text without errors
    assert isinstance(scores['flesch_reading_ease'], (int, float))
    assert scores['flesch_reading_ease'] >= 0


# Medical Term Counting Tests

def test_count_terms_single_words():
    """Test counting single-word medical terms"""
    text = "Patient has diabetes and hypertension with asthma."
    terms = ['diabetes', 'hypertension', 'asthma']
    
    count = count_terms(text, terms)
    
    # Should find all 3 terms
    assert count == 3


def test_count_terms_multi_word_phrases():
    """Test counting multi-word medical phrases"""
    text = "Patient diagnosed with congestive heart failure and myocardial infarction."
    terms = ['congestive heart failure', 'myocardial infarction']
    
    count = count_terms(text, terms)
    
    # Should find both multi-word terms
    assert count == 2


def test_count_terms_case_insensitive():
    """Test that term counting is case insensitive"""
    text = "Patient has DIABETES and Hypertension and diabetes."
    terms = ['diabetes', 'hypertension']
    
    count = count_terms(text, terms)
    
    # Should find diabetes twice (case insensitive)
    assert count == 3


def test_count_terms_empty_text():
    """Test term counting with empty or None text"""
    assert count_terms('', ['diabetes']) == 0
    assert count_terms(None, ['diabetes']) == 0


def test_count_med_suffixes():
    """Test medication suffix detection"""
    text = "Patient on lisinopril, metoprolol, and atorvastatin."
    suffixes = ['pril', 'olol', 'statin']
    
    count = count_med_suffixes(text, suffixes)
    
    # Should find all 3 medication-like words
    assert count == 3


def test_count_med_suffixes_empty():
    """Test medication suffix counting with empty text"""
    assert count_med_suffixes('', ['pril']) == 0
    assert count_med_suffixes(None, ['pril']) == 0


# ICD Code Parsing Tests

def test_parse_icd_list_normal():
    """Test parsing comma-separated ICD codes"""
    icd_string = "995.92, 785.52, 584.9, 995.92"  # Has duplicate
    
    codes = parse_icd_list(icd_string)
    
    # Should return sorted unique list
    assert isinstance(codes, list)
    assert len(codes) == 3  # Duplicate removed
    assert '995.92' in codes


def test_parse_icd_list_empty():
    """Test ICD parsing with empty or None input"""
    assert parse_icd_list('') == []
    assert parse_icd_list(None) == []
    assert parse_icd_list('   ') == []


# Clinical Risk Feature Tests

def test_calculate_clinical_risk_features(sample_feature_data):
    """Test clinical risk feature calculation"""
    df_with_risk = calculate_clinical_risk_features(sample_feature_data, 'cleaned_text')
    
    # Check new columns created
    assert 'high_risk_score' in df_with_risk.columns
    assert 'positive_outcome_score' in df_with_risk.columns
    assert 'risk_outcome_ratio' in df_with_risk.columns
    assert 'acute_presentation_score' in df_with_risk.columns
    assert 'chronic_condition_score' in df_with_risk.columns
    assert 'acute_chronic_ratio' in df_with_risk.columns
    
    # First record has high-risk terms (sepsis, shock, critical, ICU)
    assert df_with_risk.iloc[0]['high_risk_score'] > 0
    
    # Second record has positive outcome terms (improved, stable, discharged)
    assert df_with_risk.iloc[1]['positive_outcome_score'] > 0
    
    # Third record has chronic terms (chronic, long-standing, history)
    assert df_with_risk.iloc[2]['chronic_condition_score'] > 0
    
    # Fourth record has acute terms (acute, emergency, immediate)
    assert df_with_risk.iloc[3]['acute_presentation_score'] > 0


def test_risk_outcome_ratio_clipping(sample_feature_data):
    """Test that risk-outcome ratio is properly clipped"""
    df_with_risk = calculate_clinical_risk_features(sample_feature_data, 'cleaned_text')
    
    # Ratio should be clipped at 10.0
    assert df_with_risk['risk_outcome_ratio'].max() <= 10.0
    assert df_with_risk['risk_outcome_ratio'].min() >= 0


# Medical Density Tests

def test_calculate_medical_density(sample_feature_data):
    """Test medical density calculation"""
    # Add required kw columns
    sample_feature_data['kw_chronic_disease'] = [1, 0, 3, 1]
    sample_feature_data['kw_medications'] = [2, 1, 0, 2]
    sample_feature_data['kw_symptoms'] = [1, 0, 0, 1]
    sample_feature_data['sentences'] = [8, 7, 7, 6]
    
    df_with_density = calculate_medical_density(sample_feature_data)
    
    # Check new columns
    assert 'disease_density' in df_with_density.columns
    assert 'medication_density' in df_with_density.columns
    assert 'symptom_density' in df_with_density.columns
    
    # Density should be non-negative
    assert (df_with_density['disease_density'] >= 0).all()
    assert (df_with_density['medication_density'] >= 0).all()
    
    # Density should be clipped at 5.0
    assert df_with_density['disease_density'].max() <= 5.0


def test_medical_density_with_zero_sentences(sample_feature_data):
    """Test medical density when sentence count is zero"""
    sample_feature_data['kw_chronic_disease'] = [1, 0, 3, 1]
    sample_feature_data['kw_medications'] = [2, 1, 0, 2]
    sample_feature_data['kw_symptoms'] = [1, 0, 0, 1]
    sample_feature_data['sentences'] = [0, 7, 7, 6]  # First row has 0 sentences
    
    df_with_density = calculate_medical_density(sample_feature_data)
    
    # Should handle division by zero gracefully (fillna with 0)
    assert df_with_density.iloc[0]['disease_density'] == 0
    assert not np.isnan(df_with_density['disease_density']).any()


# Treatment Complexity Tests

def test_calculate_treatment_complexity(sample_feature_data):
    """Test treatment complexity feature calculation"""
    # Add required columns
    sample_feature_data['kw_medications'] = [8, 3, 2, 5]  # First has polypharmacy
    sample_feature_data['abnormal_lab_count'] = [8, 2, 1, 5]
    sample_feature_data['comorbidity_score'] = [5, 2, 3, 2]
    
    df_with_complexity = calculate_treatment_complexity(sample_feature_data)
    
    # Check new columns
    assert 'polypharmacy_flag' in df_with_complexity.columns
    assert 'treatment_intensity' in df_with_complexity.columns
    
    # First record should be flagged for polypharmacy (8 medications >= 5)
    assert df_with_complexity.iloc[0]['polypharmacy_flag'] == 1
    
    # Third record should not be flagged (2 medications < 5)
    assert df_with_complexity.iloc[2]['polypharmacy_flag'] == 0
    
    # Treatment intensity should be non-negative and clipped
    assert (df_with_complexity['treatment_intensity'] >= 0).all()
    assert df_with_complexity['treatment_intensity'].max() <= 20.0


# Documentation Quality Tests

def test_calculate_missingness_features(sample_feature_data):
    """Test documentation missingness feature calculation"""
    df_with_miss = calculate_missingness_features(sample_feature_data)
    
    # Check new columns
    assert 'missing_section_count' in df_with_miss.columns
    assert 'documentation_completeness' in df_with_miss.columns
    
    # Completeness should be between 0 and 1
    assert (df_with_miss['documentation_completeness'] >= 0).all()
    assert (df_with_miss['documentation_completeness'] <= 1).all()
    
    # Record with all sections should have high completeness
    complete_record = df_with_miss.iloc[0]  # Has all sections
    assert complete_record['documentation_completeness'] > 0
    
    # Record with missing sections should have lower completeness
    incomplete_record = df_with_miss.iloc[2]  # Has empty sections
    assert incomplete_record['documentation_completeness'] < 1


def test_missing_section_count_accuracy(sample_feature_data):
    """Test that missing section count is calculated correctly"""
    df_with_miss = calculate_missingness_features(sample_feature_data)
    
    # Third record has empty discharge_diagnosis, discharge_medications, follow_up = 3 missing
    third_record_missing = df_with_miss.iloc[2]['missing_section_count']
    assert third_record_missing >= 3


# Edge Case Tests

def test_feature_engineering_with_all_nulls():
    """Test feature engineering with completely null text"""
    null_data = pd.DataFrame({
        'hadm_id': [1001],
        'subject_id': [101],
        'cleaned_text': [None],
        'text_length': [0],
        'word_count': [0]
    })
    
    # Should handle without crashing
    scores = calculate_readability_scores(None)
    assert scores['flesch_reading_ease'] == 0


def test_feature_engineering_with_special_characters():
    """Test handling of special characters in medical text"""
    special_text = "Patient's BP=120/80, HR=75bpm, Temp=98.6°F, O2=95%"
    
    scores = calculate_readability_scores(special_text)
    
    # Should handle special characters
    assert scores is not None
    assert isinstance(scores['flesch_reading_ease'], (int, float))


def test_icd_code_parsing_with_spaces():
    """Test ICD code parsing with inconsistent spacing"""
    icd_string = " 995.92 , 785.52,  584.9  ,  995.92 "
    
    codes = parse_icd_list(icd_string)
    
    # Should handle spaces and duplicates
    assert len(codes) == 3
    assert '995.92' in codes


def test_term_counting_with_overlapping_terms():
    """Test that overlapping terms are counted correctly"""
    text = "Patient with heart failure and congestive heart failure"
    terms = ['heart failure', 'congestive heart failure']
    
    count = count_terms(text, terms)
    
    # Should count both occurrences
    assert count >= 2


# Integration Tests

def test_multiple_feature_calculations_together(sample_feature_data):
    """Test that multiple feature calculations work together"""
    # Add required columns
    sample_feature_data['kw_chronic_disease'] = [1, 0, 3, 1]
    sample_feature_data['kw_medications'] = [6, 3, 2, 5]
    sample_feature_data['kw_symptoms'] = [2, 1, 0, 1]
    sample_feature_data['sentences'] = [8, 7, 7, 6]
    sample_feature_data['comorbidity_score'] = [5, 2, 3, 2]
    
    # Apply multiple transformations
    df = sample_feature_data.copy()
    df = calculate_clinical_risk_features(df, 'cleaned_text')
    df = calculate_medical_density(df)
    df = calculate_treatment_complexity(df)
    df = calculate_missingness_features(df)
    
    # Verify all feature columns exist
    expected_columns = [
        'high_risk_score', 'positive_outcome_score', 'risk_outcome_ratio',
        'disease_density', 'medication_density', 'symptom_density',
        'polypharmacy_flag', 'treatment_intensity',
        'missing_section_count', 'documentation_completeness'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"


def test_feature_dtypes_are_correct(sample_feature_data):
    """Test that feature data types are appropriate"""
    # Add required columns
    sample_feature_data['kw_medications'] = [6, 3, 2, 5]
    sample_feature_data['abnormal_lab_count'] = [8, 2, 1, 5]
    sample_feature_data['comorbidity_score'] = [5, 2, 3, 2]
    
    df = calculate_treatment_complexity(sample_feature_data)
    
    # Polypharmacy flag should be integer (0 or 1)
    assert df['polypharmacy_flag'].dtype in [np.int8, np.int16, np.int32, np.int64, int]
    
    # Treatment intensity should be float
    assert df['treatment_intensity'].dtype in [np.float64, np.float32, float]


# Boundary Tests

def test_flesch_score_boundaries():
    """Test that Flesch score stays within 0-100 bounds"""
    # Very simple text (should have high score)
    simple_text = "Cat sat. Dog ran. Sun set."
    scores = calculate_readability_scores(simple_text)
    assert 0 <= scores['flesch_reading_ease'] <= 100
    
    # Complex medical text (should have lower score)
    complex_text = "The patient presented with decompensated congestive heart failure exacerbation."
    scores = calculate_readability_scores(complex_text)
    assert 0 <= scores['flesch_reading_ease'] <= 100


def test_vocabulary_richness_bounds():
    """Test that vocabulary richness is between 0 and 1"""
    # All same word (low richness)
    repetitive = "test test test test test"
    scores = calculate_readability_scores(repetitive)
    assert 0 <= scores['vocabulary_richness'] <= 1
    
    # All unique words (high richness)
    unique = "cat dog bird fish mouse"
    scores = calculate_readability_scores(unique)
    assert scores['vocabulary_richness'] == 1.0


# Performance Tests

def test_feature_engineering_performance_large_dataset():
    """Test that feature engineering handles large datasets efficiently"""
    # Create larger dataset
    large_data = pd.DataFrame({
        'cleaned_text': ['Patient with diabetes and hypertension.'] * 1000,
        'sentences': [5] * 1000,
        'kw_chronic_disease': [2] * 1000,
        'kw_medications': [3] * 1000,
        'kw_symptoms': [1] * 1000
    })
    
    import time
    start = time.time()
    
    df_density = calculate_medical_density(large_data)
    
    duration = time.time() - start
    
    # Should complete in reasonable time (< 5 seconds for 1000 records)
    assert duration < 5.0
    assert len(df_density) == 1000


# Run tests with: pytest tests/test_feature_engineering.py -v