"""
Unit Tests for MIMIC Preprocessing Pipeline
Tests data loading, duplicate removal, demographic standardization, and feature creation
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, scripts_path)

from preprocessing import MIMICPreprocessor

# Fixtures provide reusable test data

@pytest.fixture
def sample_data():
    """Create sample MIMIC data for testing"""
    data = {
        'hadm_id': [1001, 1002, 1003, 1003, 1004, 1005],
        'subject_id': [101, 102, 103, 103, 104, 105],
        'cleaned_text': [
            'Chief Complaint: Chest pain\nDischarge Diagnosis: MI\nDischarge Medications: Aspirin\nFollow-up: Cardiology',
            'Chief Complaint: Shortness of breath\nDischarge Diagnosis: CHF\nDischarge Medications: Lasix\nFollow-up: PCP',
            'Chief Complaint: Abdominal pain\nDischarge Diagnosis: Appendicitis\nDischarge Medications: Antibiotics\nFollow-up: Surgery',
            'Chief Complaint: Abdominal pain\nDischarge Diagnosis: Appendicitis\nDischarge Medications: Antibiotics\nFollow-up: Surgery',
            'Chief Complaint: Falls\nDischarge Diagnosis: Hip fracture\nDischarge Medications: Pain meds\nFollow-up: Ortho',
            'Discharge Diagnosis: Observation\nDischarge Medications: None\nFollow-up: None'
        ],
        'ethnicity': ['WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC OR LATINO', 
                     'HISPANIC OR LATINO', 'ASIAN - CHINESE', 'WHITE - RUSSIAN'],
        'age_at_admission': [45, 67, 55, 55, 82, 35],
        'gender': ['M', 'F', 'M', 'M', 'F', 'M'],
        'lab_summary': ['labs', 'labs', 'labs', 'labs', 'labs', 'labs']
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create preprocessor instance for testing"""
    return MIMICPreprocessor(
        input_path='tests/data',
        output_path='tests/output'
    )


# Test Data Loading

def test_remove_duplicates(preprocessor, sample_data):
    """Test that duplicate records are properly removed"""
    # Initial: 6 records with 1 duplicate
    assert len(sample_data) == 6
    
    # Remove duplicates
    cleaned = preprocessor.remove_duplicates(sample_data)
    
    # Should have 5 unique records (1003 appears twice)
    assert len(cleaned) == 5
    
    # Verify no duplicate hadm_ids remain
    assert cleaned['hadm_id'].duplicated().sum() == 0


def test_standardize_ethnicity(preprocessor):
    """Test ethnicity standardization to 5 categories"""
    # Test various MIMIC ethnicity strings
    test_cases = {
        'WHITE': 'WHITE',
        'WHITE - RUSSIAN': 'WHITE',
        'WHITE - OTHER EUROPEAN': 'WHITE',
        'BLACK/AFRICAN AMERICAN': 'BLACK',
        'BLACK/CAPE VERDEAN': 'BLACK',
        'HISPANIC OR LATINO': 'HISPANIC',
        'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
        'ASIAN': 'ASIAN',
        'ASIAN - CHINESE': 'ASIAN',
        'UNKNOWN': 'OTHER',
        None: 'UNKNOWN',
        'PATIENT DECLINED TO ANSWER': 'OTHER'
    }
    
    for input_eth, expected_output in test_cases.items():
        result = preprocessor.standardize_ethnicity(input_eth)
        assert result == expected_output, f"Failed for {input_eth}"


def test_create_demographic_features(preprocessor, sample_data):
    """Test demographic feature creation including age groups"""
    # Create demographic features
    df_with_demo = preprocessor.create_demographic_features(sample_data)
    
    # Check new columns exist
    assert 'ethnicity_clean' in df_with_demo.columns
    assert 'age_group' in df_with_demo.columns
    assert 'gender' in df_with_demo.columns
    
    # Verify ethnicity standardization
    assert df_with_demo['ethnicity_clean'].nunique() <= 5
    assert 'WHITE' in df_with_demo['ethnicity_clean'].values
    assert 'BLACK' in df_with_demo['ethnicity_clean'].values
    
    # Verify age grouping
    age_groups = df_with_demo['age_group'].dropna().unique()
    expected_groups = ['<18', '18-35', '35-50', '50-65', '65+']
    for group in age_groups:
        assert str(group) in expected_groups
    
    # Check specific age mappings
    age_35_row = df_with_demo[df_with_demo['age_at_admission'] == 35]
    assert str(age_35_row['age_group'].iloc[0]) == '18-35'
    
    age_67_row = df_with_demo[df_with_demo['age_at_admission'] == 67]
    assert str(age_67_row['age_group'].iloc[0]) == '65+'


def test_expand_abbreviations(preprocessor):
    """Test medical abbreviation expansion"""
    # Test common abbreviations
    test_text = "Pt admitted with CHF and DM. S/P cardiac procedure. W/ no complications."
    
    expanded = preprocessor.expand_abbreviations(test_text)
    
    # Check expansions
    assert 'patient' in expanded.lower()
    assert 'congestive heart failure' in expanded.lower()
    assert 'diabetes mellitus' in expanded.lower()
    assert 'status post' in expanded.lower()
    assert 'with' in expanded.lower()
    
    # Test None handling
    assert preprocessor.expand_abbreviations(None) is None
    
    # Test empty string
    assert preprocessor.expand_abbreviations('') == ''


def test_clean_text(preprocessor):
    """Test text cleaning functionality"""
    # Test extra whitespace removal
    dirty_text = "Patient   admitted    with multiple    spaces"
    clean = preprocessor.clean_text(dirty_text)
    assert '  ' not in clean  # No double spaces
    
    # Test repeated punctuation removal
    dirty_text = "Patient stable!!!!"
    clean = preprocessor.clean_text(dirty_text)
    assert '!!!!' not in clean
    assert '!' in clean
    
    # Test None handling
    assert preprocessor.clean_text(None) == ""
    
    # Test empty string
    assert preprocessor.clean_text('') == ""


def test_extract_sections(preprocessor):
    """Test section extraction from discharge summaries"""
    # Sample discharge text with sections
    discharge_text = """
    Chief Complaint: Chest pain
    
    History of Present Illness: Patient presented with acute chest pain.
    
    Discharge Diagnosis: Myocardial infarction
    
    Discharge Medications: Aspirin 81mg daily, Lisinopril 10mg daily
    
    Follow-up: Cardiology in 2 weeks
    """
    
    sections = preprocessor.extract_sections(discharge_text)
    
    # Verify sections were extracted
    assert 'chief_complaint' in sections
    assert 'chest pain' in sections['chief_complaint'].lower()
    
    assert 'discharge_diagnosis' in sections
    assert 'myocardial infarction' in sections['discharge_diagnosis'].lower()
    
    assert 'discharge_medications' in sections
    assert 'aspirin' in sections['discharge_medications'].lower()
    
    assert 'follow_up' in sections
    assert 'cardiology' in sections['follow_up'].lower()
    
    # Test None handling
    empty_sections = preprocessor.extract_sections(None)
    assert all(v == '' for v in empty_sections.values())


def test_calculate_text_features(preprocessor, sample_data):
    """Test text feature calculation"""
    # Add required columns for feature calculation
    sample_data['discharge_medications'] = ['med1', '', 'med2', 'med2', '', '']
    sample_data['follow_up'] = ['followup', '', '', '', 'followup', '']
    sample_data['discharge_diagnosis'] = ['dx1', 'dx2', '', '', 'dx3', '']
    
    df_features = preprocessor.calculate_text_features(sample_data)
    
    # Check new columns exist
    assert 'text_length' in df_features.columns
    assert 'word_count' in df_features.columns
    assert 'sentence_count' in df_features.columns
    assert 'medical_term_count' in df_features.columns
    assert 'has_medications' in df_features.columns
    assert 'has_follow_up' in df_features.columns
    
    # Verify calculations are reasonable
    assert df_features['text_length'].min() >= 0
    assert df_features['word_count'].min() >= 0
    
    # Test boolean flags
    assert df_features['has_medications'].dtype == bool or df_features['has_medications'].dtype == np.bool_


def test_handle_missing_values(preprocessor, sample_data):
    """Test missing value handling"""
    # Create data with missing values
    data_with_nulls = sample_data.copy()
    data_with_nulls.loc[0, 'cleaned_text'] = None
    data_with_nulls.loc[1, 'text_length'] = None
    
    # Add optional columns
    data_with_nulls['discharge_diagnosis'] = [None, 'dx', None, 'dx', None, '']
    data_with_nulls['abnormal_count'] = [None, 5, None, 2, None, 0]
    
    df_handled = preprocessor.handle_missing_values(data_with_nulls)
    
    # Verify text fields filled with empty string
    assert df_handled['cleaned_text'].isna().sum() == 0
    assert df_handled['discharge_diagnosis'].isna().sum() == 0
    
    # Verify numeric fields handled appropriately
    assert df_handled['abnormal_count'].isna().sum() == 0


def test_full_preprocessing_pipeline(preprocessor, sample_data, tmp_path):
    """Test complete preprocessing pipeline end-to-end"""
    # Set up temporary paths
    preprocessor.input_path = str(tmp_path / 'input')
    preprocessor.output_path = str(tmp_path / 'output')
    
    os.makedirs(preprocessor.input_path, exist_ok=True)
    os.makedirs(preprocessor.output_path, exist_ok=True)
    
    # Save sample data
    input_file = os.path.join(preprocessor.input_path, 'mimic_discharge_labs.csv')
    sample_data.to_csv(input_file, index=False)
    
    # Run pipeline
    df_processed, report = preprocessor.run_preprocessing_pipeline()
    
    # Verify output
    assert df_processed is not None
    assert len(df_processed) == 5  # Duplicates removed
    assert 'ethnicity_clean' in df_processed.columns
    assert 'age_group' in df_processed.columns
    
    # Verify report generated
    assert 'initial_records' in report
    assert 'final_records' in report
    assert report['initial_records'] == 6
    assert report['final_records'] == 5
    
    # Verify output file created
    output_file = os.path.join(preprocessor.output_path, 'processed_discharge_summaries.csv')
    assert os.path.exists(output_file)


# Edge Case Tests

def test_empty_dataframe(preprocessor):
    """Test handling of empty DataFrame"""
    empty_df = pd.DataFrame()
    
    # Should handle gracefully
    result = preprocessor.remove_duplicates(empty_df)
    assert len(result) == 0


def test_all_duplicates(preprocessor):
    """Test when all records are duplicates"""
    duplicate_data = pd.DataFrame({
        'hadm_id': [1001, 1001, 1001],
        'subject_id': [101, 101, 101],
        'cleaned_text': ['same', 'same', 'same']
    })
    
    result = preprocessor.remove_duplicates(duplicate_data)
    
    # Should keep only 1 record
    assert len(result) == 1
    assert result['hadm_id'].iloc[0] == 1001


def test_missing_demographic_columns(preprocessor):
    """Test demographic features when columns are missing"""
    minimal_data = pd.DataFrame({
        'hadm_id': [1001, 1002],
        'subject_id': [101, 102],
        'cleaned_text': ['text1', 'text2']
    })
    
    # Should handle missing ethnicity and age gracefully
    # This might raise an error or return the dataframe unchanged
    # depending on implementation - adjust based on your actual behavior
    try:
        result = preprocessor.create_demographic_features(minimal_data)
        # If it doesn't raise error, check it handles gracefully
        assert result is not None
    except KeyError:
        # Expected if columns are required
        pass


def test_special_characters_in_text(preprocessor):
    """Test text cleaning with special characters"""
    text_with_special = "Patient's vital signs: BP=120/80, HR=75bpm, Temp=98.6°F"
    
    clean = preprocessor.clean_text(text_with_special)
    
    # Should clean but preserve meaningful content
    assert len(clean) > 0
    assert 'vital signs' in clean.lower()


def test_very_long_text(preprocessor):
    """Test handling of very long discharge summaries"""
    long_text = "Patient admitted. " * 5000  # Very long text
    
    # Should handle without errors
    clean = preprocessor.clean_text(long_text)
    assert len(clean) > 0
    assert clean.count('  ') == 0  # No double spaces


def test_abbreviations_case_insensitive(preprocessor):
    """Test that abbreviation expansion is case insensitive"""
    # Test various cases
    test_cases = [
        "Pt admitted",
        "PT admitted",
        "pt admitted"
    ]
    
    for text in test_cases:
        expanded = preprocessor.expand_abbreviations(text)
        assert 'patient' in expanded.lower()


def test_age_group_boundaries(preprocessor):
    """Test age group boundary conditions"""
    boundary_data = pd.DataFrame({
        'hadm_id': range(1001, 1007),
        'subject_id': range(101, 107),
        'cleaned_text': ['text'] * 6,
        'ethnicity': ['WHITE'] * 6,
        'gender': ['M'] * 6,
        'age_at_admission': [0, 18, 35, 50, 65, 120]  # Boundary ages
    })
    
    df_with_groups = preprocessor.create_demographic_features(boundary_data)
    
    # Verify boundary handling
    assert df_with_groups is not None
    assert 'age_group' in df_with_groups.columns
    
    # Check that all ages got assigned to groups
    assert df_with_groups['age_group'].isna().sum() == 0


# Integration Tests

def test_preprocessing_maintains_record_count_logic(preprocessor, sample_data):
    """Test that preprocessing maintains logical record counts"""
    initial_unique = sample_data['hadm_id'].nunique()
    
    cleaned = preprocessor.remove_duplicates(sample_data)
    
    # After deduplication, unique count should equal total count
    assert len(cleaned) == cleaned['hadm_id'].nunique()
    assert len(cleaned) <= len(sample_data)


def test_features_are_numeric_where_expected(preprocessor, sample_data):
    """Test that created features have correct data types"""
    # Add required columns
    sample_data['discharge_medications'] = ['meds'] * len(sample_data)
    sample_data['follow_up'] = ['followup'] * len(sample_data)
    sample_data['discharge_diagnosis'] = ['dx'] * len(sample_data)
    
    df_features = preprocessor.calculate_text_features(sample_data)
    
    # Numeric features should be numeric
    assert pd.api.types.is_numeric_dtype(df_features['text_length'])
    assert pd.api.types.is_numeric_dtype(df_features['word_count'])
    assert pd.api.types.is_numeric_dtype(df_features['sentence_count'])
    assert pd.api.types.is_numeric_dtype(df_features['medical_term_count'])


# Run tests with: pytest tests/test_preprocessing.py -v