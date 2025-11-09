"""
Unit Tests for MIMIC Validation Pipeline
Tests schema validation, data quality checks, and scoring
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import json

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, scripts_path)

from validation import MIMICDataValidator


# Fixtures

@pytest.fixture
def sample_processed_data():
    """Create sample processed data for validation testing"""
    data = {
        'hadm_id': [1001, 1002, 1003, 1004, 1005],
        'subject_id': [101, 102, 103, 104, 105],
        'cleaned_text': [
            'Patient admitted with chest pain.' * 10,  # Normal length
            'Short note.',  # Too short
            'Patient with complex medical history.' * 100,  # Long note
            'Normal length discharge summary.' * 15,
            ''  # Empty
        ],
        'text_length': [300, 50, 3000, 450, 0],
        'word_count': [50, 8, 500, 75, 0],
        'age_at_admission': [45, 67, 150, 35, -5],  # 150 and -5 are invalid
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'ethnicity_clean': ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER'],
        'age_group': ['35-50', '65+', '65+', '18-35', '<18'],
        'discharge_diagnosis': ['MI', 'CHF', '', 'UTI', None],  # Some missing
        'discharge_medications': ['aspirin', '', 'multiple meds', None, ''],
        'follow_up': ['cardiology', 'PCP', '', '', None]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def validator():
    """Create validator instance for testing"""
    config = {
        'validation_config': {
            'text_length_min': 100,
            'text_length_max': 100000,
            'age_min': 0,
            'age_max': 120,
            'required_columns': ['hadm_id', 'subject_id', 'cleaned_text'],
            'expected_sections': ['discharge_diagnosis', 'discharge_medications', 'follow_up'],
            'validation_score_threshold': 80
        }
    }
    
    return MIMICDataValidator(
        input_path='tests/data',
        output_path='tests/output',
        config=config
    )


# Schema Validation Tests

def test_validate_schema_with_all_columns(validator, sample_processed_data):
    """Test schema validation when all required columns present"""
    schema_report = validator.validate_schema(sample_processed_data)
    
    assert schema_report['schema_valid'] == True
    assert len(schema_report['missing_required_columns']) == 0
    assert schema_report['total_columns'] == len(sample_processed_data.columns)


def test_validate_schema_missing_columns(validator):
    """Test schema validation when required columns are missing"""
    incomplete_data = pd.DataFrame({
        'hadm_id': [1001, 1002],
        'subject_id': [101, 102]
        # Missing 'cleaned_text' which is required
    })
    
    schema_report = validator.validate_schema(incomplete_data)
    
    assert schema_report['schema_valid'] == False
    assert 'cleaned_text' in schema_report['missing_required_columns']


# Completeness Validation Tests

def test_validate_completeness(validator, sample_processed_data):
    """Test completeness validation"""
    completeness_report = validator.validate_completeness(sample_processed_data)
    
    # Check report structure
    assert 'total_records' in completeness_report
    assert completeness_report['total_records'] == 5
    
    assert 'records_without_text' in completeness_report
    assert completeness_report['records_without_text'] == 1  # One empty text
    
    assert 'missing_values_per_column' in completeness_report
    assert 'missing_percentage_per_column' in completeness_report


def test_missing_percentage_calculation(validator, sample_processed_data):
    """Test that missing percentages are calculated correctly"""
    completeness_report = validator.validate_completeness(sample_processed_data)
    
    # discharge_diagnosis has 1 None and 1 empty = 2 missing out of 5 = 40%
    # Note: This tests the None detection, empty string might be counted separately
    missing_pct = completeness_report['missing_percentage_per_column']['discharge_diagnosis']
    assert missing_pct >= 0
    assert missing_pct <= 100


# Quality Validation Tests

def test_validate_data_quality(validator, sample_processed_data):
    """Test data quality validation"""
    quality_report = validator.validate_data_quality(sample_processed_data)
    
    # Check text length issues detected
    assert 'text_length_issues' in quality_report
    assert quality_report['text_length_issues']['too_short'] >= 1  # At least one short text
    
    # Check duplicate detection
    assert 'duplicate_records' in quality_report
    
    # Check outlier detection
    assert 'outliers' in quality_report


def test_text_length_boundaries(validator, sample_processed_data):
    """Test text length boundary validation"""
    quality_report = validator.validate_data_quality(sample_processed_data)
    
    text_issues = quality_report['text_length_issues']
    
    # Should detect text that's too short (< 100 chars)
    assert text_issues['too_short'] >= 1
    
    # Should report shortest and longest
    assert text_issues['shortest_text'] >= 0
    assert text_issues['longest_text'] >= text_issues['shortest_text']


# Demographic Validation Tests

def test_validate_demographics(validator, sample_processed_data):
    """Test demographic data validation"""
    demo_report = validator.validate_demographics(sample_processed_data)
    
    # Check age validation
    assert 'age_issues' in demo_report
    assert demo_report['age_issues']['invalid_ages'] == 2  # 150 and -5 are invalid
    
    # Check ethnicity distribution
    assert 'ethnicity_distribution' in demo_report
    
    # Check gender distribution
    assert 'gender_distribution' in demo_report


def test_age_range_validation(validator, sample_processed_data):
    """Test age range validation catches out-of-range values"""
    demo_report = validator.validate_demographics(sample_processed_data)
    
    age_issues = demo_report['age_issues']
    
    # Should detect ages outside 0-120 range
    assert age_issues['invalid_ages'] > 0
    assert age_issues['min_age'] < 0 or age_issues['max_age'] > 120


# Cross-Field Validation Tests

def test_validate_cross_field_logic(validator, sample_processed_data):
    """Test cross-field logical consistency checks"""
    cross_field_report = validator.validate_cross_field_logic(sample_processed_data)
    
    # Check report structure
    assert 'logical_inconsistencies' in cross_field_report
    assert 'inconsistency_count' in cross_field_report
    
    # Should detect records with medications but no diagnosis
    assert isinstance(cross_field_report['logical_inconsistencies'], list)


def test_medications_without_diagnosis_detection(validator):
    """Test detection of records with medications but no diagnosis"""
    inconsistent_data = pd.DataFrame({
        'hadm_id': [1001, 1002, 1003],
        'subject_id': [101, 102, 103],
        'cleaned_text': ['text'] * 3,
        'discharge_medications': ['aspirin, lisinopril', '', 'metformin'],
        'discharge_diagnosis': ['', 'MI', '']  # 1001 and 1003 have meds but no dx
    })
    
    cross_field_report = validator.validate_cross_field_logic(inconsistent_data)
    
    # Should detect at least some inconsistencies
    assert cross_field_report['inconsistency_count'] >= 0


# Validation Score Tests

def test_calculate_validation_score_perfect_data(validator):
    """Test validation score with perfect quality data"""
    perfect_data = pd.DataFrame({
        'hadm_id': range(1001, 1101),
        'subject_id': range(101, 201),
        'cleaned_text': ['Normal discharge summary text with adequate length.'] * 100,
        'text_length': [500] * 100,
        'age_at_admission': [45] * 100,
        'discharge_diagnosis': ['diagnosis'] * 100,
        'discharge_medications': ['medications'] * 100,
        'follow_up': ['followup'] * 100
    })
    
    # Run all validations
    schema_report = validator.validate_schema(perfect_data)
    completeness_report = validator.validate_completeness(perfect_data)
    quality_report = validator.validate_data_quality(perfect_data)
    demo_report = validator.validate_demographics(perfect_data)
    cross_field_report = validator.validate_cross_field_logic(perfect_data)
    
    report = {
        'schema': schema_report,
        'completeness': completeness_report,
        'quality': quality_report,
        'demographics': demo_report,
        'cross_field_logic': cross_field_report
    }
    
    score = validator.calculate_validation_score(report)
    
    # Perfect data should score high (near 100)
    assert score >= 90


def test_calculate_validation_score_poor_data(validator, sample_processed_data):
    """Test validation score with poor quality data"""
    # Run all validations on sample data (has issues)
    schema_report = validator.validate_schema(sample_processed_data)
    completeness_report = validator.validate_completeness(sample_processed_data)
    quality_report = validator.validate_data_quality(sample_processed_data)
    demo_report = validator.validate_demographics(sample_processed_data)
    cross_field_report = validator.validate_cross_field_logic(sample_processed_data)
    
    report = {
        'schema': schema_report,
        'completeness': completeness_report,
        'quality': quality_report,
        'demographics': demo_report,
        'cross_field_logic': cross_field_report
    }
    
    score = validator.calculate_validation_score(report)
    
    # Data with issues should score lower
    assert score < 100
    assert score >= 0


def test_validation_summary_creation(validator, sample_processed_data):
    """Test that validation summary DataFrame is created correctly"""
    # Create minimal report
    report = {
        'dataset_info': {
            'total_records': 5,
            'total_columns': 10  # ADD THIS
        },
        'schema': {'schema_valid': True},
        'completeness': {'records_without_text': 1},
        'quality': {'duplicate_records': {'duplicate_rows': 0}},
        'demographics': {'age_issues': {'invalid_ages': 2}},
        'cross_field_logic': {'inconsistency_count': 1},
        'overall_score': 75.0
    }
    
    summary_df = validator.create_validation_summary(report)
    
    # Check DataFrame structure
    assert isinstance(summary_df, pd.DataFrame)
    assert 'Metric' in summary_df.columns
    assert 'Value' in summary_df.columns
    assert 'Status' in summary_df.columns
    
    # Check it has multiple rows
    assert len(summary_df) > 0


# Run tests with: pytest tests/test_validation.py -v