"""
Data Validation Pipeline for MIMIC-III Processed Data
Author: Team Member 3
Description: Validates data quality, integrity, and completeness with config-driven rules
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger, log_data_operation
from utils.error_handling import (
    DataValidationError, ErrorHandler, safe_execute, 
    validate_dataframe, validate_file_path, ErrorContext
)

# Set up logging
logger = get_logger(__name__)

class MIMICDataValidator:
    """Comprehensive data validation for MIMIC-III pipeline"""
    
    def __init__(self, input_path: str = 'data/processed', output_path: str = 'logs', config: Dict = None):
        """
        Initialize validator with paths and validation rules
        
        Args:
            input_path: Path to processed data directory
            output_path: Path to save validation reports
            config: Configuration dictionary with validation rules
        """
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Initialized validator with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # Load validation rules from config or use defaults
        if config and 'validation_config' in config:
            self.validation_rules = config['validation_config']
        else:
            # Default validation rules
            self.validation_rules = {
                'text_length_min': 100,
                'text_length_max': 100000,
                'age_min': 0,
                'age_max': 120,
                'required_columns': ['hadm_id', 'subject_id', 'cleaned_text'],
                'expected_sections': ['discharge_diagnosis', 'discharge_medications', 'follow_up'],
                'validation_score_threshold': 80
            }
        
        logger.info(f"Using validation rules: {self.validation_rules}")
        
    def load_data(self, filename: str = 'processed_discharge_summaries.csv') -> pd.DataFrame:
        """
        Load processed data for validation
        
        Args:
            filename: Name of file to validate
            
        Returns:
            DataFrame containing processed data
        """
        filepath = os.path.join(self.input_path, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Processed file not found at {filepath}, trying raw data")
            filepath = os.path.join('data-pipeline/data/raw', 'mimic_discharge_labs.csv')
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records for validation")
        
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data schema and structure against required columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with schema validation results
        """
        schema_report = {
            'total_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_required_columns': [],
            'schema_valid': True
        }
        
        # Check for required columns from config
        for col in self.validation_rules['required_columns']:
            if col not in df.columns:
                schema_report['missing_required_columns'].append(col)
                schema_report['schema_valid'] = False
                logger.warning(f"Missing required column: {col}")
        
        return schema_report
    
    def validate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data completeness and missing values across all columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with completeness validation results
        """
        completeness_report = {
            'total_records': int(len(df)),
            'missing_values_per_column': {},
            'missing_percentage_per_column': {},
            'completely_empty_rows': 0,
            'records_without_text': 0,
            'records_without_diagnosis': 0,
            'records_without_medications': 0
        }
        
        # Calculate missing values for each column
        for col in df.columns:
            missing_count = df[col].isna().sum()
            completeness_report['missing_values_per_column'][col] = int(missing_count)
            completeness_report['missing_percentage_per_column'][col] = round(
                (missing_count / len(df)) * 100, 2
            )
        
        # Check for completely empty rows
        completeness_report['completely_empty_rows'] = int(df.isna().all(axis=1).sum())
        
        # Check critical text field
        if 'cleaned_text' in df.columns:
            completeness_report['records_without_text'] = int(
                (df['cleaned_text'].isna() | (df['cleaned_text'] == '')).sum()
            )
        
        # Check diagnosis field
        if 'discharge_diagnosis' in df.columns:
            completeness_report['records_without_diagnosis'] = int(
                (df['discharge_diagnosis'].isna() | (df['discharge_diagnosis'] == '')).sum()
            )
        
        # Check medications field
        if 'discharge_medications' in df.columns:
            completeness_report['records_without_medications'] = int(
                (df['discharge_medications'].isna() | (df['discharge_medications'] == '')).sum()
            )
        
        return completeness_report
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality including duplicates, outliers, and consistency
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality validation results
        """
        quality_report = {
            'text_length_issues': {},
            'duplicate_records': {},
            'outliers': {},
            'data_consistency_issues': []
        }
        
        # Validate text length against config rules
        if 'text_length' in df.columns:
            too_short = (df['text_length'] < self.validation_rules['text_length_min']).sum()
            too_long = (df['text_length'] > self.validation_rules['text_length_max']).sum()
            
            quality_report['text_length_issues'] = {
                'too_short': int(too_short),
                'too_long': int(too_long),
                'shortest_text': int(df['text_length'].min()) if len(df) > 0 else 0,
                'longest_text': int(df['text_length'].max()) if len(df) > 0 else 0,
                'average_length': float(df['text_length'].mean()) if len(df) > 0 else 0
            }
        
        # Check for duplicate records
        quality_report['duplicate_records'] = {
            'duplicate_hadm_ids': int(df['hadm_id'].duplicated().sum()) if 'hadm_id' in df.columns else 0,
            'duplicate_texts': int(df['cleaned_text'].duplicated().sum()) if 'cleaned_text' in df.columns else 0,
            'duplicate_rows': int(df.duplicated().sum())
        }
        
        # Detect outliers using IQR method for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['text_length', 'word_count', 'abnormal_count']:
                if len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                    quality_report['outliers'][col] = int(outliers)
        
        # Check data consistency between related fields
        if 'word_count' in df.columns and 'text_length' in df.columns:
            # Word count should be less than character count
            inconsistent = (df['word_count'] > df['text_length']).sum()
            if inconsistent > 0:
                quality_report['data_consistency_issues'].append(
                    f"Found {inconsistent} records where word_count > text_length"
                )
        
        return quality_report
    
    def validate_demographics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate demographic data quality and distributions
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with demographic validation results
        """
        demo_report = {
            'age_issues': {},
            'ethnicity_distribution': {},
            'gender_distribution': {},
            'age_group_distribution': {}
        }
        
        # Validate age ranges against config rules
        if 'age_at_admission' in df.columns:
            invalid_age = (
                (df['age_at_admission'] < self.validation_rules['age_min']) | 
                (df['age_at_admission'] > self.validation_rules['age_max'])
            ).sum()
            
            demo_report['age_issues'] = {
                'invalid_ages': int(invalid_age),
                'min_age': float(df['age_at_admission'].min()) if len(df) > 0 else 0,
                'max_age': float(df['age_at_admission'].max()) if len(df) > 0 else 0,
                'avg_age': float(df['age_at_admission'].mean()) if len(df) > 0 else 0,
                'missing_age': int(df['age_at_admission'].isna().sum())
            }
        
        # Check ethnicity standardization results
        if 'ethnicity_clean' in df.columns:
            demo_report['ethnicity_distribution'] = df['ethnicity_clean'].value_counts().to_dict()
            demo_report['ethnicity_missing'] = int(df['ethnicity_clean'].isna().sum())
        
        # Check gender distribution
        if 'gender' in df.columns:
            demo_report['gender_distribution'] = df['gender'].value_counts().to_dict()
            demo_report['gender_missing'] = int(df['gender'].isna().sum())
        
        # Check age group distribution from preprocessing
        if 'age_group' in df.columns:
            demo_report['age_group_distribution'] = df['age_group'].value_counts().to_dict()
        
        return demo_report
    
    def validate_cross_field_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate logical consistency across related fields
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with cross-field validation results
        """
        cross_field_report = {
            'logical_inconsistencies': [],
            'inconsistency_count': 0
        }
        
        # Check if medications exist but diagnosis doesn't
        if 'discharge_medications' in df.columns and 'discharge_diagnosis' in df.columns:
            has_meds_no_dx = (
                (df['discharge_medications'].str.len() > 10) & 
                ((df['discharge_diagnosis'].isna()) | (df['discharge_diagnosis'] == ''))
            ).sum()
            
            if has_meds_no_dx > 0:
                cross_field_report['logical_inconsistencies'].append(
                    f"Found {has_meds_no_dx} records with medications but no diagnosis"
                )
                cross_field_report['inconsistency_count'] += has_meds_no_dx
        
        # Check if follow_up exists without diagnosis
        if 'follow_up' in df.columns and 'discharge_diagnosis' in df.columns:
            has_followup_no_dx = (
                (df['follow_up'].str.len() > 10) & 
                ((df['discharge_diagnosis'].isna()) | (df['discharge_diagnosis'] == ''))
            ).sum()
            
            if has_followup_no_dx > 0:
                cross_field_report['logical_inconsistencies'].append(
                    f"Found {has_followup_no_dx} records with follow-up but no diagnosis"
                )
                cross_field_report['inconsistency_count'] += has_followup_no_dx
        
        # Check pediatric patients with unusual insurance
        if 'age_at_admission' in df.columns and 'insurance' in df.columns:
            # Children under 18 shouldn't typically have Medicare (age 65+ insurance)
            pediatric_medicare = (
                (df['age_at_admission'] < 18) & 
                (df['insurance'].str.upper().str.contains('MEDICARE', na=False))
            ).sum()
            
            if pediatric_medicare > 0:
                cross_field_report['logical_inconsistencies'].append(
                    f"Found {pediatric_medicare} pediatric patients with Medicare (unusual)"
                )
        
        return cross_field_report
    
    def validate_lab_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate lab values are within reasonable clinical ranges
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with lab value validation results
        """
        lab_report = {
            'total_records_with_labs': 0,
            'invalid_lab_values': {},
            'lab_value_statistics': {}
        }
        
        # Count records with lab data
        if 'lab_summary' in df.columns:
            lab_report['total_records_with_labs'] = int((~df['lab_summary'].isna()).sum())
        
        # Validate abnormal count statistics
        if 'abnormal_count' in df.columns:
            lab_report['abnormal_count_stats'] = {
                'min': int(df['abnormal_count'].min()) if len(df) > 0 else 0,
                'max': int(df['abnormal_count'].max()) if len(df) > 0 else 0,
                'mean': float(df['abnormal_count'].mean()) if len(df) > 0 else 0,
                'missing': int(df['abnormal_count'].isna().sum())
            }
        
        return lab_report
    
    def validate_section_extraction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that clinical sections were properly extracted from discharge summaries
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with section extraction validation results
        """
        section_report = {
            'section_extraction_rates': {},
            'average_section_lengths': {},
            'empty_sections': {}
        }
        
        # Check each expected section from config
        for section in self.validation_rules['expected_sections']:
            if section in df.columns:
                # Calculate extraction rate (percentage of non-empty sections)
                non_empty = (~df[section].isna() & (df[section] != '')).sum()
                extraction_rate = (non_empty / len(df)) * 100 if len(df) > 0 else 0
                section_report['section_extraction_rates'][section] = round(extraction_rate, 2)
                
                # Calculate average length of non-empty sections
                non_empty_sections = df[df[section] != ''][section] if section in df.columns else pd.Series()
                if len(non_empty_sections) > 0:
                    avg_length = non_empty_sections.astype(str).str.len().mean()
                    section_report['average_section_lengths'][section] = round(avg_length, 2)
                
                # Count empty sections
                empty_count = (df[section] == '').sum() if section in df.columns else 0
                section_report['empty_sections'][section] = int(empty_count)
        
        return section_report
    
    def validate_identifiers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate patient and admission identifiers for correctness
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with identifier validation results
        """
        id_report = {
            'unique_patients': 0,
            'unique_admissions': 0,
            'patients_with_multiple_admissions': 0,
            'invalid_ids': [],
            'id_format_issues': []
        }
        
        # Validate patient IDs
        if 'subject_id' in df.columns:
            id_report['unique_patients'] = int(df['subject_id'].nunique())
            
            # Check for invalid IDs (negative or zero)
            invalid_subjects = df[df['subject_id'] <= 0]['subject_id'].tolist()
            if invalid_subjects:
                id_report['invalid_ids'].extend([int(x) for x in invalid_subjects])
                logger.warning(f"Found {len(invalid_subjects)} invalid subject IDs")
        
        # Validate admission IDs
        if 'hadm_id' in df.columns:
            id_report['unique_admissions'] = int(df['hadm_id'].nunique())
            
            # Check for invalid admission IDs
            invalid_hadm = df[df['hadm_id'] <= 0]['hadm_id'].tolist()
            if invalid_hadm:
                id_report['invalid_ids'].extend([int(x) for x in invalid_hadm])
                logger.warning(f"Found {len(invalid_hadm)} invalid admission IDs")
        
        # Find patients with multiple admissions
        if 'subject_id' in df.columns and 'hadm_id' in df.columns:
            admission_counts = df.groupby('subject_id')['hadm_id'].nunique()
            id_report['patients_with_multiple_admissions'] = int((admission_counts > 1).sum())
        
        return id_report
    
    def run_validation_pipeline(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run complete validation pipeline with all checks
        
        Returns:
            Tuple of (validation report dictionary, summary DataFrame)
        """
        logger.info("Starting validation pipeline...")
        
        # Load processed data
        df = self.load_data()
        
        # Initialize validation report
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_records': int(len(df)),
                'total_columns': int(len(df.columns))
            }
        }
        
        # Run all validation checks
        logger.info("Validating schema...")
        validation_report['schema'] = self.validate_schema(df)
        
        logger.info("Validating completeness...")
        validation_report['completeness'] = self.validate_completeness(df)
        
        logger.info("Validating data quality...")
        validation_report['quality'] = self.validate_data_quality(df)
        
        logger.info("Validating demographics...")
        validation_report['demographics'] = self.validate_demographics(df)
        
        logger.info("Validating cross-field logic...")
        validation_report['cross_field_logic'] = self.validate_cross_field_logic(df)
        
        logger.info("Validating lab values...")
        validation_report['lab_values'] = self.validate_lab_values(df)
        
        logger.info("Validating section extraction...")
        validation_report['sections'] = self.validate_section_extraction(df)
        
        logger.info("Validating identifiers...")
        validation_report['identifiers'] = self.validate_identifiers(df)
        
        # Calculate overall validation score
        validation_report['overall_score'] = self.calculate_validation_score(validation_report)
        
        # Save validation report
        report_path = os.path.join(self.output_path, 'validation_report.json')
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_report = convert_to_serializable(validation_report)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        logger.info(f"Validation report saved to {report_path}")
        
        # Create and save summary
        summary_df = self.create_validation_summary(validation_report)
        summary_path = os.path.join(self.output_path, 'validation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Validation summary saved to {summary_path}")
        
        return validation_report, summary_df
    
    def calculate_validation_score(self, report: Dict) -> float:
        """
        Calculate overall validation score (0-100) based on all checks
        
        Args:
            report: Complete validation report dictionary
            
        Returns:
            Validation score between 0 and 100
        """
        score = 100.0
        
        # Define penalties for different issues
        penalties = {
            'missing_required_columns': 15,
            'duplicate_records': 10,
            'text_too_short': 5,
            'missing_critical_sections': 5,
            'invalid_ages': 5,
            'cross_field_issues': 3
        }
        
        # Apply penalty for missing required columns
        if report['schema']['missing_required_columns']:
            penalty = penalties['missing_required_columns'] * len(report['schema']['missing_required_columns'])
            score -= penalty
            logger.warning(f"Penalty applied: -{penalty} for missing required columns")
        
        # Apply penalty for duplicate records
        if report['quality']['duplicate_records']['duplicate_rows'] > 0:
            score -= penalties['duplicate_records']
            logger.warning(f"Penalty applied: -{penalties['duplicate_records']} for duplicate records")
        
        # Apply penalty for text length issues
        if report['quality']['text_length_issues'].get('too_short', 0) > 10:
            score -= penalties['text_too_short']
            logger.warning(f"Penalty applied: -{penalties['text_too_short']} for short text records")
        
        # Apply penalty for poor section extraction
        if 'sections' in report and 'section_extraction_rates' in report['sections']:
            diagnosis_rate = report['sections']['section_extraction_rates'].get('discharge_diagnosis', 100)
            if diagnosis_rate < 50:
                score -= penalties['missing_critical_sections']
                logger.warning(f"Penalty applied: -{penalties['missing_critical_sections']} for low diagnosis extraction")
        
        # Apply penalty for invalid ages
        if 'demographics' in report and 'age_issues' in report['demographics']:
            invalid_ages = report['demographics']['age_issues'].get('invalid_ages', 0)
            if invalid_ages > 0:
                score -= penalties['invalid_ages']
                logger.warning(f"Penalty applied: -{penalties['invalid_ages']} for invalid ages")
        
        # Apply penalty for cross-field inconsistencies
        if 'cross_field_logic' in report:
            inconsistency_count = report['cross_field_logic'].get('inconsistency_count', 0)
            if inconsistency_count > 10:
                score -= penalties['cross_field_issues']
                logger.warning(f"Penalty applied: -{penalties['cross_field_issues']} for cross-field issues")
        
        return max(0, score)
    
    def create_validation_summary(self, report: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame from validation report for easy review
        
        Args:
            report: Complete validation report dictionary
            
        Returns:
            DataFrame with validation summary
        """
        summary_data = {
            'Metric': [],
            'Value': [],
            'Status': []
        }
        
        # Add key validation metrics
        metrics = [
            ('Total Records', report['dataset_info']['total_records'], 'INFO'),
            ('Total Columns', report['dataset_info']['total_columns'], 'INFO'),
            ('Schema Valid', report['schema']['schema_valid'], 
             'PASS' if report['schema']['schema_valid'] else 'FAIL'),
            ('Records Without Text', report['completeness']['records_without_text'], 
             'PASS' if report['completeness']['records_without_text'] == 0 else 'WARNING'),
            ('Duplicate Records', report['quality']['duplicate_records']['duplicate_rows'],
             'PASS' if report['quality']['duplicate_records']['duplicate_rows'] == 0 else 'WARNING'),
            ('Invalid Ages', report['demographics']['age_issues'].get('invalid_ages', 0),
             'PASS' if report['demographics']['age_issues'].get('invalid_ages', 0) == 0 else 'WARNING'),
            ('Cross-Field Issues', report['cross_field_logic'].get('inconsistency_count', 0),
             'PASS' if report['cross_field_logic'].get('inconsistency_count', 0) < 10 else 'WARNING'),
            ('Validation Score', f"{report['overall_score']:.2f}%",
             'PASS' if report['overall_score'] >= 80 else 'WARNING' if report['overall_score'] >= 60 else 'FAIL')
        ]
        
        for metric, value, status in metrics:
            summary_data['Metric'].append(metric)
            summary_data['Value'].append(value)
            summary_data['Status'].append(status)
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    import json
    
    # Load configuration from project
    config_path = os.path.join(os.path.dirname(__file__), '../configs/pipeline_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize validator with config paths and rules
    validator = MIMICDataValidator(
        input_path=config['pipeline_config']['output_path'],
        output_path=config['pipeline_config']['logs_path'],
        config=config
    )
    
    # Run validation pipeline
    report, summary = validator.run_validation_pipeline()
    
    # Print validation results
    print("\n=== Validation Complete ===")
    print(f"Overall Validation Score: {report['overall_score']:.2f}%")
    print(f"Schema Valid: {report['schema']['schema_valid']}")
    print(f"Total Records: {report['dataset_info']['total_records']}")
    print(f"Duplicate Records: {report['quality']['duplicate_records']['duplicate_rows']}")
    print(f"Records Without Text: {report['completeness']['records_without_text']}")
    print(f"Invalid Ages: {report['demographics']['age_issues'].get('invalid_ages', 0)}")
    print(f"Cross-Field Issues: {report['cross_field_logic'].get('inconsistency_count', 0)}")
    print(f"\nValidation report saved to: {config['pipeline_config']['logs_path']}/validation_report.json")
    print(f"Validation summary saved to: {config['pipeline_config']['logs_path']}/validation_summary.csv")
    
    # Display summary table
    print("\n=== Validation Summary ===")
    print(summary.to_string(index=False))