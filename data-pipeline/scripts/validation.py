"""
Data Validation Pipeline for MIMIC-III Processed Data
Author: Team Member 3
Description: Validates data quality, integrity, and completeness
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIMICDataValidator:
    """Comprehensive data validation for MIMIC-III pipeline"""
    
    def __init__(self, input_path: str = 'data/processed', output_path: str = 'logs'):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Define validation rules
        self.validation_rules = {
            'text_length_min': 100,
            'text_length_max': 100000,
            'age_min': 0,
            'age_max': 120,
            'lab_value_ranges': {
                'creatinine': (0.1, 20.0),
                'glucose': (10, 1000),
                'hemoglobin': (3.0, 20.0),
                'sodium': (100, 180),
                'potassium': (1.5, 10.0)
            },
            'required_columns': ['hadm_id', 'subject_id', 'cleaned_text'],
            'expected_sections': ['discharge_diagnosis', 'discharge_medications', 'follow_up']
        }
        
    def load_data(self, filename: str = 'processed_discharge_summaries.csv') -> pd.DataFrame:
        """Load processed data for validation"""
        # Try to load from processed folder first, then fall back to raw
        filepath = os.path.join(self.input_path, filename)
        if not os.path.exists(filepath):
            # If processed file doesn't exist, try raw data
            logger.warning(f"Processed file not found at {filepath}, trying raw data")
            filepath = os.path.join('data/raw', 'mimic_discharge_labs.csv')
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records for validation")
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema and structure"""
        schema_report = {
            'total_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_required_columns': [],
            'schema_valid': True
        }
        
        # Check for required columns
        for col in self.validation_rules['required_columns']:
            if col not in df.columns:
                schema_report['missing_required_columns'].append(col)
                schema_report['schema_valid'] = False
                logger.warning(f"Missing required column: {col}")
        
        return schema_report
    
    def validate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness and missing values"""
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
        
        # Check critical fields
        if 'cleaned_text' in df.columns:
            completeness_report['records_without_text'] = int(
                (df['cleaned_text'].isna() | (df['cleaned_text'] == '')).sum()
            )
        
        if 'discharge_diagnosis' in df.columns:
            completeness_report['records_without_diagnosis'] = int(
                (df['discharge_diagnosis'].isna() | (df['discharge_diagnosis'] == '')).sum()
            )
        
        if 'discharge_medications' in df.columns:
            completeness_report['records_without_medications'] = int(
                (df['discharge_medications'].isna() | (df['discharge_medications'] == '')).sum()
            )
        
        return completeness_report
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics"""
        quality_report = {
            'text_length_issues': {},
            'duplicate_records': {},
            'outliers': {},
            'data_consistency_issues': []
        }
        
        # Check text length boundaries
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
        
        # Check data consistency
        if 'word_count' in df.columns and 'text_length' in df.columns:
            # Words should be less than characters
            inconsistent = (df['word_count'] > df['text_length']).sum()
            if inconsistent > 0:
                quality_report['data_consistency_issues'].append(
                    f"Found {inconsistent} records where word_count > text_length"
                )
        
        return quality_report
    
    def validate_lab_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate lab values are within reasonable ranges"""
        lab_report = {
            'total_records_with_labs': 0,
            'invalid_lab_values': {},
            'lab_value_statistics': {}
        }
        
        # Check if lab_summary exists
        if 'lab_summary' in df.columns:
            lab_report['total_records_with_labs'] = int((~df['lab_summary'].isna()).sum())
        
        if 'abnormal_count' in df.columns:
            lab_report['abnormal_count_stats'] = {
                'min': int(df['abnormal_count'].min()) if len(df) > 0 else 0,
                'max': int(df['abnormal_count'].max()) if len(df) > 0 else 0,
                'mean': float(df['abnormal_count'].mean()) if len(df) > 0 else 0
            }
        
        return lab_report
    
    def validate_section_extraction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that sections were properly extracted"""
        section_report = {
            'section_extraction_rates': {},
            'average_section_lengths': {},
            'empty_sections': {}
        }
        
        for section in self.validation_rules['expected_sections']:
            if section in df.columns:
                # Calculate extraction rate (non-empty)
                non_empty = (~df[section].isna() & (df[section] != '')).sum()
                extraction_rate = (non_empty / len(df)) * 100 if len(df) > 0 else 0
                section_report['section_extraction_rates'][section] = round(extraction_rate, 2)
                
                # Calculate average length of non-empty sections
                non_empty_sections = df[df[section] != ''][section] if section in df.columns else pd.Series()
                if len(non_empty_sections) > 0:
                    avg_length = non_empty_sections.str.len().mean()
                    section_report['average_section_lengths'][section] = round(avg_length, 2)
                
                # Count empty sections
                empty_count = (df[section] == '').sum() if section in df.columns else 0
                section_report['empty_sections'][section] = int(empty_count)
        
        return section_report
    
    def validate_identifiers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate patient and admission identifiers"""
        id_report = {
            'unique_patients': 0,
            'unique_admissions': 0,
            'patients_with_multiple_admissions': 0,
            'invalid_ids': [],
            'id_format_issues': []
        }
        
        if 'subject_id' in df.columns:
            id_report['unique_patients'] = int(df['subject_id'].nunique())
            
            # Check for invalid IDs (negative or zero)
            invalid_subjects = df[df['subject_id'] <= 0]['subject_id'].tolist() if 'subject_id' in df.columns else []
            if invalid_subjects:
                id_report['invalid_ids'].extend([int(x) for x in invalid_subjects])
        
        if 'hadm_id' in df.columns:
            id_report['unique_admissions'] = int(df['hadm_id'].nunique())
            
            # Check for invalid admission IDs
            invalid_hadm = df[df['hadm_id'] <= 0]['hadm_id'].tolist() if 'hadm_id' in df.columns else []
            if invalid_hadm:
                id_report['invalid_ids'].extend([int(x) for x in invalid_hadm])
        
        if 'subject_id' in df.columns and 'hadm_id' in df.columns:
            # Find patients with multiple admissions
            admission_counts = df.groupby('subject_id')['hadm_id'].nunique()
            id_report['patients_with_multiple_admissions'] = int((admission_counts > 1).sum())
        
        return id_report
    
    def run_validation_pipeline(self) -> Tuple[Dict, pd.DataFrame]:
        """Run complete validation pipeline"""
        logger.info("Starting validation pipeline...")
        
        # Load data
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
        
        logger.info("Validating lab values...")
        validation_report['lab_values'] = self.validate_lab_values(df)
        
        logger.info("Validating section extraction...")
        validation_report['sections'] = self.validate_section_extraction(df)
        
        logger.info("Validating identifiers...")
        validation_report['identifiers'] = self.validate_identifiers(df)
        
        # Calculate overall validation score
        validation_report['overall_score'] = self.calculate_validation_score(validation_report)
        
        # Save validation report with proper JSON serialization
        report_path = os.path.join(self.output_path, 'validation_report.json')
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
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
        
        # Create summary DataFrame
        summary_df = self.create_validation_summary(validation_report)
        summary_path = os.path.join(self.output_path, 'validation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Validation summary saved to {summary_path}")
        
        return validation_report, summary_df
    
    def calculate_validation_score(self, report: Dict) -> float:
        """Calculate overall validation score (0-100)"""
        score = 100.0
        penalties = {
            'missing_required_columns': 10,
            'duplicate_records': 5,
            'text_too_short': 2,
            'missing_critical_sections': 3,
            'invalid_lab_values': 2
        }
        
        # Apply penalties
        if report['schema']['missing_required_columns']:
            score -= penalties['missing_required_columns'] * len(report['schema']['missing_required_columns'])
        
        if report['quality']['duplicate_records']['duplicate_rows'] > 0:
            score -= penalties['duplicate_records']
        
        if report['quality']['text_length_issues'].get('too_short', 0) > 10:
            score -= penalties['text_too_short']
        
        # Check section extraction rates if they exist
        if 'sections' in report and 'section_extraction_rates' in report['sections']:
            if report['sections']['section_extraction_rates'].get('discharge_diagnosis', 100) < 50:
                score -= penalties['missing_critical_sections']
        
        return max(0, score)
    
    def create_validation_summary(self, report: Dict) -> pd.DataFrame:
        """Create a summary DataFrame from validation report"""
        summary_data = {
            'Metric': [],
            'Value': [],
            'Status': []
        }
        
        # Add key metrics
        metrics = [
            ('Total Records', report['dataset_info']['total_records'], 'INFO'),
            ('Schema Valid', report['schema']['schema_valid'], 'PASS' if report['schema']['schema_valid'] else 'FAIL'),
            ('Records Without Text', report['completeness']['records_without_text'], 
             'PASS' if report['completeness']['records_without_text'] == 0 else 'WARNING'),
            ('Duplicate Records', report['quality']['duplicate_records']['duplicate_rows'],
             'PASS' if report['quality']['duplicate_records']['duplicate_rows'] == 0 else 'WARNING'),
            ('Validation Score', f"{report['overall_score']:.2f}%",
             'PASS' if report['overall_score'] >= 80 else 'WARNING' if report['overall_score'] >= 60 else 'FAIL')
        ]
        
        for metric, value, status in metrics:
            summary_data['Metric'].append(metric)
            summary_data['Value'].append(value)
            summary_data['Status'].append(status)
        
        return pd.DataFrame(summary_data)

if __name__ == "__main__":
    # Run validation
    validator = MIMICDataValidator()
    report, summary = validator.run_validation_pipeline()
    
    print("\n=== Validation Complete ===")
    print(f"Overall Validation Score: {report['overall_score']:.2f}%")
    print(f"Schema Valid: {report['schema']['schema_valid']}")
    print(f"Total Records: {report['dataset_info']['total_records']}")
    print(f"Duplicate Records: {report['quality']['duplicate_records']['duplicate_rows']}")
    print(f"Records Without Text: {report['completeness']['records_without_text']}")
    print("\nValidation report saved to: logs/validation_report.json")