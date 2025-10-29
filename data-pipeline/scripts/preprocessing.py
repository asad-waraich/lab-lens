
# Data Preprocessing Pipeline for MIMIC-III Discharge Summaries
# Author: Team Member 2
# Description: Cleans and processes discharge summaries for ML pipeline

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger, log_data_operation
from utils.error_handling import (
    DataProcessingError, ErrorHandler, safe_execute, 
    validate_dataframe, validate_file_path, ErrorContext
)

# Set up logging
logger = get_logger(__name__)

class MIMICPreprocessor:
    """Preprocessor for MIMIC-III discharge summaries and lab data"""
    
    def __init__(self, input_path: str = 'data/raw', output_path: str = 'data/processed'):
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Initialized preprocessor with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # Medical abbreviations dictionary
        self.medical_abbrev = {
            'pt': 'patient',
            'pts': 'patients',
            'hx': 'history',
            'h/o': 'history of',
            'c/o': 'complaining of',
            'w/': 'with',
            'w/o': 'without',
            's/p': 'status post',
            'yo': 'year old',
            'y/o': 'year old',
            'pod': 'post operative day',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'sx': 'symptoms',
            'abd': 'abdomen',
            'gi': 'gastrointestinal',
            'cv': 'cardiovascular'
        }
        
    @safe_execute("load_data", logger, ErrorHandler(logger))
    @log_data_operation(logger, "load_data")
    def load_data(self, filename: str = 'mimic_discharge_labs.csv') -> pd.DataFrame:
        """Load the raw data"""
        filepath = os.path.join(self.input_path, filename)
        
        validate_file_path(filepath, logger, must_exist=True)
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate DataFrame structure
        required_columns = ['hadm_id', 'subject_id', 'cleaned_text']
        validate_dataframe(df, required_columns, logger)
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from discharge summary"""
        sections = {
            'chief_complaint': '',
            'history_present_illness': '',
            'past_medical_history': '',
            'medications': '',
            'discharge_diagnosis': '',
            'discharge_medications': '',
            'follow_up': '',
            'physical_exam': '',
            'hospital_course': ''
        }
        
        if pd.isna(text):
            return sections
        
        # Section patterns
        patterns = {
            'chief_complaint': r'(?:chief complaint|c\.c\.|cc):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'history_present_illness': r'(?:history of present illness|hpi|present illness):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'past_medical_history': r'(?:past medical history|pmh|medical history):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'medications': r'(?:medications?|meds|current medications?):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'discharge_diagnosis': r'(?:discharge diagnos[ie]s|discharge dx|final diagnos[ie]s):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'discharge_medications': r'(?:discharge medications?|discharge meds):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'follow_up': r'(?:follow[\s-]?up|followup instructions?|disposition):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'physical_exam': r'(?:physical exam|pe|examination):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)',
            'hospital_course': r'(?:hospital course|brief hospital course|summary of hospital course):(.*?)(?=\n[A-Z][a-z]*:|\n\n|\Z)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations"""
        if pd.isna(text):
            return text
        
        expanded = text
        for abbrev, full_form in self.medical_abbrev.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded = re.sub(pattern, full_form, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def clean_text(self, text: str) -> str:
        """Additional text cleaning beyond de-identification"""
        if pd.isna(text):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Fix common OCR/typing errors
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', text)  # Fix number ranges
        
        # Remove isolated special characters
        text = re.sub(r'\s[^\w\s]\s', ' ', text)
        
        return text.strip()
    
    def calculate_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate text-based features for analysis"""
        logger.info("Calculating text features...")
        
        # Basic counts
        df['text_length'] = df['cleaned_text'].fillna('').str.len()
        df['word_count'] = df['cleaned_text'].fillna('').str.split().str.len()
        df['sentence_count'] = df['cleaned_text'].fillna('').str.split('[.!?]').str.len()
        df['avg_word_length'] = df['cleaned_text'].fillna('').apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        # Medical complexity indicators
        df['medical_term_count'] = df['cleaned_text'].fillna('').str.count(
            r'\b(?:diagnosis|medication|symptom|procedure|laboratory|imaging)\b', 
            flags=re.IGNORECASE
        )
        
        # Check for key sections
        df['has_medications'] = df['discharge_medications'].fillna('').str.len() > 10
        df['has_follow_up'] = df['follow_up'].fillna('').str.len() > 10
        df['has_diagnosis'] = df['discharge_diagnosis'].fillna('').str.len() > 10
        
        return df
    
    def process_lab_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and structure lab summary data"""
        logger.info("Processing lab summaries...")
        
        # Handle missing lab data
        df['has_labs'] = ~df['lab_summary'].isna()
        df['lab_summary_clean'] = df['lab_summary'].fillna('No lab results available')
        
        # Extract specific lab patterns if present
        def extract_critical_labs(lab_text):
            if pd.isna(lab_text):
                return {}
            
            critical_labs = {}
            # Look for common critical labs
            patterns = {
                'creatinine': r'(?:creatinine|Cr)[:\s]+([0-9.]+)',
                'glucose': r'(?:glucose|Glu)[:\s]+([0-9.]+)',
                'hemoglobin': r'(?:hemoglobin|Hgb|Hb)[:\s]+([0-9.]+)',
                'wbc': r'(?:white blood cells?|WBC)[:\s]+([0-9.]+)',
                'sodium': r'(?:sodium|Na)[:\s]+([0-9.]+)',
                'potassium': r'(?:potassium|K)[:\s]+([0-9.]+)'
            }
            
            for lab, pattern in patterns.items():
                match = re.search(pattern, lab_text, re.IGNORECASE)
                if match:
                    critical_labs[lab] = float(match.group(1))
            
            return critical_labs
        
        df['critical_labs'] = df['lab_summary'].apply(extract_critical_labs)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        logger.info("Handling missing values...")
        
        # Text fields - fill with empty string
        text_columns = ['cleaned_text', 'chief_complaint', 'history_present_illness', 
                       'past_medical_history', 'medications', 'discharge_diagnosis',
                       'discharge_medications', 'follow_up', 'physical_exam', 'hospital_course']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Numeric fields - fill with 0 or median
        if 'abnormal_count' in df.columns:
            df['abnormal_count'] = df['abnormal_count'].fillna(0)
        
        if 'text_length' in df.columns:
            df['text_length'] = df['text_length'].fillna(df['text_length'].median())
        
        return df
    
    def create_summary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary features for ML models"""
        logger.info("Creating summary features...")
        
        # Completeness score (how many sections are filled)
        section_cols = ['chief_complaint', 'discharge_diagnosis', 'discharge_medications', 'follow_up']
        df['completeness_score'] = df[section_cols].apply(
            lambda row: sum([1 for val in row if len(str(val)) > 10]) / len(section_cols), 
            axis=1
        )
        
        # Urgency indicators
        urgent_terms = r'\b(?:urgent|emergency|immediate|critical|severe|acute)\b'
        df['urgency_indicator'] = df['cleaned_text'].str.contains(urgent_terms, case=False, regex=True).astype(int)
        
        # Complexity score based on length and medical terms
        df['complexity_score'] = (
            df['text_length'] / df['text_length'].max() * 0.5 +
            df['medical_term_count'] / df['medical_term_count'].max() * 0.5
        )
        
        return df
    
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_data()
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape}")
        
        # Extract sections from discharge summaries
        logger.info("Extracting sections from discharge summaries...")
        sections_data = df['cleaned_text'].apply(self.extract_sections)
        sections_df = pd.DataFrame(sections_data.tolist())
        df = pd.concat([df, sections_df], axis=1)
        
        # Expand abbreviations in key sections
        logger.info("Expanding medical abbreviations...")
        for col in ['discharge_diagnosis', 'discharge_medications', 'chief_complaint']:
            if col in df.columns:
                df[f'{col}_expanded'] = df[col].apply(self.expand_abbreviations)
        
        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text_final'] = df['cleaned_text'].apply(self.clean_text)
        
        # Calculate features
        df = self.calculate_text_features(df)
        df = self.process_lab_summary(df)
        df = self.handle_missing_values(df)
        df = self.create_summary_features(df)
        
        # Save processed data
        output_file = os.path.join(self.output_path, 'processed_discharge_summaries.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Generate processing report
        report = {
            'initial_records': initial_shape[0],
            'initial_columns': initial_shape[1],
            'final_records': df.shape[0],
            'final_columns': df.shape[1],
            'missing_text_count': (df['cleaned_text'] == '').sum(),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'records_with_labs': df['has_labs'].sum(),
            'records_with_medications': df['has_medications'].sum(),
            'records_with_follow_up': df['has_follow_up'].sum()
        }
        
        # Save report
        report_df = pd.DataFrame([report])
        report_file = os.path.join(self.output_path, 'preprocessing_report.csv')
        report_df.to_csv(report_file, index=False)
        logger.info(f"Saved preprocessing report to {report_file}")
        
        return df, report

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = MIMICPreprocessor(
        input_path='data/raw',
        output_path='data/processed'
    )
    
    df_processed, report = preprocessor.run_preprocessing_pipeline()
    
    print("\n=== Preprocessing Complete ===")
    print(f"Records processed: {report['initial_records']}")
    print(f"Features created: {report['final_columns'] - report['initial_columns']} new columns")
    print(f"Average text length: {report['avg_text_length']:.0f} characters")
    print(f"Records with labs: {report['records_with_labs']}")
    print(f"Data saved to: data/processed/processed_discharge_summaries.csv")
# This script handles all the preprocessing needs:

# Section extraction
# Abbreviation expansion
# Text cleaning
# Feature engineering
# Lab data processing
# Missing value handling
# Summary statistics generation

# The processed data will be ready for validation, bias detection, and ML model training.