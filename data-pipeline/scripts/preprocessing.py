# Data Preprocessing Pipeline for MIMIC-III Discharge Summaries
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
        """
        Initialize the preprocessor with input and output paths
        
        Args:
            input_path: Path to raw data directory
            output_path: Path to save processed data
        """
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Initialized preprocessor with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # Expanded medical abbreviations dictionary for better text processing
        self.medical_abbrev = {
            # Basic medical terms
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
            
            # Dosing and administration
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'qd': 'once daily',
            'prn': 'as needed',
            'qhs': 'at bedtime',
            'po': 'by mouth',
            'iv': 'intravenous',
            'pod': 'post operative day',
            
            # Common diagnoses
            'mi': 'myocardial infarction',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'cad': 'coronary artery disease',
            'ckd': 'chronic kidney disease',
            'afib': 'atrial fibrillation',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'uti': 'urinary tract infection',
            
            # Medical terms and procedures
            'dx': 'diagnosis',
            'tx': 'treatment',
            'sx': 'symptoms',
            'rx': 'prescription',
            'fx': 'fracture',
            'abd': 'abdomen',
            'gi': 'gastrointestinal',
            'cv': 'cardiovascular',
            'bpm': 'beats per minute',
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'wt': 'weight',
            
            # Lab tests and imaging
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'ekg': 'electrocardiogram',
            'cxr': 'chest x-ray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'us': 'ultrasound',
            'echo': 'echocardiogram'
        }
        
    @safe_execute("load_data", logger, ErrorHandler(logger))
    @log_data_operation(logger, "load_data")
    def load_data(self, filename: str = 'mimic_discharge_labs.csv') -> pd.DataFrame:
        """
        Load the raw data from CSV file
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame containing the raw data
        """
        filepath = os.path.join(self.input_path, filename)
        
        # Validate file exists before loading
        validate_file_path(filepath, logger, must_exist=True)
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate DataFrame has required columns
        required_columns = ['hadm_id', 'subject_id', 'cleaned_text']
        validate_dataframe(df, required_columns, logger)
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records based on admission ID
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicates...")
        initial_count = len(df)
        
        # Keep first occurrence of each admission ID
        df = df.drop_duplicates(subset=['hadm_id'], keep='first')
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} duplicate records")
        
        return df
    
    def standardize_ethnicity(self, ethnicity: str) -> str:
        """
        Standardize ethnicity values from MIMIC-III variations to consistent categories
        
        Args:
            ethnicity: Raw ethnicity string from MIMIC-III
            
        Returns:
            Standardized ethnicity category
        """
        if pd.isna(ethnicity):
            return 'UNKNOWN'
        
        # Convert to uppercase for comparison
        eth_upper = str(ethnicity).upper()
        
        # Map variations to standard categories
        if 'WHITE' in eth_upper:
            return 'WHITE'
        elif 'BLACK' in eth_upper or 'AFRICAN' in eth_upper:
            return 'BLACK'
        elif 'HISPANIC' in eth_upper or 'LATINO' in eth_upper:
            return 'HISPANIC'
        elif 'ASIAN' in eth_upper:
            return 'ASIAN'
        else:
            return 'OTHER'
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create standardized demographic features for bias detection and analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added demographic features
        """
        logger.info("Creating demographic features...")
        
        # Standardize ethnicity categories
        df['ethnicity_clean'] = df['ethnicity'].apply(self.standardize_ethnicity)
        
        # Create age groups for fairness analysis
        df['age_group'] = pd.cut(
            df['age_at_admission'], 
            bins=[0, 18, 35, 50, 65, 120],
            labels=['<18', '18-35', '35-50', '50-65', '65+'],
            include_lowest=True
        )
        
        # Standardize gender values
        df['gender'] = df['gender'].str.upper().fillna('UNKNOWN')
        
        # Log distribution for verification
        logger.info(f"Ethnicity distribution: {df['ethnicity_clean'].value_counts().to_dict()}")
        logger.info(f"Age group distribution: {df['age_group'].value_counts().to_dict()}")
        
        return df
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from discharge summary using regex patterns
        
        Args:
            text: Full discharge summary text
            
        Returns:
            Dictionary with extracted section contents
        """
        # Initialize empty sections
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
        
        # Regex patterns for common discharge summary sections
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
        
        # Extract each section using pattern matching
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common medical abbreviations to full terms
        
        Args:
            text: Text containing medical abbreviations
            
        Returns:
            Text with abbreviations expanded
        """
        if pd.isna(text):
            return text
        
        expanded = text
        
        # Replace each abbreviation with full form using word boundaries
        for abbrev, full_form in self.medical_abbrev.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded = re.sub(pattern, full_form, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def clean_text(self, text: str) -> str:
        """
        Additional text cleaning beyond de-identification
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Fix common OCR and typing errors
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', text)
        
        # Remove isolated special characters
        text = re.sub(r'\s[^\w\s]\s', ' ', text)
        
        return text.strip()
    
    def calculate_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate text-based features for analysis and modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added text features
        """
        logger.info("Calculating text features...")
        
        # Basic text statistics
        df['text_length'] = df['cleaned_text'].fillna('').str.len()
        df['word_count'] = df['cleaned_text'].fillna('').str.split().str.len()
        df['sentence_count'] = df['cleaned_text'].fillna('').str.split('[.!?]').str.len()
        
        # Average word length as complexity indicator
        df['avg_word_length'] = df['cleaned_text'].fillna('').apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        # Count medical terminology as complexity indicator
        df['medical_term_count'] = df['cleaned_text'].fillna('').str.count(
            r'\b(?:diagnosis|medication|symptom|procedure|laboratory|imaging)\b', 
            flags=re.IGNORECASE
        )
        
        # Check presence of key sections
        df['has_medications'] = df['discharge_medications'].fillna('').str.len() > 10
        df['has_follow_up'] = df['follow_up'].fillna('').str.len() > 10
        df['has_diagnosis'] = df['discharge_diagnosis'].fillna('').str.len() > 10
        
        return df
    
    def process_lab_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and structure lab summary data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed lab features
        """
        logger.info("Processing lab summaries...")
        
        # Flag records with lab data
        df['has_labs'] = ~df['lab_summary'].isna()
        df['lab_summary_clean'] = df['lab_summary'].fillna('No lab results available')
        
        def extract_critical_labs(lab_text):
            """Extract specific lab values from summary text"""
            if pd.isna(lab_text):
                return {}
            
            critical_labs = {}
            
            # Patterns for common critical lab values
            patterns = {
                'creatinine': r'(?:creatinine|Cr)[:\s]+([0-9.]+)',
                'glucose': r'(?:glucose|Glu)[:\s]+([0-9.]+)',
                'hemoglobin': r'(?:hemoglobin|Hgb|Hb)[:\s]+([0-9.]+)',
                'wbc': r'(?:white blood cells?|WBC)[:\s]+([0-9.]+)',
                'sodium': r'(?:sodium|Na)[:\s]+([0-9.]+)',
                'potassium': r'(?:potassium|K)[:\s]+([0-9.]+)'
            }
            
            # Extract each lab value if present
            for lab, pattern in patterns.items():
                match = re.search(pattern, lab_text, re.IGNORECASE)
                if match:
                    critical_labs[lab] = float(match.group(1))
            
            return critical_labs
        
        # Apply extraction to all lab summaries
        df['critical_labs'] = df['lab_summary'].apply(extract_critical_labs)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values appropriately based on data type
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        # Text columns - fill with empty string
        text_columns = ['cleaned_text', 'chief_complaint', 'history_present_illness', 
                       'past_medical_history', 'medications', 'discharge_diagnosis',
                       'discharge_medications', 'follow_up', 'physical_exam', 'hospital_course']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Numeric columns - fill appropriately
        if 'abnormal_count' in df.columns:
            df['abnormal_count'] = df['abnormal_count'].fillna(0)
        
        if 'text_length' in df.columns:
            df['text_length'] = df['text_length'].fillna(df['text_length'].median())
        
        return df
    
    def create_summary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary features for ML models
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added summary features
        """
        logger.info("Creating summary features...")
        
        # Completeness score - how many key sections are present
        section_cols = ['chief_complaint', 'discharge_diagnosis', 'discharge_medications', 'follow_up']
        df['completeness_score'] = df[section_cols].apply(
            lambda row: sum([1 for val in row if len(str(val)) > 10]) / len(section_cols), 
            axis=1
        )
        
        # Urgency indicator based on keywords
        urgent_terms = r'\b(?:urgent|emergency|immediate|critical|severe|acute)\b'
        df['urgency_indicator'] = df['cleaned_text'].str.contains(urgent_terms, case=False, regex=True).astype(int)
        
        # Complexity score combining length and medical terminology
        df['complexity_score'] = (
            df['text_length'] / df['text_length'].max() * 0.5 +
            df['medical_term_count'] / df['medical_term_count'].max() * 0.5
        )
        
        return df
    
    def run_preprocessing_pipeline(self):
        """
        Run the complete preprocessing pipeline
        
        Returns:
            Tuple of (processed DataFrame, processing report dictionary)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Load raw data
        df = self.load_data()
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape}")
        
        # Step 2: Remove duplicate records and filter invalid entries 
        df = self.remove_duplicates(df)

        # Filter out empty text records
        df = df[df['cleaned_text'].str.len() >= 100]

        # Filter invalid ages
        df = df[(df['age_at_admission'] >= 0) & (df['age_at_admission'] <= 120)]
        
        # Step 3: Create standardized demographic features
        df = self.create_demographic_features(df)
        
        # Step 4: Extract sections from discharge summaries
        logger.info("Extracting sections from discharge summaries...")
        sections_data = df['cleaned_text'].apply(self.extract_sections)
        sections_df = pd.DataFrame(sections_data.tolist())
        df = pd.concat([df, sections_df], axis=1)
        
        # Step 5: Expand medical abbreviations in key sections
        logger.info("Expanding medical abbreviations...")
        for col in ['discharge_diagnosis', 'discharge_medications', 'chief_complaint']:
            if col in df.columns:
                df[f'{col}_expanded'] = df[col].apply(self.expand_abbreviations)
        
        # Step 6: Clean text
        logger.info("Cleaning text...")
        df['cleaned_text_final'] = df['cleaned_text'].apply(self.clean_text)
        
        # Step 7: Calculate text features
        df = self.calculate_text_features(df)
        
        # Step 8: Process lab summaries
        df = self.process_lab_summary(df)
        
        # Step 9: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 10: Create summary features
        df = self.create_summary_features(df)
        
        # Step 11: Save processed data
        output_file = os.path.join(self.output_path, 'processed_discharge_summaries.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Step 12: Generate processing report
        report = {
            'initial_records': initial_shape[0],
            'initial_columns': initial_shape[1],
            'final_records': df.shape[0],
            'final_columns': df.shape[1],
            'duplicates_removed': initial_shape[0] - df.shape[0],
            'missing_text_count': (df['cleaned_text'] == '').sum(),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'records_with_labs': df['has_labs'].sum(),
            'records_with_medications': df['has_medications'].sum(),
            'records_with_follow_up': df['has_follow_up'].sum()
        }
        
        # Step 13: Save processing report
        report_df = pd.DataFrame([report])
        report_file = os.path.join(self.output_path, 'preprocessing_report.csv')
        report_df.to_csv(report_file, index=False)
        logger.info(f"Saved preprocessing report to {report_file}")
        
        return df, report


if __name__ == "__main__":
    import json
    
    # Load configuration from project
    config_path = os.path.join(os.path.dirname(__file__), '../configs/pipeline_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize preprocessor with config paths
    preprocessor = MIMICPreprocessor(
        input_path=config['pipeline_config']['input_path'],
        output_path=config['pipeline_config']['output_path']
    )
    
    # Run preprocessing pipeline
    df_processed, report = preprocessor.run_preprocessing_pipeline()
    
    # Print summary
    print("\n=== Preprocessing Complete ===")
    print(f"Records processed: {report['initial_records']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Features created: {report['final_columns'] - report['initial_columns']} new columns")
    print(f"Average text length: {report['avg_text_length']:.0f} characters")
    print(f"Records with labs: {report['records_with_labs']}")
    print(f"Data saved to: {config['pipeline_config']['output_path']}/processed_discharge_summaries.csv")