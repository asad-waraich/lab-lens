"""
Bias Detection Pipeline for MIMIC-III Data
Description: Detects and analyzes potential biases in medical documentation
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from scipy import stats
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIMICBiasDetector:
    """Comprehensive bias detection for MIMIC-III medical records"""
    
    def __init__(self, input_path: str = 'data/raw', output_path: str = 'logs'):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'bias_plots'), exist_ok=True)
        
        # Define demographic groups for analysis
        self.demographic_groups = {
            'gender': ['M', 'F'],
            'ethnicity_groups': {
                'white': ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN'],
                'black': ['BLACK/AFRICAN AMERICAN', 'BLACK/AFRICAN', 'BLACK/CAPE VERDEAN'],
                'hispanic': ['HISPANIC OR LATINO', 'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC/LATINO - DOMINICAN'],
                'asian': ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - ASIAN INDIAN', 'ASIAN - VIETNAMESE'],
                'other': ['OTHER', 'UNKNOWN/NOT SPECIFIED', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER']
            },
            'age_bins': [(18, 35), (35, 50), (50, 65), (65, 80), (80, 120)],
            'insurance_categories': {
                'private': ['Private'],
                'public': ['Medicare', 'Medicaid'],
                'self_pay': ['Self Pay'],
                'government': ['Government']
            }
        }
    
    def load_data(self, filename: str = 'mimic_complete_with_demographics.csv') -> pd.DataFrame:
        """Load data with demographics"""
        filepath = os.path.join(self.input_path, filename)
        
        # Check if demographic data exists, otherwise load basic data
        if not os.path.exists(filepath):
            logger.warning(f"Demographics file not found, loading basic data")
            filepath = os.path.join(self.input_path, 'mimic_discharge_labs.csv')
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records for bias detection")
        return df
    
    def categorize_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize demographics for bias analysis"""
        # Simplify ethnicity if present
        if 'ethnicity' in df.columns:
            df['ethnicity_group'] = 'other'
            for group, ethnicities in self.demographic_groups['ethnicity_groups'].items():
                # Handle NaN values in ethnicity
                mask = df['ethnicity'].notna() & df['ethnicity'].str.upper().isin([e.upper() for e in ethnicities])
                df.loc[mask, 'ethnicity_group'] = group
        
        # Create age groups if age present
        if 'age_at_admission' in df.columns:
            # Handle potential NaN values in age
            df = df[df['age_at_admission'].notna()]
            df['age_group'] = pd.cut(
                df['age_at_admission'], 
                bins=[18, 35, 50, 65, 80, 120],
                labels=['18-35', '36-50', '51-65', '66-80', '80+']
            )
        
        # Simplify insurance if present
        if 'insurance' in df.columns:
            df['insurance_type'] = df['insurance'].apply(self.categorize_insurance)
        
        return df
    
    def categorize_insurance(self, insurance: str) -> str:
        """Categorize insurance into simplified groups"""
        if pd.isna(insurance):
            return 'unknown'
        insurance_lower = insurance.lower()
        if 'medicare' in insurance_lower:
            return 'Medicare'
        elif 'medicaid' in insurance_lower:
            return 'Medicaid'
        elif 'private' in insurance_lower:
            return 'Private'
        elif 'self' in insurance_lower:
            return 'Self Pay'
        elif 'government' in insurance_lower:
            return 'Government'
        else:
            return 'Other'
    
    def detect_documentation_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias in documentation length and quality"""
        bias_report = {
            'documentation_length_bias': {},
            'section_completeness_bias': {},
            'medical_attention_bias': {}
        }
        
        # Analyze by gender if available
        if 'gender' in df.columns and 'text_length' in df.columns:
            # Build aggregation dict based on available columns
            agg_dict = {'text_length': ['mean', 'median', 'std']}
            
            # Only add columns that exist
            if 'word_count' in df.columns:
                agg_dict['word_count'] = ['mean', 'median']
            
            gender_stats = df.groupby('gender').agg(agg_dict)
            # bias_report['documentation_length_bias']['by_gender'] = gender_stats.to_dict()
            bias_report['documentation_length_bias']['by_gender'] = json.loads(gender_stats.to_json())
            
            # Statistical test for significance
            unique_genders = df['gender'].dropna().unique()
            if len(unique_genders) == 2:
                male_lengths = df[df['gender'] == 'M']['text_length'].dropna()
                female_lengths = df[df['gender'] == 'F']['text_length'].dropna()
                if len(male_lengths) > 0 and len(female_lengths) > 0:
                    t_stat, p_value = stats.ttest_ind(male_lengths, female_lengths)
                    bias_report['documentation_length_bias']['gender_ttest'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
        
        # Analyze by ethnicity if available
        if 'ethnicity_group' in df.columns and 'text_length' in df.columns:
            ethnicity_stats = df.groupby('ethnicity_group').agg({
                'text_length': ['mean', 'median', 'std', 'count']
            })
            bias_report['documentation_length_bias']['by_ethnicity'] = json.loads(ethnicity_stats.to_json())
            
            # ANOVA test for multiple groups
            ethnicity_groups = [group['text_length'].dropna().values 
                               for name, group in df.groupby('ethnicity_group')
                               if len(group['text_length'].dropna()) > 0]
            if len(ethnicity_groups) > 1:
                f_stat, p_value = stats.f_oneway(*ethnicity_groups)
                bias_report['documentation_length_bias']['ethnicity_anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # Analyze by age if available
        if 'age_group' in df.columns and 'text_length' in df.columns:
            age_stats = df.groupby('age_group').agg({
                'text_length': ['mean', 'median', 'count']
            })
            # bias_report['documentation_length_bias']['by_age'] = age_stats.to_dict()
            bias_report['documentation_length_bias']['by_age'] = json.loads(age_stats.to_json())

        
        return bias_report
    
    def detect_lab_testing_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias in laboratory testing patterns"""
        lab_bias_report = {
            'testing_frequency_bias': {},
            'abnormal_result_patterns': {}
        }
        
        # Check if we have either of the lab-related columns
        has_lab_data = 'total_labs' in df.columns or 'abnormal_count' in df.columns
        
        if not has_lab_data:
            logger.warning("No lab data available for bias detection")
            return lab_bias_report
        
        # Analyze lab testing by demographics
        demographic_cols = ['gender', 'ethnicity_group', 'age_group', 'insurance_type']
        
        for demo_col in demographic_cols:
            if demo_col in df.columns:
                agg_dict = {}
                
                if 'abnormal_count' in df.columns:
                    agg_dict['abnormal_count'] = ['mean', 'median', 'std']
                
                if 'total_labs' in df.columns:
                    agg_dict['total_labs'] = ['mean', 'median', 'std']
                
                if agg_dict:  # Only aggregate if we have columns to aggregate
                    agg_dict['hadm_id'] = 'count'  # Number of patients
                    lab_stats = df.groupby(demo_col).agg(agg_dict)
                    # lab_bias_report['testing_frequency_bias'][demo_col] = lab_stats.to_dict()
                    lab_bias_report['testing_frequency_bias'][demo_col] = json.loads(lab_stats.to_json())
        
        return lab_bias_report
    
    def detect_section_completeness_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias in which sections are completed for different groups"""
        section_bias = {}
        
        sections_to_check = ['discharge_diagnosis', 'discharge_medications', 'follow_up']
        demographic_cols = ['gender', 'ethnicity_group', 'age_group']
        
        for section in sections_to_check:
            if section in df.columns:
                section_bias[section] = {}
                
                for demo_col in demographic_cols:
                    if demo_col in df.columns:
                        # Calculate completeness rate by group
                        df[f'{section}_complete'] = (~df[section].isna() & (df[section] != ''))
                        completeness = df.groupby(demo_col)[f'{section}_complete'].mean()
                        section_bias[section][demo_col] = json.loads(completeness.to_json())

        return section_bias
    
    def detect_complexity_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias in documentation complexity"""
        complexity_bias = {}
        
        demographic_cols = ['gender', 'ethnicity_group', 'age_group', 'insurance_type']
        
        for demo_col in demographic_cols:
            if demo_col in df.columns:
                agg_dict = {}
                
                if 'medical_term_count' in df.columns:
                    agg_dict['medical_term_count'] = ['mean', 'median']
                
                if 'avg_word_length' in df.columns:
                    agg_dict['avg_word_length'] = ['mean']
                
                # Use text_length as a proxy for complexity if other metrics aren't available
                if not agg_dict and 'text_length' in df.columns:
                    agg_dict['text_length'] = ['mean', 'median']
                
                if agg_dict:
                    complexity_stats = df.groupby(demo_col).agg(agg_dict)
                    # complexity_bias[demo_col] = complexity_stats.to_dict()
                    complexity_bias[demo_col] = json.loads(complexity_stats.to_json())
        
        return complexity_bias
    
    def detect_temporal_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect temporal patterns in documentation"""
        temporal_bias = {}
        
        if 'admission_type' in df.columns and 'text_length' in df.columns:
            admission_stats = df.groupby('admission_type').agg({
                'text_length': ['mean', 'median'],
                'hadm_id': 'count'
            })
            # temporal_bias['by_admission_type'] = admission_stats.to_dict()
            temporal_bias['by_admission_type'] = json.loads(admission_stats.to_json())
        
        return temporal_bias
    
    def calculate_bias_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary bias metrics"""
        metrics = {}
        
        # Calculate coefficient of variation for text length across groups
        if 'gender' in df.columns and 'text_length' in df.columns:
            gender_means = df.groupby('gender')['text_length'].mean()
            if len(gender_means) > 1 and gender_means.mean() > 0:
                metrics['gender_cv'] = (gender_means.std() / gender_means.mean()) * 100
        
        if 'ethnicity_group' in df.columns and 'text_length' in df.columns:
            ethnicity_means = df.groupby('ethnicity_group')['text_length'].mean()
            if len(ethnicity_means) > 1 and ethnicity_means.mean() > 0:
                metrics['ethnicity_cv'] = (ethnicity_means.std() / ethnicity_means.mean()) * 100
        
        if 'age_group' in df.columns and 'text_length' in df.columns:
            age_means = df.groupby('age_group')['text_length'].mean()
            if len(age_means) > 1 and age_means.mean() > 0:
                metrics['age_cv'] = (age_means.std() / age_means.mean()) * 100
        
        # Calculate overall bias score (lower is better)
        cv_values = [v for k, v in metrics.items() if 'cv' in k]
        if cv_values:
            metrics['overall_bias_score'] = np.mean(cv_values)
        
        return metrics
    
    def create_bias_visualizations(self, df: pd.DataFrame):
        """Create visualizations for bias patterns"""
        # Set style
        plt.style.use('default')  # Use default style to avoid issues
        sns.set_palette("husl")
        
        # 1. Text length by gender
        if 'gender' in df.columns and 'text_length' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='text_length', by='gender', ax=ax)
            ax.set_title('Documentation Length by Gender')
            ax.set_ylabel('Text Length (characters)')
            plt.suptitle('')
            plt.savefig(os.path.join(self.output_path, 'bias_plots', 'text_length_by_gender.png'))
            plt.close()
        
        # 2. Text length by ethnicity
        if 'ethnicity_group' in df.columns and 'text_length' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            df.boxplot(column='text_length', by='ethnicity_group', ax=ax)
            ax.set_title('Documentation Length by Ethnicity')
            ax.set_ylabel('Text Length (characters)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'bias_plots', 'text_length_by_ethnicity.png'))
            plt.close()
        
        # 3. Lab testing patterns
        if 'abnormal_count' in df.columns and 'age_group' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.groupby('age_group')['abnormal_count'].mean().plot(kind='bar', ax=ax)
            ax.set_title('Average Abnormal Lab Results by Age Group')
            ax.set_ylabel('Average Abnormal Lab Count')
            ax.set_xlabel('Age Group')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'bias_plots', 'abnormal_labs_by_age.png'))
            plt.close()
        
        logger.info("Bias visualizations saved to logs/bias_plots/")
    
    def run_bias_detection_pipeline(self) -> Tuple[Dict, pd.DataFrame]:
        """Run complete bias detection pipeline"""
        logger.info("Starting bias detection pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Categorize demographics
        df = self.categorize_demographics(df)
        
        # Initialize bias report
        bias_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records_analyzed': len(df),
            'demographic_distribution': {}
        }
        
        # Get demographic distributions
        for col in ['gender', 'ethnicity_group', 'age_group', 'insurance_type']:
            if col in df.columns:
                distribution = df[col].value_counts().to_dict()
                # Convert numpy types to native Python types
                distribution = {str(k): int(v) for k, v in distribution.items()}
                bias_report['demographic_distribution'][col] = distribution
        
        # Run bias detection analyses
        logger.info("Detecting documentation bias...")
        bias_report['documentation_bias'] = self.detect_documentation_bias(df)
        
        logger.info("Detecting lab testing bias...")
        bias_report['lab_testing_bias'] = self.detect_lab_testing_bias(df)
        
        logger.info("Detecting section completeness bias...")
        bias_report['section_completeness_bias'] = self.detect_section_completeness_bias(df)
        
        logger.info("Detecting complexity bias...")
        bias_report['complexity_bias'] = self.detect_complexity_bias(df)
        
        logger.info("Detecting temporal bias...")
        bias_report['temporal_bias'] = self.detect_temporal_bias(df)
        
        logger.info("Calculating bias metrics...")
        bias_report['summary_metrics'] = self.calculate_bias_metrics(df)
        
        # Create visualizations
        logger.info("Creating bias visualizations...")
        self.create_bias_visualizations(df)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        bias_report = convert_numpy(bias_report)
        
        # Save bias report
        report_path = os.path.join(self.output_path, 'bias_report.json')
        with open(report_path, 'w') as f:
            json.dump(bias_report, f, indent=2, default=str)
        logger.info(f"Bias report saved to {report_path}")
        
        # Create summary DataFrame
        summary_df = self.create_bias_summary(bias_report)
        summary_path = os.path.join(self.output_path, 'bias_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Bias summary saved to {summary_path}")
        
        return bias_report, summary_df
    
    def create_bias_summary(self, report: Dict) -> pd.DataFrame:
        """Create summary DataFrame from bias report"""
        summary_data = {
            'Bias Type': [],
            'Metric': [],
            'Value': [],
            'Significance': []
        }
        
        # Documentation bias summary
        if 'documentation_bias' in report:
            doc_bias = report['documentation_bias']
            if 'documentation_length_bias' in doc_bias:
                if 'gender_ttest' in doc_bias['documentation_length_bias']:
                    summary_data['Bias Type'].append('Documentation Length')
                    summary_data['Metric'].append('Gender Difference')
                    summary_data['Value'].append(doc_bias['documentation_length_bias']['gender_ttest']['p_value'])
                    summary_data['Significance'].append('Significant' if doc_bias['documentation_length_bias']['gender_ttest']['significant'] else 'Not Significant')
        
        # Summary metrics
        if 'summary_metrics' in report:
            for metric, value in report['summary_metrics'].items():
                summary_data['Bias Type'].append('Overall')
                summary_data['Metric'].append(metric)
                summary_data['Value'].append(value)
                summary_data['Significance'].append('N/A')
        
        return pd.DataFrame(summary_data)

if __name__ == "__main__":
    # Run bias detection
    detector = MIMICBiasDetector()
    report, summary = detector.run_bias_detection_pipeline()
    
    print("\n=== Bias Detection Complete ===")
    if 'summary_metrics' in report:
        if 'overall_bias_score' in report['summary_metrics']:
            print(f"Overall Bias Score: {report['summary_metrics']['overall_bias_score']:.2f}")
        if 'gender_cv' in report['summary_metrics']:
            print(f"Gender Coefficient of Variation: {report['summary_metrics']['gender_cv']:.2f}%")
    print(f"Total Records Analyzed: {report['total_records_analyzed']}")
    print("\nBias report saved to: logs/bias_report.json")
    print("Visualizations saved to: logs/bias_plots/")