"""
Bias Detection Pipeline for MIMIC-III Data
Description: Detects and analyzes potential biases across demographic groups in medical documentation
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from scipy import stats
from datetime import datetime
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger, log_data_operation
from utils.error_handling import (
    BiasDetectionError, ErrorHandler, safe_execute, 
    validate_dataframe, validate_file_path, ErrorContext
)

# Set up logging
logger = get_logger(__name__)


class MIMICBiasDetector:
    """Comprehensive bias detection for MIMIC-III medical records across demographic groups"""
    
    def __init__(self, input_path: str = 'data/processed', output_path: str = 'logs', config: Dict = None):
        """
        Initialize bias detector with paths and configuration
        
        Args:
            input_path: Path to processed/features data directory
            output_path: Path to save bias reports and visualizations
            config: Configuration dictionary with bias detection rules
        """
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        try:
            # Create output directories
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, 'bias_plots'), exist_ok=True)
            logger.info(f"Initialized bias detector with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # Load bias detection thresholds from config or use defaults
        if config and 'bias_detection_config' in config:
            self.thresholds = config['bias_detection_config'].get('alert_thresholds', {})
        else:
            self.thresholds = {
                'gender_cv_max': 5.0,
                'ethnicity_cv_max': 10.0,
                'age_cv_max': 8.0,
                'overall_bias_score_max': 10.0,
                'min_sample_size': 30,
                'significance_threshold': 0.05
            }
        
        logger.info(f"Using bias detection thresholds: {self.thresholds}")
    
    @safe_execute("load_data", logger, ErrorHandler(logger))
    @log_data_operation(logger, "load_data")
    def load_data(self, filename: str = 'mimic_features.csv') -> pd.DataFrame:
        """
        Load feature-engineered data for bias detection
        
        Args:
            filename: Name of features CSV file
            
        Returns:
            DataFrame with engineered features and demographics
        """
        filepath = os.path.join(self.input_path, filename)
        
        validate_file_path(filepath, logger, must_exist=True)
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate DataFrame structure
        required_columns = ['hadm_id', 'subject_id']
        validate_dataframe(df, required_columns, logger)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features for bias detection")
        return df
    
    def detect_documentation_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect bias in documentation length, quality, and completeness
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with documentation bias analysis results
        """
        bias_report = {
            'documentation_length_bias': {},
            'documentation_quality_bias': {},
            'section_completeness_bias': {}
        }
        
        # Check for required columns
        if 'text_chars' not in df.columns:
            logger.warning("text_chars column not found, skipping documentation length analysis")
            return bias_report
        
        # Analyze documentation length by gender
        if 'gender' in df.columns:
            gender_stats = df.groupby('gender').agg({
                'text_chars': ['mean', 'median', 'std', 'count']
            })
            bias_report['documentation_length_bias']['by_gender'] = json.loads(gender_stats.to_json())
            
            # Statistical test for gender differences
            unique_genders = df['gender'].dropna().unique()
            if len(unique_genders) >= 2:
                male_lengths = df[df['gender'] == 'M']['text_chars'].dropna()
                female_lengths = df[df['gender'] == 'F']['text_chars'].dropna()
                
                if len(male_lengths) >= self.thresholds['min_sample_size'] and \
                   len(female_lengths) >= self.thresholds['min_sample_size']:
                    t_stat, p_value = stats.ttest_ind(male_lengths, female_lengths)
                    bias_report['documentation_length_bias']['gender_ttest'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.thresholds['significance_threshold'],
                        'mean_difference': float(male_lengths.mean() - female_lengths.mean())
                    }
        
        # Analyze documentation length by ethnicity
        if 'ethnicity_clean' in df.columns:
            ethnicity_stats = df.groupby('ethnicity_clean').agg({
                'text_chars': ['mean', 'median', 'std', 'count']
            })
            bias_report['documentation_length_bias']['by_ethnicity'] = json.loads(ethnicity_stats.to_json())
            
            # ANOVA test for ethnicity differences
            ethnicity_groups = [
                group['text_chars'].dropna().values 
                for name, group in df.groupby('ethnicity_clean')
                if len(group['text_chars'].dropna()) >= self.thresholds['min_sample_size']
            ]
            
            if len(ethnicity_groups) > 1:
                f_stat, p_value = stats.f_oneway(*ethnicity_groups)
                bias_report['documentation_length_bias']['ethnicity_anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < self.thresholds['significance_threshold']
                }
        
        # Analyze documentation length by age group
        if 'age_group' in df.columns:
            age_stats = df.groupby('age_group').agg({
                'text_chars': ['mean', 'median', 'std', 'count']
            })
            bias_report['documentation_length_bias']['by_age'] = json.loads(age_stats.to_json())
        
        # Analyze documentation quality metrics
        if 'documentation_completeness' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    quality_stats = df.groupby(demo_col).agg({
                        'documentation_completeness': ['mean', 'std', 'count']
                    })
                    bias_report['documentation_quality_bias'][demo_col] = json.loads(quality_stats.to_json())
        
        return bias_report
    
    def detect_clinical_risk_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect bias in clinical risk assessment and scoring
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with clinical risk bias analysis results
        """
        risk_bias_report = {
            'high_risk_scoring': {},
            'risk_outcome_patterns': {},
            'acuity_assessment': {}
        }
        
        # Analyze high risk scoring across demographics
        if 'high_risk_score' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    risk_stats = df.groupby(demo_col).agg({
                        'high_risk_score': ['mean', 'median', 'std', 'count']
                    })
                    risk_bias_report['high_risk_scoring'][demo_col] = json.loads(risk_stats.to_json())
        
        # Analyze risk-outcome ratio patterns
        if 'risk_outcome_ratio' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    ratio_stats = df.groupby(demo_col).agg({
                        'risk_outcome_ratio': ['mean', 'median', 'count']
                    })
                    risk_bias_report['risk_outcome_patterns'][demo_col] = json.loads(ratio_stats.to_json())
        
        # Analyze acute vs chronic presentation patterns
        if 'acute_chronic_ratio' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    acuity_stats = df.groupby(demo_col).agg({
                        'acute_chronic_ratio': ['mean', 'median', 'count']
                    })
                    risk_bias_report['acuity_assessment'][demo_col] = json.loads(acuity_stats.to_json())
        
        return risk_bias_report
    
    def detect_treatment_complexity_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect bias in treatment complexity and intensity measures
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with treatment complexity bias analysis results
        """
        treatment_bias = {
            'polypharmacy_patterns': {},
            'treatment_intensity': {},
            'medication_patterns': {}
        }
        
        # Analyze polypharmacy flag distribution
        if 'polypharmacy_flag' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    polypharm_rate = df.groupby(demo_col)['polypharmacy_flag'].mean()
                    treatment_bias['polypharmacy_patterns'][demo_col] = json.loads(polypharm_rate.to_json())
        
        # Analyze treatment intensity scores
        if 'treatment_intensity' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    intensity_stats = df.groupby(demo_col).agg({
                        'treatment_intensity': ['mean', 'median', 'std', 'count']
                    })
                    treatment_bias['treatment_intensity'][demo_col] = json.loads(intensity_stats.to_json())
        
        # Analyze medication keyword patterns
        if 'kw_medications' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    med_stats = df.groupby(demo_col).agg({
                        'kw_medications': ['mean', 'median', 'count']
                    })
                    treatment_bias['medication_patterns'][demo_col] = json.loads(med_stats.to_json())
        
        return treatment_bias
    
    def detect_lab_testing_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect bias in laboratory testing patterns and abnormalities
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with lab testing bias analysis results
        """
        lab_bias_report = {
            'testing_frequency': {},
            'abnormal_result_patterns': {},
            'lab_ratio_patterns': {}
        }
        
        # Analyze total lab testing patterns
        if 'total_labs' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    lab_stats = df.groupby(demo_col).agg({
                        'total_labs': ['mean', 'median', 'std', 'count']
                    })
                    lab_bias_report['testing_frequency'][demo_col] = json.loads(lab_stats.to_json())
        
        # Analyze abnormal lab count patterns
        if 'abnormal_lab_count' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    abnormal_stats = df.groupby(demo_col).agg({
                        'abnormal_lab_count': ['mean', 'median', 'count']
                    })
                    lab_bias_report['abnormal_result_patterns'][demo_col] = json.loads(abnormal_stats.to_json())
        
        # Analyze abnormal lab ratio
        if 'abnormal_lab_ratio' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    ratio_stats = df.groupby(demo_col).agg({
                        'abnormal_lab_ratio': ['mean', 'median', 'count']
                    })
                    lab_bias_report['lab_ratio_patterns'][demo_col] = json.loads(ratio_stats.to_json())
        
        return lab_bias_report
    
    def detect_readability_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect bias in documentation readability and complexity
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with readability bias analysis results
        """
        readability_bias = {
            'flesch_score_patterns': {},
            'vocabulary_patterns': {},
            'medical_density_patterns': {}
        }
        
        # Analyze Flesch Reading Ease scores
        if 'flesch_reading_ease' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    flesch_stats = df.groupby(demo_col).agg({
                        'flesch_reading_ease': ['mean', 'median', 'std', 'count']
                    })
                    readability_bias['flesch_score_patterns'][demo_col] = json.loads(flesch_stats.to_json())
        
        # Analyze vocabulary richness
        if 'vocabulary_richness' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    vocab_stats = df.groupby(demo_col).agg({
                        'vocabulary_richness': ['mean', 'median', 'count']
                    })
                    readability_bias['vocabulary_patterns'][demo_col] = json.loads(vocab_stats.to_json())
        
        # Analyze medical term density
        if 'disease_density' in df.columns:
            for demo_col in ['gender', 'ethnicity_clean', 'age_group']:
                if demo_col in df.columns:
                    density_stats = df.groupby(demo_col).agg({
                        'disease_density': ['mean', 'median', 'count']
                    })
                    readability_bias['medical_density_patterns'][demo_col] = json.loads(density_stats.to_json())
        
        return readability_bias
    
    def calculate_bias_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary bias metrics using coefficient of variation
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary with summary bias metrics
        """
        metrics = {}
        
        # Calculate coefficient of variation for text length across gender
        if 'gender' in df.columns and 'text_chars' in df.columns:
            gender_means = df.groupby('gender')['text_chars'].mean()
            if len(gender_means) > 1 and gender_means.mean() > 0:
                cv = (gender_means.std() / gender_means.mean()) * 100
                metrics['gender_cv'] = float(cv)
                metrics['gender_bias_alert'] = cv > self.thresholds['gender_cv_max']
        
        # Calculate coefficient of variation for text length across ethnicity
        if 'ethnicity_clean' in df.columns and 'text_chars' in df.columns:
            ethnicity_means = df.groupby('ethnicity_clean')['text_chars'].mean()
            if len(ethnicity_means) > 1 and ethnicity_means.mean() > 0:
                cv = (ethnicity_means.std() / ethnicity_means.mean()) * 100
                metrics['ethnicity_cv'] = float(cv)
                metrics['ethnicity_bias_alert'] = cv > self.thresholds['ethnicity_cv_max']
        
        # Calculate coefficient of variation for text length across age groups
        if 'age_group' in df.columns and 'text_chars' in df.columns:
            age_means = df.groupby('age_group')['text_chars'].mean()
            if len(age_means) > 1 and age_means.mean() > 0:
                cv = (age_means.std() / age_means.mean()) * 100
                metrics['age_cv'] = float(cv)
                metrics['age_bias_alert'] = cv > self.thresholds['age_cv_max']
        
        # Calculate overall bias score (average of CVs, lower is better)
        cv_values = [v for k, v in metrics.items() if k.endswith('_cv')]
        if cv_values:
            overall_score = float(np.mean(cv_values))
            metrics['overall_bias_score'] = overall_score
            metrics['overall_bias_alert'] = overall_score > self.thresholds['overall_bias_score_max']
        
        return metrics
    
    def create_bias_visualizations(self, df: pd.DataFrame):
        """
        Create visualizations for bias patterns across demographics
        
        Args:
            df: Input DataFrame with features
        """
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
        plots_dir = os.path.join(self.output_path, 'bias_plots')
        
        # 1. Documentation length by gender
        if 'gender' in df.columns and 'text_chars' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='text_chars', by='gender', ax=ax)
            ax.set_title('Documentation Length by Gender')
            ax.set_ylabel('Text Length (characters)')
            ax.set_xlabel('Gender')
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'text_length_by_gender.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: text_length_by_gender.png")
        
        # 2. Documentation length by ethnicity
        if 'ethnicity_clean' in df.columns and 'text_chars' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            df.boxplot(column='text_chars', by='ethnicity_clean', ax=ax)
            ax.set_title('Documentation Length by Ethnicity')
            ax.set_ylabel('Text Length (characters)')
            ax.set_xlabel('Ethnicity')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'text_length_by_ethnicity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: text_length_by_ethnicity.png")
        
        # 3. Documentation completeness by demographics
        if 'documentation_completeness' in df.columns and 'ethnicity_clean' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            completeness_means = df.groupby('ethnicity_clean')['documentation_completeness'].mean()
            completeness_means.plot(kind='bar', ax=ax)
            ax.set_title('Documentation Completeness by Ethnicity')
            ax.set_ylabel('Completeness Score')
            ax.set_xlabel('Ethnicity')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'completeness_by_ethnicity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: completeness_by_ethnicity.png")
        
        # 4. Treatment intensity by age group
        if 'treatment_intensity' in df.columns and 'age_group' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='treatment_intensity', by='age_group', ax=ax)
            ax.set_title('Treatment Intensity by Age Group')
            ax.set_ylabel('Treatment Intensity Score')
            ax.set_xlabel('Age Group')
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'treatment_intensity_by_age.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: treatment_intensity_by_age.png")
        
        # 5. High risk scoring by gender
        if 'high_risk_score' in df.columns and 'gender' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            risk_means = df.groupby('gender')['high_risk_score'].mean()
            risk_means.plot(kind='bar', ax=ax)
            ax.set_title('Average High Risk Score by Gender')
            ax.set_ylabel('High Risk Score')
            ax.set_xlabel('Gender')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'high_risk_by_gender.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created visualization: high_risk_by_gender.png")
        
        logger.info(f"All bias visualizations saved to {plots_dir}/")
    
    def run_bias_detection_pipeline(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run complete bias detection pipeline
        
        Returns:
            Tuple of (bias report dictionary, summary DataFrame)
        """
        logger.info("="*60)
        logger.info("BIAS DETECTION PIPELINE")
        logger.info("="*60)
        
        # Load feature-engineered data
        df = self.load_data()
        
        # Initialize bias report
        bias_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records_analyzed': len(df),
            'total_features_analyzed': len(df.columns),
            'demographic_distribution': {},
            'bias_thresholds': self.thresholds
        }
        
        # Get demographic distributions
        logger.info("Analyzing demographic distributions...")
        for col in ['gender', 'ethnicity_clean', 'age_group']:
            if col in df.columns:
                distribution = df[col].value_counts().to_dict()
                distribution = {str(k): int(v) for k, v in distribution.items()}
                bias_report['demographic_distribution'][col] = distribution
        
        # Run bias detection analyses
        logger.info("Detecting documentation bias...")
        bias_report['documentation_bias'] = self.detect_documentation_bias(df)
        
        logger.info("Detecting clinical risk assessment bias...")
        bias_report['clinical_risk_bias'] = self.detect_clinical_risk_bias(df)
        
        logger.info("Detecting treatment complexity bias...")
        bias_report['treatment_complexity_bias'] = self.detect_treatment_complexity_bias(df)
        
        logger.info("Detecting lab testing bias...")
        bias_report['lab_testing_bias'] = self.detect_lab_testing_bias(df)
        
        logger.info("Detecting readability and complexity bias...")
        bias_report['readability_bias'] = self.detect_readability_bias(df)
        
        logger.info("Calculating summary bias metrics...")
        bias_report['summary_metrics'] = self.calculate_bias_metrics(df)
        
        # Create visualizations
        logger.info("Creating bias visualizations...")
        self.create_bias_visualizations(df)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            """Recursively convert numpy types to Python native types"""
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
        
        # Create and save summary DataFrame
        summary_df = self.create_bias_summary(bias_report)
        summary_path = os.path.join(self.output_path, 'bias_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Bias summary saved to {summary_path}")
        
        logger.info("="*60)
        logger.info("BIAS DETECTION COMPLETE")
        logger.info("="*60)
        
        return bias_report, summary_df
    
    def create_bias_summary(self, report: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame from bias report for easy review
        
        Args:
            report: Complete bias report dictionary
            
        Returns:
            DataFrame with bias summary metrics
        """
        summary_data = {
            'Bias Category': [],
            'Metric': [],
            'Value': [],
            'Alert': []
        }
        
        # Add summary metrics
        if 'summary_metrics' in report:
            for metric, value in report['summary_metrics'].items():
                if not metric.endswith('_alert'):
                    summary_data['Bias Category'].append('Overall')
                    summary_data['Metric'].append(metric)
                    summary_data['Value'].append(value)
                    
                    alert_key = f"{metric.split('_')[0]}_bias_alert"
                    alert_status = report['summary_metrics'].get(alert_key, False)
                    summary_data['Alert'].append('YES' if alert_status else 'NO')
        
        # Add documentation bias significance tests
        if 'documentation_bias' in report:
            doc_bias = report['documentation_bias']
            if 'documentation_length_bias' in doc_bias:
                if 'gender_ttest' in doc_bias['documentation_length_bias']:
                    gender_test = doc_bias['documentation_length_bias']['gender_ttest']
                    summary_data['Bias Category'].append('Documentation')
                    summary_data['Metric'].append('Gender Difference (p-value)')
                    summary_data['Value'].append(gender_test['p_value'])
                    summary_data['Alert'].append('YES' if gender_test['significant'] else 'NO')
                
                if 'ethnicity_anova' in doc_bias['documentation_length_bias']:
                    eth_test = doc_bias['documentation_length_bias']['ethnicity_anova']
                    summary_data['Bias Category'].append('Documentation')
                    summary_data['Metric'].append('Ethnicity Difference (p-value)')
                    summary_data['Value'].append(eth_test['p_value'])
                    summary_data['Alert'].append('YES' if eth_test['significant'] else 'NO')
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    import json
    
    # Load configuration from project
    config_path = os.path.join(os.path.dirname(__file__), '../configs/pipeline_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize bias detector with config paths and rules
    detector = MIMICBiasDetector(
        input_path=config['pipeline_config']['output_path'],
        output_path=config['pipeline_config']['logs_path'],
        config=config
    )
    
    # Run bias detection pipeline
    report, summary = detector.run_bias_detection_pipeline()
    
    # Print results summary
    print("\n" + "="*60)
    print("BIAS DETECTION RESULTS")
    print("="*60)
    print(f"Total Records Analyzed: {report['total_records_analyzed']}")
    print(f"Total Features Analyzed: {report['total_features_analyzed']}")
    
    if 'summary_metrics' in report:
        print("\nBias Metrics:")
        if 'overall_bias_score' in report['summary_metrics']:
            score = report['summary_metrics']['overall_bias_score']
            alert = report['summary_metrics'].get('overall_bias_alert', False)
            print(f"  Overall Bias Score: {score:.2f} {'[ALERT]' if alert else '[OK]'}")
        
        if 'gender_cv' in report['summary_metrics']:
            cv = report['summary_metrics']['gender_cv']
            alert = report['summary_metrics'].get('gender_bias_alert', False)
            print(f"  Gender CV: {cv:.2f}% {'[ALERT]' if alert else '[OK]'}")
        
        if 'ethnicity_cv' in report['summary_metrics']:
            cv = report['summary_metrics']['ethnicity_cv']
            alert = report['summary_metrics'].get('ethnicity_bias_alert', False)
            print(f"  Ethnicity CV: {cv:.2f}% {'[ALERT]' if alert else '[OK]'}")
    
    print("\nOutput Files:")
    print(f"  Bias Report: {config['pipeline_config']['logs_path']}/bias_report.json")
    print(f"  Bias Summary: {config['pipeline_config']['logs_path']}/bias_summary.csv")
    print(f"  Visualizations: {config['pipeline_config']['logs_path']}/bias_plots/")
    print("="*60)
    
    # Display summary table
    print("\nBias Summary Table:")
    print(summary.to_string(index=False))