"""
Automated Bias Mitigation Pipeline for MIMIC-III Data
Author: Lab Lens Team
Description: Detects bias and applies mitigation strategies to reduce demographic disparities
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger, log_data_operation
from utils.error_handling import (
    BiasDetectionError, ErrorHandler, safe_execute, 
    validate_dataframe, validate_file_path, ErrorContext
)

# Set up logging
logger = get_logger(__name__)


class AutomatedBiasHandler:
    """Automated bias detection and mitigation system for healthcare data"""
    
    def __init__(self, input_path: str = 'data/processed', output_path: str = 'logs', config: Dict = None):
        """
        Initialize automated bias handler
        
        Args:
            input_path: Path to processed data directory
            output_path: Path to save mitigation reports
            config: Configuration dictionary with bias thresholds
        """
        self.input_path = input_path
        self.output_path = output_path
        self.error_handler = ErrorHandler(logger)
        
        # Create output directories
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Initialized bias handler with input: {input_path}, output: {output_path}")
        except Exception as e:
            raise self.error_handler.handle_file_error("directory_creation", output_path, e)
        
        # Load bias detection thresholds from config
        if config and 'bias_detection_config' in config:
            self.thresholds = config['bias_detection_config'].get('alert_thresholds', {})
            self.mitigation_config = config['bias_detection_config']
        else:
            # Default thresholds
            self.thresholds = {
                'gender_cv_max': 5.0,
                'ethnicity_cv_max': 10.0,
                'age_cv_max': 8.0,
                'overall_bias_score_max': 10.0,
                'min_sample_size': 30,
                'significance_threshold': 0.05
            }
            self.mitigation_config = {
                'auto_mitigation': True,
                'mitigation_enabled': True
            }
        
        logger.info(f"Using bias thresholds: {self.thresholds}")
    
    def load_bias_report(self, filename: str = 'bias_report.json') -> Dict[str, Any]:
        """
        Load bias detection report from previous pipeline step
        
        Args:
            filename: Name of bias report JSON file
            
        Returns:
            Dictionary with bias detection results
        """
        filepath = os.path.join(self.output_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bias report not found at {filepath}. Run bias_detection.py first.")
        
        logger.info(f"Loading bias report from {filepath}")
        with open(filepath, 'r') as f:
            bias_report = json.load(f)
        
        return bias_report
    
    def load_features_data(self, filename: str = 'mimic_features.csv') -> pd.DataFrame:
        """
        Load feature-engineered data for mitigation
        
        Args:
            filename: Name of features CSV file
            
        Returns:
            DataFrame with features
        """
        filepath = os.path.join(self.input_path, filename)
        
        validate_file_path(filepath, logger, must_exist=True)
        
        logger.info(f"Loading features data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def analyze_bias_severity(self, bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze bias severity and determine mitigation strategy
        
        Args:
            bias_report: Bias detection report
            
        Returns:
            Dictionary with severity analysis and recommendations
        """
        analysis = {
            'critical_biases': [],
            'mitigation_needed': False,
            'severity_level': 'none',
            'primary_bias_type': None
        }
        
        # Check overall bias score
        if 'summary_metrics' in bias_report:
            overall_score = bias_report['summary_metrics'].get('overall_bias_score', 0)
            
            if overall_score > self.thresholds['overall_bias_score_max']:
                analysis['mitigation_needed'] = True
                
                # Determine severity level
                if overall_score > 20:
                    analysis['severity_level'] = 'critical'
                elif overall_score > 15:
                    analysis['severity_level'] = 'high'
                elif overall_score > 10:
                    analysis['severity_level'] = 'medium'
                
                analysis['critical_biases'].append({
                    'type': 'overall_bias_score',
                    'value': overall_score,
                    'threshold': self.thresholds['overall_bias_score_max']
                })
            
            # Identify primary bias type (highest CV)
            cv_scores = {}
            for bias_type in ['gender_cv', 'ethnicity_cv', 'age_cv']:
                if bias_type in bias_report['summary_metrics']:
                    cv_value = bias_report['summary_metrics'][bias_type]
                    cv_scores[bias_type] = cv_value
                    
                    threshold_key = f"{bias_type.split('_')[0]}_cv_max"
                    if cv_value > self.thresholds.get(threshold_key, 999):
                        analysis['critical_biases'].append({
                            'type': bias_type,
                            'value': cv_value,
                            'threshold': self.thresholds[threshold_key]
                        })
            
            # Set primary bias type as the one with highest CV
            if cv_scores:
                analysis['primary_bias_type'] = max(cv_scores, key=cv_scores.get)
        
        logger.info(f"Bias severity: {analysis['severity_level']}, Primary bias: {analysis['primary_bias_type']}")
        
        return analysis
    
    def apply_stratified_balancing(self, df: pd.DataFrame, demo_col: str) -> pd.DataFrame:
        """
        Apply stratified balancing to equalize group sizes
        
        Args:
            df: Input DataFrame
            demo_col: Demographic column to balance (gender, ethnicity_clean, age_group)
            
        Returns:
            Balanced DataFrame
        """
        if demo_col not in df.columns:
            logger.warning(f"Column {demo_col} not found, skipping stratified balancing")
            return df
        
        # Get group counts
        group_counts = df[demo_col].value_counts()
        logger.info(f"Original {demo_col} distribution: {group_counts.to_dict()}")
        
        # Calculate target size (average of all groups, or median to avoid outliers)
        target_size = int(group_counts.median())
        
        balanced_groups = []
        
        for group_value in group_counts.index:
            group_data = df[df[demo_col] == group_value]
            current_size = len(group_data)
            
            if current_size > target_size:
                # Downsample larger groups
                sampled = group_data.sample(n=target_size, random_state=42)
                balanced_groups.append(sampled)
            elif current_size < target_size:
                # Upsample smaller groups (duplicate existing records)
                upsampled = group_data.sample(n=target_size, replace=True, random_state=42)
                balanced_groups.append(upsampled)
            else:
                balanced_groups.append(group_data)
        
        balanced_df = pd.concat(balanced_groups, ignore_index=True)
        
        # Verify balancing
        new_counts = balanced_df[demo_col].value_counts()
        logger.info(f"Balanced {demo_col} distribution: {new_counts.to_dict()}")
        logger.info(f"Applied stratified balancing: {len(df)} -> {len(balanced_df)} records")
        
        return balanced_df
    
    def apply_oversampling_minority_groups(self, df: pd.DataFrame, demo_col: str) -> pd.DataFrame:
        """
        Apply oversampling to minority groups to match majority group size
        
        Args:
            df: Input DataFrame
            demo_col: Demographic column to balance
            
        Returns:
            DataFrame with oversampled minority groups
        """
        if demo_col not in df.columns:
            logger.warning(f"Column {demo_col} not found, skipping oversampling")
            return df
        
        # Get group sizes
        group_counts = df[demo_col].value_counts()
        max_size = group_counts.max()
        
        oversampled_groups = []
        
        for group_value in group_counts.index:
            group_data = df[df[demo_col] == group_value]
            current_size = len(group_data)
            
            if current_size < max_size:
                # Oversample to match largest group
                factor = max_size / current_size
                oversampled = group_data.sample(n=max_size, replace=True, random_state=42)
                oversampled_groups.append(oversampled)
                logger.info(f"Oversampled {demo_col}={group_value}: {current_size} -> {max_size} ({factor:.2f}x)")
            else:
                oversampled_groups.append(group_data)
        
        balanced_df = pd.concat(oversampled_groups, ignore_index=True)
        logger.info(f"Applied oversampling: {len(df)} -> {len(balanced_df)} records")
        
        return balanced_df
    
    def apply_demographic_weighting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add demographic weights to balance representation without changing data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with demographic_weight column added
        """
        # Identify available demographic columns
        demo_cols = [col for col in ['gender', 'ethnicity_clean', 'age_group'] if col in df.columns]
        
        if not demo_cols:
            logger.warning("No demographic columns found for weighting")
            return df
        
        # Calculate inverse frequency weights
        weights = []
        
        for idx, row in df.iterrows():
            # Create group key from demographics
            group_values = tuple(row[col] for col in demo_cols if pd.notna(row[col]))
            
            # Find group size
            mask = pd.Series([True] * len(df))
            for col in demo_cols:
                if pd.notna(row[col]):
                    mask &= (df[col] == row[col])
            
            group_size = mask.sum()
            
            # Calculate weight as inverse of group proportion
            if group_size > 0:
                weight = len(df) / (len(demo_cols) * group_size)
            else:
                weight = 1.0
            
            weights.append(weight)
        
        df['demographic_weight'] = weights
        
        logger.info(f"Applied demographic weighting. Weight range: [{min(weights):.3f}, {max(weights):.3f}]")
        
        return df
    
    def apply_mitigation_strategy(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply appropriate mitigation strategy based on bias analysis
        
        Args:
            df: Input DataFrame
            analysis: Bias severity analysis
            
        Returns:
            Mitigated DataFrame
        """
        if not analysis['mitigation_needed']:
            logger.info("No mitigation needed - bias within acceptable thresholds")
            return df
        
        mitigated_df = df.copy()
        
        # Determine which demographic column to focus on
        primary_bias = analysis.get('primary_bias_type', 'age_cv')
        demo_col_map = {
            'gender_cv': 'gender',
            'ethnicity_cv': 'ethnicity_clean',
            'age_cv': 'age_group'
        }
        
        target_demo_col = demo_col_map.get(primary_bias, 'age_group')
        
        # Apply mitigation based on severity
        severity = analysis['severity_level']
        
        if severity == 'critical':
            # Critical: Apply stratified balancing (most aggressive)
            logger.info(f"Applying critical mitigation: stratified balancing on {target_demo_col}")
            mitigated_df = self.apply_stratified_balancing(mitigated_df, target_demo_col)
            
        elif severity == 'high':
            # High: Apply oversampling of minority groups
            logger.info(f"Applying high severity mitigation: oversampling on {target_demo_col}")
            mitigated_df = self.apply_oversampling_minority_groups(mitigated_df, target_demo_col)
            
        elif severity == 'medium':
            # Medium: Apply demographic weighting
            logger.info(f"Applying medium severity mitigation: demographic weighting")
            mitigated_df = self.apply_demographic_weighting(mitigated_df)
        
        # Add mitigation metadata
        mitigated_df['bias_mitigated'] = True
        mitigated_df['mitigation_strategy'] = severity
        mitigated_df['mitigation_timestamp'] = datetime.now().isoformat()
        
        return mitigated_df
    
    def calculate_post_mitigation_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate bias metrics after mitigation to verify improvement
        
        Args:
            df: Mitigated DataFrame
            
        Returns:
            Dictionary with post-mitigation bias metrics
        """
        metrics = {}
        
        # Calculate CV for each demographic group if text_chars column exists
        if 'text_chars' in df.columns:
            # Gender CV
            if 'gender' in df.columns:
                gender_means = df.groupby('gender')['text_chars'].mean()
                if len(gender_means) > 1 and gender_means.mean() > 0:
                    cv = (gender_means.std() / gender_means.mean()) * 100
                    metrics['gender_cv'] = float(cv)
            
            # Ethnicity CV
            if 'ethnicity_clean' in df.columns:
                ethnicity_means = df.groupby('ethnicity_clean')['text_chars'].mean()
                if len(ethnicity_means) > 1 and ethnicity_means.mean() > 0:
                    cv = (ethnicity_means.std() / ethnicity_means.mean()) * 100
                    metrics['ethnicity_cv'] = float(cv)
            
            # Age CV
            if 'age_group' in df.columns:
                age_means = df.groupby('age_group')['text_chars'].mean()
                if len(age_means) > 1 and age_means.mean() > 0:
                    cv = (age_means.std() / age_means.mean()) * 100
                    metrics['age_cv'] = float(cv)
        
        # Calculate overall bias score
        cv_values = [v for k, v in metrics.items() if k.endswith('_cv')]
        if cv_values:
            metrics['overall_bias_score'] = float(np.mean(cv_values))
        
        return metrics
    
    def generate_mitigation_report(self, original_metrics: Dict, post_mitigation_metrics: Dict,
                                   analysis: Dict[str, Any], mitigation_applied: bool) -> Dict[str, Any]:
        """
        Generate comprehensive mitigation report with before/after comparison
        
        Args:
            original_metrics: Bias metrics before mitigation
            post_mitigation_metrics: Bias metrics after mitigation
            analysis: Bias severity analysis
            mitigation_applied: Whether mitigation was applied
            
        Returns:
            Dictionary with complete mitigation report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'mitigation_applied': mitigation_applied,
            'severity_level': analysis['severity_level'],
            'primary_bias_type': analysis['primary_bias_type'],
            'before_mitigation': original_metrics,
            'after_mitigation': post_mitigation_metrics if mitigation_applied else original_metrics,
            'improvement': {},
            'compliance_status': {},
            'recommendations': []
        }
        
        # Calculate improvement
        if mitigation_applied:
            for metric_name in ['overall_bias_score', 'gender_cv', 'ethnicity_cv', 'age_cv']:
                if metric_name in original_metrics and metric_name in post_mitigation_metrics:
                    original_value = original_metrics[metric_name]
                    new_value = post_mitigation_metrics[metric_name]
                    improvement_pct = ((original_value - new_value) / original_value) * 100
                    
                    report['improvement'][metric_name] = {
                        'original': original_value,
                        'new': new_value,
                        'improvement_percentage': improvement_pct,
                        'absolute_reduction': original_value - new_value
                    }
        
        # Check compliance with thresholds
        metrics_to_check = post_mitigation_metrics if mitigation_applied else original_metrics
        
        compliance = {
            'overall_compliant': True,
            'violations': []
        }
        
        for metric_name, threshold_key in [
            ('overall_bias_score', 'overall_bias_score_max'),
            ('gender_cv', 'gender_cv_max'),
            ('ethnicity_cv', 'ethnicity_cv_max'),
            ('age_cv', 'age_cv_max')
        ]:
            if metric_name in metrics_to_check:
                value = metrics_to_check[metric_name]
                threshold = self.thresholds.get(threshold_key, 999)
                
                if value > threshold:
                    compliance['overall_compliant'] = False
                    compliance['violations'].append({
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'excess': value - threshold
                    })
        
        report['compliance_status'] = compliance
        
        # Generate recommendations
        if not compliance['overall_compliant']:
            if mitigation_applied:
                report['recommendations'].append("Additional mitigation required - consider more aggressive strategies")
                report['recommendations'].append("Review data collection process for systematic biases")
            else:
                report['recommendations'].append("Apply bias mitigation strategies")
            
            report['recommendations'].append("Implement continuous bias monitoring")
            report['recommendations'].append("Review model fairness metrics")
        else:
            report['recommendations'].append("Bias mitigation successful - continue monitoring")
            report['recommendations'].append("Maintain current data quality standards")
        
        return report
    
    def run_mitigation_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete automated bias mitigation pipeline
        
        Returns:
            Tuple of (mitigated DataFrame, mitigation report)
        """
        logger.info("="*60)
        logger.info("AUTOMATED BIAS MITIGATION PIPELINE")
        logger.info("="*60)
        
        # Load bias report from detection step
        bias_report = self.load_bias_report()
        
        # Load features data
        df = self.load_features_data()
        original_size = len(df)
        
        # Analyze bias severity
        logger.info("Analyzing bias severity...")
        analysis = self.analyze_bias_severity(bias_report)
        
        # Store original metrics
        original_metrics = bias_report.get('summary_metrics', {})
        
        # Apply mitigation if needed
        mitigated_df = df
        mitigation_applied = False
        
        if analysis['mitigation_needed'] and self.mitigation_config.get('mitigation_enabled', True):
            logger.info("Applying bias mitigation strategies...")
            mitigated_df = self.apply_mitigation_strategy(df, analysis)
            mitigation_applied = True
            
            # Calculate post-mitigation metrics
            logger.info("Calculating post-mitigation bias metrics...")
            post_mitigation_metrics = self.calculate_post_mitigation_metrics(mitigated_df)
        else:
            logger.info("No mitigation needed or mitigation disabled")
            post_mitigation_metrics = original_metrics
        
        # Generate comprehensive report
        logger.info("Generating mitigation report...")
        mitigation_report = self.generate_mitigation_report(
            original_metrics,
            post_mitigation_metrics,
            analysis,
            mitigation_applied
        )
        
        # Save mitigated dataset
        if mitigation_applied:
            output_file = os.path.join(self.input_path, 'mimic_features_mitigated.csv')
            mitigated_df.to_csv(output_file, index=False)
            logger.info(f"Saved mitigated dataset: {output_file}")
            mitigation_report['mitigated_dataset_path'] = output_file
        
        # Save mitigation report
        report_file = os.path.join(self.output_path, 'bias_mitigation_report.json')
        with open(report_file, 'w') as f:
            json.dump(mitigation_report, f, indent=2, default=str)
        logger.info(f"Saved mitigation report: {report_file}")
        
        logger.info("="*60)
        logger.info("BIAS MITIGATION COMPLETE")
        logger.info("="*60)
        
        return mitigated_df, mitigation_report


if __name__ == "__main__":
    import json
    
    # Load configuration from project
    config_path = os.path.join(os.path.dirname(__file__), '../configs/pipeline_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize bias handler with config
    handler = AutomatedBiasHandler(
        input_path=config['pipeline_config']['output_path'],
        output_path=config['pipeline_config']['logs_path'],
        config=config
    )
    
    # Run mitigation pipeline
    mitigated_df, report = handler.run_mitigation_pipeline()
    
    # Print results
    print("\n" + "="*60)
    print("BIAS MITIGATION RESULTS")
    print("="*60)
    print(f"Mitigation Applied: {report['mitigation_applied']}")
    print(f"Severity Level: {report['severity_level']}")
    print(f"Primary Bias Type: {report['primary_bias_type']}")
    
    if report['mitigation_applied']:
        print("\nBias Score Changes:")
        if 'improvement' in report and 'overall_bias_score' in report['improvement']:
            imp = report['improvement']['overall_bias_score']
            print(f"  Before: {imp['original']:.2f}")
            print(f"  After:  {imp['new']:.2f}")
            print(f"  Improvement: {imp['improvement_percentage']:.1f}%")
        
        print("\nRecords:")
        print(f"  Original: {report['before_mitigation'].get('record_count', 'N/A')}")
        print(f"  Mitigated: {len(mitigated_df)}")
    
    print("\nCompliance Status:")
    print(f"  Overall Compliant: {report['compliance_status']['overall_compliant']}")
    
    if report['compliance_status']['violations']:
        print("  Remaining Violations:")
        for violation in report['compliance_status']['violations']:
            print(f"    - {violation['metric']}: {violation['value']:.2f} (threshold: {violation['threshold']:.2f})")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\nOutput Files:")
    print(f"  Mitigation Report: {config['pipeline_config']['logs_path']}/bias_mitigation_report.json")
    if report['mitigation_applied']:
        print(f"  Mitigated Dataset: {report.get('mitigated_dataset_path', 'N/A')}")
    print("="*60)