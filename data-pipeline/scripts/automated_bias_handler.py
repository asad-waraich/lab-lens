"""
Automated Bias Detection and Mitigation Pipeline
Provides automated bias detection, reporting, and mitigation strategies
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


class AutomatedBiasHandler:
    """Automated bias detection and mitigation system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Bias thresholds
        self.bias_thresholds = {
            'gender_cv_max': 5.0,  # Maximum coefficient of variation for gender
            'ethnicity_cv_max': 10.0,  # Maximum coefficient of variation for ethnicity
            'age_cv_max': 8.0,  # Maximum coefficient of variation for age
            'overall_bias_score_max': 10.0,  # Maximum overall bias score
            'min_sample_size': 30,  # Minimum sample size per group
            'significance_threshold': 0.05  # Statistical significance threshold
        }
        
        # Mitigation strategies
        self.mitigation_strategies = {
            'sampling': 'stratified_sampling',
            'weighting': 'demographic_weighting',
            'augmentation': 'synthetic_data_generation',
            'balancing': 'oversampling_minority_groups'
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load bias detection configuration"""
        default_config = {
            'enabled': True,
            'auto_mitigation': True,
            'report_generation': True,
            'alert_thresholds': self.bias_thresholds,
            'mitigation_enabled': True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded bias configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    @safe_execute("detect_bias", self.logger, ErrorHandler(self.logger))
    def detect_bias(self, df: pd.DataFrame, bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect bias patterns and determine if mitigation is needed
        
        Args:
            df: DataFrame with demographic data
            bias_report: Existing bias analysis report
            
        Returns:
            Dictionary with bias detection results and recommendations
        """
        detection_results = {
            'timestamp': datetime.now().isoformat(),
            'bias_detected': False,
            'critical_biases': [],
            'warnings': [],
            'recommendations': [],
            'mitigation_needed': False,
            'bias_score': 0.0
        }
        
        # Check overall bias score
        if 'summary_metrics' in bias_report:
            overall_score = bias_report['summary_metrics'].get('overall_bias_score', 0)
            detection_results['bias_score'] = overall_score
            
            if overall_score > self.bias_thresholds['overall_bias_score_max']:
                detection_results['bias_detected'] = True
                detection_results['critical_biases'].append({
                    'type': 'overall_bias_score',
                    'value': overall_score,
                    'threshold': self.bias_thresholds['overall_bias_score_max'],
                    'severity': 'high' if overall_score > 20 else 'medium'
                })
        
        # Check demographic variations
        for demo_type in ['gender_cv', 'ethnicity_cv', 'age_cv']:
            if demo_type in bias_report.get('summary_metrics', {}):
                cv_value = bias_report['summary_metrics'][demo_type]
                threshold_key = f"{demo_type.split('_')[0]}_cv_max"
                
                if cv_value > self.bias_thresholds[threshold_key]:
                    detection_results['bias_detected'] = True
                    detection_results['critical_biases'].append({
                        'type': demo_type,
                        'value': cv_value,
                        'threshold': self.bias_thresholds[threshold_key],
                        'severity': 'high' if cv_value > self.bias_thresholds[threshold_key] * 2 else 'medium'
                    })
        
        # Check statistical significance
        if 'documentation_bias' in bias_report:
            doc_bias = bias_report['documentation_bias']
            if 'documentation_length_bias' in doc_bias:
                if 'gender_ttest' in doc_bias['documentation_length_bias']:
                    p_value = doc_bias['documentation_length_bias']['gender_ttest']['p_value']
                    if p_value < self.bias_thresholds['significance_threshold']:
                        detection_results['bias_detected'] = True
                        detection_results['critical_biases'].append({
                            'type': 'statistical_significance',
                            'value': p_value,
                            'threshold': self.bias_thresholds['significance_threshold'],
                            'severity': 'high'
                        })
        
        # Generate recommendations
        detection_results['recommendations'] = self._generate_recommendations(detection_results)
        
        # Determine if mitigation is needed
        detection_results['mitigation_needed'] = (
            detection_results['bias_detected'] and 
            self.config.get('mitigation_enabled', True)
        )
        
        return detection_results
    
    def _generate_recommendations(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        for bias in detection_results['critical_biases']:
            bias_type = bias['type']
            severity = bias['severity']
            
            if bias_type == 'overall_bias_score':
                recommendations.append({
                    'type': 'comprehensive_rebalancing',
                    'priority': 'high' if severity == 'high' else 'medium',
                    'description': 'Implement comprehensive demographic rebalancing',
                    'strategy': 'stratified_sampling',
                    'estimated_impact': 'high'
                })
            
            elif 'cv' in bias_type:
                demo_type = bias_type.split('_')[0]
                recommendations.append({
                    'type': f'{demo_type}_balancing',
                    'priority': 'high' if severity == 'high' else 'medium',
                    'description': f'Balance {demo_type} representation in dataset',
                    'strategy': 'oversampling_minority_groups',
                    'estimated_impact': 'medium'
                })
            
            elif bias_type == 'statistical_significance':
                recommendations.append({
                    'type': 'statistical_correction',
                    'priority': 'high',
                    'description': 'Apply statistical correction for significant differences',
                    'strategy': 'demographic_weighting',
                    'estimated_impact': 'high'
                })
        
        return recommendations
    
    @safe_execute("apply_mitigation", self.logger, ErrorHandler(self.logger))
    def apply_mitigation(self, df: pd.DataFrame, recommendations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply bias mitigation strategies to the dataset
        
        Args:
            df: Original DataFrame
            recommendations: List of mitigation recommendations
            
        Returns:
            Mitigated DataFrame
        """
        mitigated_df = df.copy()
        
        for rec in recommendations:
            strategy = rec['strategy']
            
            if strategy == 'stratified_sampling':
                mitigated_df = self._apply_stratified_sampling(mitigated_df)
            
            elif strategy == 'oversampling_minority_groups':
                mitigated_df = self._apply_oversampling(mitigated_df, rec['type'])
            
            elif strategy == 'demographic_weighting':
                mitigated_df = self._apply_demographic_weighting(mitigated_df)
            
            elif strategy == 'synthetic_data_generation':
                mitigated_df = self._apply_synthetic_augmentation(mitigated_df)
        
        self.logger.info(f"Applied bias mitigation: {len(df)} -> {len(mitigated_df)} records")
        return mitigated_df
    
    def _apply_stratified_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stratified sampling to balance demographics"""
        # Identify demographic columns
        demo_cols = [col for col in df.columns if col in ['gender', 'ethnicity_group', 'age_group']]
        
        if not demo_cols:
            self.logger.warning("No demographic columns found for stratified sampling")
            return df
        
        # Calculate target sample size per group
        min_group_size = df.groupby(demo_cols).size().min()
        target_size = min(min_group_size, len(df) // 10)  # Cap at 10% of total
        
        # Stratified sampling
        sampled_groups = []
        for name, group in df.groupby(demo_cols):
            if len(group) >= target_size:
                sampled_group = group.sample(n=target_size, random_state=42)
                sampled_groups.append(sampled_group)
        
        if sampled_groups:
            balanced_df = pd.concat(sampled_groups, ignore_index=True)
            self.logger.info(f"Applied stratified sampling: {len(df)} -> {len(balanced_df)} records")
            return balanced_df
        
        return df
    
    def _apply_oversampling(self, df: pd.DataFrame, bias_type: str) -> pd.DataFrame:
        """Apply oversampling to minority groups"""
        if 'gender' in df.columns and 'gender' in bias_type:
            # Oversample minority gender
            gender_counts = df['gender'].value_counts()
            minority_gender = gender_counts.idxmin()
            majority_gender = gender_counts.idxmax()
            
            minority_group = df[df['gender'] == minority_gender]
            oversample_factor = gender_counts[majority_gender] // gender_counts[minority_gender]
            
            if oversample_factor > 1:
                oversampled_minority = minority_group.sample(
                    n=len(minority_group) * oversample_factor, 
                    replace=True, 
                    random_state=42
                )
                
                balanced_df = pd.concat([
                    df[df['gender'] == majority_gender],
                    oversampled_minority
                ], ignore_index=True)
                
                self.logger.info(f"Applied gender oversampling: {len(df)} -> {len(balanced_df)} records")
                return balanced_df
        
        return df
    
    def _apply_demographic_weighting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply demographic weighting to balance representation"""
        # Calculate weights based on inverse frequency
        demo_cols = [col for col in df.columns if col in ['gender', 'ethnicity_group', 'age_group']]
        
        if demo_cols:
            weights = []
            for _, row in df.iterrows():
                group_key = tuple(row[col] for col in demo_cols)
                group_count = len(df.groupby(demo_cols).get_group(group_key))
                weight = len(df) / (len(df.groupby(demo_cols)) * group_count)
                weights.append(weight)
            
            df['demographic_weight'] = weights
            self.logger.info("Applied demographic weighting")
        
        return df
    
    def _apply_synthetic_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply synthetic data augmentation for minority groups"""
        # Simple augmentation by duplicating and slightly modifying records
        demo_cols = [col for col in df.columns if col in ['gender', 'ethnicity_group', 'age_group']]
        
        if demo_cols:
            augmented_records = []
            
            for name, group in df.groupby(demo_cols):
                if len(group) < self.bias_thresholds['min_sample_size']:
                    # Create synthetic records by duplicating and modifying
                    needed = self.bias_thresholds['min_sample_size'] - len(group)
                    
                    for _ in range(min(needed, len(group))):
                        synthetic_record = group.sample(1).copy()
                        # Add small random variations to numeric columns
                        numeric_cols = synthetic_record.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col not in ['hadm_id', 'subject_id']:  # Don't modify IDs
                                synthetic_record[col] *= np.random.uniform(0.95, 1.05)
                        
                        synthetic_record['is_synthetic'] = True
                        augmented_records.append(synthetic_record)
            
            if augmented_records:
                synthetic_df = pd.concat(augmented_records, ignore_index=True)
                df['is_synthetic'] = False
                augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
                
                self.logger.info(f"Applied synthetic augmentation: {len(df)} -> {len(augmented_df)} records")
                return augmented_df
        
        return df
    
    @safe_execute("generate_bias_report", self.logger, ErrorHandler(self.logger))
    def generate_bias_report(self, detection_results: Dict[str, Any], 
                           mitigation_applied: bool = False) -> Dict[str, Any]:
        """Generate comprehensive bias detection and mitigation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'bias_detection_summary': {
                'bias_detected': detection_results['bias_detected'],
                'bias_score': detection_results['bias_score'],
                'critical_biases_count': len(detection_results['critical_biases']),
                'mitigation_needed': detection_results['mitigation_needed']
            },
            'critical_biases': detection_results['critical_biases'],
            'recommendations': detection_results['recommendations'],
            'mitigation_status': {
                'applied': mitigation_applied,
                'strategies_used': [rec['strategy'] for rec in detection_results['recommendations']]
            },
            'compliance_status': self._assess_compliance(detection_results),
            'next_steps': self._generate_next_steps(detection_results, mitigation_applied)
        }
        
        return report
    
    def _assess_compliance(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with bias thresholds"""
        compliance = {
            'overall_compliant': True,
            'threshold_violations': [],
            'compliance_score': 100
        }
        
        for bias in detection_results['critical_biases']:
            compliance['overall_compliant'] = False
            compliance['threshold_violations'].append({
                'type': bias['type'],
                'severity': bias['severity'],
                'violation_amount': bias['value'] - bias['threshold']
            })
            
            # Reduce compliance score based on severity
            if bias['severity'] == 'high':
                compliance['compliance_score'] -= 20
            else:
                compliance['compliance_score'] -= 10
        
        compliance['compliance_score'] = max(0, compliance['compliance_score'])
        
        return compliance
    
    def _generate_next_steps(self, detection_results: Dict[str, Any], 
                            mitigation_applied: bool) -> List[str]:
        """Generate next steps based on bias detection results"""
        next_steps = []
        
        if detection_results['bias_detected']:
            if not mitigation_applied:
                next_steps.append("Apply recommended bias mitigation strategies")
                next_steps.append("Re-run bias detection after mitigation")
            
            next_steps.append("Monitor bias metrics in production")
            next_steps.append("Implement continuous bias monitoring")
        
        if detection_results['bias_score'] > self.bias_thresholds['overall_bias_score_max']:
            next_steps.append("Review data collection processes")
            next_steps.append("Consider additional data sources")
        
        if not next_steps:
            next_steps.append("Continue monitoring bias metrics")
            next_steps.append("Maintain current data quality standards")
        
        return next_steps
    
    @safe_execute("save_bias_report", self.logger, ErrorHandler(self.logger))
    def save_bias_report(self, report: Dict[str, Any], output_path: str):
        """Save bias detection and mitigation report"""
        os.makedirs(output_path, exist_ok=True)
        
        report_file = os.path.join(output_path, f"automated_bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Saved automated bias report to {report_file}")
        return report_file


def run_automated_bias_handling(df: pd.DataFrame, bias_report: Dict[str, Any], 
                               config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run automated bias detection and mitigation pipeline
    
    Args:
        df: DataFrame to analyze and potentially mitigate
        bias_report: Existing bias analysis report
        config_path: Path to configuration file
        
    Returns:
        Tuple of (mitigated_dataframe, bias_handling_report)
    """
    handler = AutomatedBiasHandler(config_path)
    
    # Detect bias
    detection_results = handler.detect_bias(df, bias_report)
    
    # Apply mitigation if needed
    mitigated_df = df
    mitigation_applied = False
    
    if detection_results['mitigation_needed']:
        mitigated_df = handler.apply_mitigation(df, detection_results['recommendations'])
        mitigation_applied = True
    
    # Generate comprehensive report
    report = handler.generate_bias_report(detection_results, mitigation_applied)
    
    return mitigated_df, report


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    
    # This would typically be called from the main pipeline
    logger.info("Automated bias handling module loaded successfully")
