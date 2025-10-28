#!/usr/bin/env python3
"""
Main Data Pipeline for Lab Lens MIMIC-III Processing
Integrates preprocessing, validation, bias detection, and automated bias handling
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger, log_data_operation
from utils.error_handling import (
    ErrorHandler, safe_execute, ErrorContext, 
    validate_file_path, log_error_summary
)

# Import pipeline components
from scripts.preprocessing import MIMICPreprocessor
from scripts.validation import MIMICDataValidator
from scripts.bias_detection import MIMICBiasDetector
from scripts.automated_bias_handler import AutomatedBiasHandler, run_automated_bias_handling


class LabLensPipeline:
    """Main pipeline orchestrator for Lab Lens data processing"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize pipeline components
        self.preprocessor = MIMICPreprocessor(
            input_path=self.config.get('input_path', 'data/raw'),
            output_path=self.config.get('output_path', 'data/processed')
        )
        
        self.validator = MIMICDataValidator(
            input_path=self.config.get('output_path', 'data/processed'),
            output_path=self.config.get('logs_path', 'logs')
        )
        
        self.bias_detector = MIMICBiasDetector(
            input_path=self.config.get('input_path', 'data/raw'),
            output_path=self.config.get('logs_path', 'logs')
        )
        
        self.bias_handler = AutomatedBiasHandler(
            config_path=self.config.get('bias_config_path')
        )
        
        # Pipeline state
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'steps_failed': [],
            'data_shape': {},
            'validation_score': 0,
            'bias_score': 0,
            'mitigation_applied': False
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            'input_path': 'data/raw',
            'output_path': 'data/processed',
            'logs_path': 'logs',
            'enable_preprocessing': True,
            'enable_validation': True,
            'enable_bias_detection': True,
            'enable_automated_bias_handling': True,
            'auto_mitigation': True,
            'save_intermediate_results': True,
            'log_level': 'INFO'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded pipeline configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    @safe_execute("run_preprocessing", logger, ErrorHandler(logger))
    @log_data_operation(logger, "preprocessing")
    def run_preprocessing(self) -> Dict[str, Any]:
        """Run data preprocessing step"""
        with ErrorContext("preprocessing", self.logger, self.error_handler) as ctx:
            self.logger.info("Starting preprocessing step...")
            
            df_processed, report = self.preprocessor.run_preprocessing_pipeline()
            
            self.pipeline_state['data_shape']['preprocessed'] = df_processed.shape
            self.pipeline_state['steps_completed'].append('preprocessing')
            
            self.logger.info(f"Preprocessing completed: {df_processed.shape[0]} records, {df_processed.shape[1]} features")
            
            return {
                'dataframe': df_processed,
                'report': report,
                'success': True
            }
    
    @safe_execute("run_validation", logger, ErrorHandler(logger))
    @log_data_operation(logger, "validation")
    def run_validation(self) -> Dict[str, Any]:
        """Run data validation step"""
        with ErrorContext("validation", self.logger, self.error_handler) as ctx:
            self.logger.info("Starting validation step...")
            
            validation_report, summary_df = self.validator.run_validation_pipeline()
            
            self.pipeline_state['validation_score'] = validation_report.get('overall_score', 0)
            self.pipeline_state['steps_completed'].append('validation')
            
            self.logger.info(f"Validation completed with score: {self.pipeline_state['validation_score']:.2f}%")
            
            return {
                'report': validation_report,
                'summary': summary_df,
                'success': True
            }
    
    @safe_execute("run_bias_detection", logger, ErrorHandler(logger))
    @log_data_operation(logger, "bias_detection")
    def run_bias_detection(self) -> Dict[str, Any]:
        """Run bias detection step"""
        with ErrorContext("bias_detection", self.logger, self.error_handler) as ctx:
            self.logger.info("Starting bias detection step...")
            
            bias_report, summary_df = self.bias_detector.run_bias_detection_pipeline()
            
            self.pipeline_state['bias_score'] = bias_report.get('summary_metrics', {}).get('overall_bias_score', 0)
            self.pipeline_state['steps_completed'].append('bias_detection')
            
            self.logger.info(f"Bias detection completed with score: {self.pipeline_state['bias_score']:.2f}")
            
            return {
                'report': bias_report,
                'summary': summary_df,
                'success': True
            }
    
    @safe_execute("run_automated_bias_handling", logger, ErrorHandler(logger))
    @log_data_operation(logger, "automated_bias_handling")
    def run_automated_bias_handling(self, df: Any, bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """Run automated bias handling step"""
        with ErrorContext("automated_bias_handling", self.logger, self.error_handler) as ctx:
            self.logger.info("Starting automated bias handling step...")
            
            # Load the processed data for bias handling
            processed_file = os.path.join(self.config['output_path'], 'processed_discharge_summaries.csv')
            if os.path.exists(processed_file):
                df_processed = pd.read_csv(processed_file)
            else:
                # Fallback to raw data
                raw_file = os.path.join(self.config['input_path'], 'mimic_discharge_labs.csv')
                df_processed = pd.read_csv(raw_file)
            
            mitigated_df, handling_report = run_automated_bias_handling(
                df_processed, bias_report, 
                self.config.get('bias_config_path')
            )
            
            # Save mitigated data if mitigation was applied
            if len(mitigated_df) != len(df_processed):
                mitigated_file = os.path.join(self.config['output_path'], 'mitigated_discharge_summaries.csv')
                mitigated_df.to_csv(mitigated_file, index=False)
                self.pipeline_state['mitigation_applied'] = True
                self.logger.info(f"Saved mitigated data to {mitigated_file}")
            
            # Save bias handling report
            self.bias_handler.save_bias_report(handling_report, self.config['logs_path'])
            
            self.pipeline_state['steps_completed'].append('automated_bias_handling')
            
            self.logger.info("Automated bias handling completed")
            
            return {
                'mitigated_dataframe': mitigated_df,
                'handling_report': handling_report,
                'success': True
            }
    
    @safe_execute("run_complete_pipeline", logger, ErrorHandler(logger))
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete data processing pipeline"""
        self.pipeline_state['start_time'] = datetime.now()
        
        self.logger.info("Starting Lab Lens complete data pipeline...")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        pipeline_results = {}
        
        try:
            # Step 1: Preprocessing
            if self.config.get('enable_preprocessing', True):
                preprocessing_result = self.run_preprocessing()
                pipeline_results['preprocessing'] = preprocessing_result
            
            # Step 2: Validation
            if self.config.get('enable_validation', True):
                validation_result = self.run_validation()
                pipeline_results['validation'] = validation_result
            
            # Step 3: Bias Detection
            if self.config.get('enable_bias_detection', True):
                bias_result = self.run_bias_detection()
                pipeline_results['bias_detection'] = bias_result
            
            # Step 4: Automated Bias Handling
            if self.config.get('enable_automated_bias_handling', True) and 'bias_detection' in pipeline_results:
                bias_handling_result = self.run_automated_bias_handling(
                    None,  # Will load data internally
                    pipeline_results['bias_detection']['report']
                )
                pipeline_results['bias_handling'] = bias_handling_result
            
            # Generate final pipeline report
            self.pipeline_state['end_time'] = datetime.now()
            pipeline_results['pipeline_summary'] = self._generate_pipeline_summary()
            
            # Save pipeline results
            self._save_pipeline_results(pipeline_results)
            
            self.logger.info("Pipeline completed successfully!")
            return pipeline_results
            
        except Exception as e:
            log_error_summary(self.logger, e, "complete_pipeline")
            self.pipeline_state['steps_failed'].append('pipeline_execution')
            raise
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary"""
        duration = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        
        summary = {
            'pipeline_execution': {
                'start_time': self.pipeline_state['start_time'].isoformat(),
                'end_time': self.pipeline_state['end_time'].isoformat(),
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'steps_completed': self.pipeline_state['steps_completed'],
                'steps_failed': self.pipeline_state['steps_failed'],
                'success_rate': len(self.pipeline_state['steps_completed']) / 
                              (len(self.pipeline_state['steps_completed']) + len(self.pipeline_state['steps_failed']))
            },
            'data_metrics': {
                'validation_score': self.pipeline_state['validation_score'],
                'bias_score': self.pipeline_state['bias_score'],
                'mitigation_applied': self.pipeline_state['mitigation_applied']
            },
            'quality_assessment': {
                'validation_passed': self.pipeline_state['validation_score'] >= 80,
                'bias_acceptable': self.pipeline_state['bias_score'] <= 10,
                'overall_quality': 'excellent' if self.pipeline_state['validation_score'] >= 90 and 
                                 self.pipeline_state['bias_score'] <= 5 else 'good'
            }
        }
        
        return summary
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to files"""
        os.makedirs(self.config['logs_path'], exist_ok=True)
        
        # Save complete results
        results_file = os.path.join(
            self.config['logs_path'], 
            f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline results saved to {results_file}")
        
        # Save summary
        summary_file = os.path.join(self.config['logs_path'], 'pipeline_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results['pipeline_summary'], f, indent=2, default=str)
        
        self.logger.info(f"Pipeline summary saved to {summary_file}")


def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(description='Lab Lens Data Processing Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input-path', type=str, help='Input data path')
    parser.add_argument('--output-path', type=str, help='Output data path')
    parser.add_argument('--logs-path', type=str, help='Logs output path')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation step')
    parser.add_argument('--skip-bias-detection', action='store_true', help='Skip bias detection step')
    parser.add_argument('--skip-bias-handling', action='store_true', help='Skip automated bias handling')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config_overrides = {}
    if args.input_path:
        config_overrides['input_path'] = args.input_path
    if args.output_path:
        config_overrides['output_path'] = args.output_path
    if args.logs_path:
        config_overrides['logs_path'] = args.logs_path
    if args.skip_preprocessing:
        config_overrides['enable_preprocessing'] = False
    if args.skip_validation:
        config_overrides['enable_validation'] = False
    if args.skip_bias_detection:
        config_overrides['enable_bias_detection'] = False
    if args.skip_bias_handling:
        config_overrides['enable_automated_bias_handling'] = False
    if args.log_level:
        config_overrides['log_level'] = args.log_level
    
    # Create temporary config file if overrides provided
    temp_config_path = None
    if config_overrides:
        temp_config_path = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config_overrides, f, indent=2)
    
    try:
        # Initialize and run pipeline
        pipeline = LabLensPipeline(temp_config_path or args.config)
        results = pipeline.run_complete_pipeline()
        
        print("\n=== Pipeline Execution Summary ===")
        summary = results['pipeline_summary']
        print(f"Duration: {summary['pipeline_execution']['duration_minutes']:.2f} minutes")
        print(f"Steps Completed: {', '.join(summary['pipeline_execution']['steps_completed'])}")
        print(f"Validation Score: {summary['data_metrics']['validation_score']:.2f}%")
        print(f"Bias Score: {summary['data_metrics']['bias_score']:.2f}")
        print(f"Mitigation Applied: {summary['data_metrics']['mitigation_applied']}")
        print(f"Overall Quality: {summary['quality_assessment']['overall_quality']}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)
    
    finally:
        # Clean up temporary config file
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()
