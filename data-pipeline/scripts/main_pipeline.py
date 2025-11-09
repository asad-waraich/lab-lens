"""
Main Pipeline Orchestrator for Lab Lens MIMIC-III Processing
Author: Lab Lens Team
Description: Orchestrates complete pipeline from preprocessing to bias mitigation
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)


class LabLensPipeline:
    """Main pipeline orchestrator that runs all processing steps in sequence"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline orchestrator
        
        Args:
            config_path: Path to pipeline configuration file
        """
        self.logger = logger
        
        # Find project root and config
        self.project_root = self._find_project_root()
        self.scripts_dir = self.project_root / 'data-pipeline' / 'scripts'
        
        # Load configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.project_root / 'data-pipeline' / 'configs' / 'pipeline_config.json'
        
        self.config = self._load_config()
        
        # Pipeline state tracking
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'steps_completed': [],
            'steps_failed': [],
            'step_durations': {},
            'validation_score': 0,
            'bias_score_before': 0,
            'bias_score_after': 0,
            'mitigation_applied': False
        }
        
        self.logger.info(f"Initialized pipeline orchestrator")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Config file: {self.config_path}")
    
    def _find_project_root(self) -> Path:
        """
        Find project root directory by looking for data-pipeline folder
        
        Returns:
            Path to project root
        """
        current = Path(__file__).resolve().parent
        
        # Go up directory tree looking for data-pipeline
        while current != current.parent:
            if (current / 'data-pipeline').exists():
                return current
            current = current.parent
        
        # If not found, use current directory
        return Path.cwd()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load pipeline configuration from JSON file
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info("Loaded pipeline configuration")
            return config.get('pipeline_config', config)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default pipeline configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'input_path': 'data-pipeline/data/raw',
            'output_path': 'data-pipeline/data/processed',
            'logs_path': 'data-pipeline/logs',
            'enable_preprocessing': True,
            'enable_validation': True,
            'enable_bias_detection': True,
            'enable_automated_bias_handling': True
        }
    
    def _run_script(self, script_name: str, step_name: str) -> bool:
        """
        Run a pipeline script using subprocess
        
        Args:
            script_name: Name of Python script to run
            step_name: Name of pipeline step for logging
            
        Returns:
            True if script succeeded, False otherwise
        """
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            return False
        
        self.logger.info(f"Running {step_name}...")
        self.logger.info(f"Script: {script_path}")
        
        step_start = time.time()
        
        try:
            # Run script in subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output
            if result.stdout:
                self.logger.info(f"{step_name} output:\n{result.stdout}")
            
            step_duration = time.time() - step_start
            self.pipeline_state['step_durations'][step_name] = step_duration
            self.pipeline_state['steps_completed'].append(step_name)
            
            self.logger.info(f"{step_name} completed in {step_duration:.2f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{step_name} failed with error code {e.returncode}")
            if e.stdout:
                self.logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                self.logger.error(f"stderr: {e.stderr}")
            
            self.pipeline_state['steps_failed'].append(step_name)
            return False
        
        except Exception as e:
            self.logger.error(f"Unexpected error in {step_name}: {str(e)}")
            self.pipeline_state['steps_failed'].append(step_name)
            return False
    
    def run_preprocessing(self) -> bool:
        """Run preprocessing step"""
        return self._run_script('preprocessing.py', 'preprocessing')
    
    def run_validation(self) -> bool:
        """Run validation step"""
        return self._run_script('validation.py', 'validation')
    
    def run_feature_engineering(self) -> bool:
        """Run feature engineering step"""
        return self._run_script('feature_engineering.py', 'feature_engineering')
    
    def run_bias_detection(self) -> bool:
        """Run bias detection step"""
        return self._run_script('bias_detection.py', 'bias_detection')
    
    def run_bias_mitigation(self) -> bool:
        """Run automated bias mitigation step"""
        return self._run_script('automated_bias_handler.py', 'bias_mitigation')
    
    def _load_validation_score(self) -> float:
        """
        Load validation score from validation report
        
        Returns:
            Validation score (0-100)
        """
        try:
            report_path = self.project_root / self.config['logs_path'] / 'validation_report.json'
            with open(report_path, 'r') as f:
                report = json.load(f)
            return report.get('overall_score', 0)
        except Exception as e:
            self.logger.warning(f"Could not load validation score: {e}")
            return 0
    
    def _load_bias_scores(self) -> Dict[str, float]:
        """
        Load bias scores from bias detection and mitigation reports
        
        Returns:
            Dictionary with before and after bias scores
        """
        scores = {'before': 0, 'after': 0}
        
        try:
            # Load bias detection report
            bias_report_path = self.project_root / self.config['logs_path'] / 'bias_report.json'
            with open(bias_report_path, 'r') as f:
                bias_report = json.load(f)
            scores['before'] = bias_report.get('summary_metrics', {}).get('overall_bias_score', 0)
            
            # Load mitigation report if exists
            mitigation_report_path = self.project_root / self.config['logs_path'] / 'bias_mitigation_report.json'
            if mitigation_report_path.exists():
                with open(mitigation_report_path, 'r') as f:
                    mitigation_report = json.load(f)
                
                if 'after_mitigation' in mitigation_report:
                    scores['after'] = mitigation_report['after_mitigation'].get('overall_bias_score', scores['before'])
                    self.pipeline_state['mitigation_applied'] = mitigation_report.get('mitigation_applied', False)
            
        except Exception as e:
            self.logger.warning(f"Could not load bias scores: {e}")
        
        return scores
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline in sequence
        
        Returns:
            Dictionary with pipeline execution results
        """
        self.pipeline_state['start_time'] = datetime.now()
        
        self.logger.info("="*60)
        self.logger.info("STARTING LAB LENS COMPLETE PIPELINE")
        self.logger.info("="*60)
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Track overall success
        pipeline_success = True
        
        # Step 1: Preprocessing
        if self.config.get('enable_preprocessing', True):
            success = self.run_preprocessing()
            if not success:
                self.logger.error("Preprocessing failed - stopping pipeline")
                pipeline_success = False
                return self._generate_results(success=False)
        else:
            self.logger.info("Preprocessing skipped (disabled in config)")
        
        # Step 2: Validation
        if self.config.get('enable_validation', True):
            success = self.run_validation()
            if not success:
                self.logger.warning("Validation failed - continuing with caution")
            
            # Load validation score
            self.pipeline_state['validation_score'] = self._load_validation_score()
        else:
            self.logger.info("Validation skipped (disabled in config)")
        
        # Step 3: Feature Engineering
        # Always run feature engineering if bias detection is enabled
        if self.config.get('enable_bias_detection', True):
            success = self.run_feature_engineering()
            if not success:
                self.logger.error("Feature engineering failed - cannot run bias detection")
                pipeline_success = False
                return self._generate_results(success=False)
        
        # Step 4: Bias Detection
        if self.config.get('enable_bias_detection', True):
            success = self.run_bias_detection()
            if not success:
                self.logger.warning("Bias detection failed - skipping mitigation")
            else:
                bias_scores = self._load_bias_scores()
                self.pipeline_state['bias_score_before'] = bias_scores['before']
        else:
            self.logger.info("Bias detection skipped (disabled in config)")
        
        # Step 5: Automated Bias Mitigation
        if self.config.get('enable_automated_bias_handling', True):
            success = self.run_bias_mitigation()
            if not success:
                self.logger.warning("Bias mitigation failed")
            else:
                bias_scores = self._load_bias_scores()
                self.pipeline_state['bias_score_after'] = bias_scores['after']
        else:
            self.logger.info("Bias mitigation skipped (disabled in config)")
        
        # Pipeline complete
        self.pipeline_state['end_time'] = datetime.now()
        
        self.logger.info("="*60)
        self.logger.info("PIPELINE EXECUTION COMPLETE")
        self.logger.info("="*60)
        
        return self._generate_results(success=pipeline_success)
    
    def _generate_results(self, success: bool) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline results report
        
        Args:
            success: Whether pipeline completed successfully
            
        Returns:
            Dictionary with complete pipeline results
        """
        if self.pipeline_state['end_time'] is None:
            self.pipeline_state['end_time'] = datetime.now()
        
        duration = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        
        results = {
            'pipeline_execution': {
                'success': success,
                'start_time': self.pipeline_state['start_time'].isoformat(),
                'end_time': self.pipeline_state['end_time'].isoformat(),
                'total_duration_seconds': duration,
                'total_duration_minutes': duration / 60,
                'steps_completed': self.pipeline_state['steps_completed'],
                'steps_failed': self.pipeline_state['steps_failed'],
                'step_durations': self.pipeline_state['step_durations']
            },
            'quality_metrics': {
                'validation_score': self.pipeline_state['validation_score'],
                'validation_passed': self.pipeline_state['validation_score'] >= 80,
                'bias_score_before': self.pipeline_state['bias_score_before'],
                'bias_score_after': self.pipeline_state['bias_score_after'],
                'bias_improvement': self.pipeline_state['bias_score_before'] - self.pipeline_state['bias_score_after'],
                'mitigation_applied': self.pipeline_state['mitigation_applied']
            },
            'overall_status': {
                'pipeline_healthy': success and self.pipeline_state['validation_score'] >= 80,
                'data_quality': 'excellent' if self.pipeline_state['validation_score'] >= 90 else 
                               'good' if self.pipeline_state['validation_score'] >= 80 else 'needs_improvement',
                'bias_status': 'acceptable' if self.pipeline_state['bias_score_after'] <= 10 else 
                              'moderate' if self.pipeline_state['bias_score_after'] <= 15 else 'high',
                'ready_for_modeling': success and self.pipeline_state['validation_score'] >= 80
            }
        }
        
        # Save results to file
        self._save_pipeline_results(results)
        
        return results
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """
        Save pipeline results to JSON file
        
        Args:
            results: Pipeline results dictionary
        """
        try:
            logs_dir = self.project_root / self.config['logs_path']
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save timestamped results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = logs_dir / f'pipeline_results_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline results saved to {results_file}")
            
            # Also save as latest results
            latest_file = logs_dir / 'pipeline_results_latest.json'
            with open(latest_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Latest results saved to {latest_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline results: {e}")


def main():
    """Main entry point for the pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Lab Lens Complete Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default config
  python main_pipeline.py
  
  # Run with custom config file
  python main_pipeline.py --config /path/to/config.json
  
  # Skip specific steps
  python main_pipeline.py --skip-preprocessing --skip-validation
  
  # Run only specific steps
  python main_pipeline.py --skip-bias-detection --skip-bias-handling
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip preprocessing step')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip validation step')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                       help='Skip feature engineering step')
    parser.add_argument('--skip-bias-detection', action='store_true', 
                       help='Skip bias detection step')
    parser.add_argument('--skip-bias-handling', action='store_true', 
                       help='Skip automated bias handling step')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = LabLensPipeline(config_path=args.config)
        
        # Update config based on command line arguments
        if args.skip_preprocessing:
            pipeline.config['enable_preprocessing'] = False
        if args.skip_validation:
            pipeline.config['enable_validation'] = False
        if args.skip_feature_engineering:
            pipeline.config['enable_feature_engineering'] = False
        if args.skip_bias_detection:
            pipeline.config['enable_bias_detection'] = False
        if args.skip_bias_handling:
            pipeline.config['enable_automated_bias_handling'] = False
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Print summary to console
        print("\n" + "="*60)
        print("LAB LENS PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        exec_info = results['pipeline_execution']
        print(f"Status: {'SUCCESS' if exec_info['success'] else 'FAILED'}")
        print(f"Duration: {exec_info['total_duration_minutes']:.2f} minutes")
        print(f"Steps Completed: {len(exec_info['steps_completed'])}/{len(exec_info['steps_completed']) + len(exec_info['steps_failed'])}")
        
        if exec_info['steps_completed']:
            print(f"  Completed: {', '.join(exec_info['steps_completed'])}")
        
        if exec_info['steps_failed']:
            print(f"  Failed: {', '.join(exec_info['steps_failed'])}")
        
        print("\nQuality Metrics:")
        quality = results['quality_metrics']
        print(f"  Validation Score: {quality['validation_score']:.2f}% - {'PASS' if quality['validation_passed'] else 'FAIL'}")
        print(f"  Bias Score (Before): {quality['bias_score_before']:.2f}")
        print(f"  Bias Score (After): {quality['bias_score_after']:.2f}")
        
        if quality['mitigation_applied']:
            print(f"  Bias Improvement: {quality['bias_improvement']:.2f} points")
        
        print("\nOverall Status:")
        status = results['overall_status']
        print(f"  Data Quality: {status['data_quality'].upper()}")
        print(f"  Bias Status: {status['bias_status'].upper()}")
        print(f"  Ready for Modeling: {'YES' if status['ready_for_modeling'] else 'NO'}")
        
        print("\nOutput Files:")
        print(f"  Pipeline Results: {pipeline.project_root / pipeline.config['logs_path']}/pipeline_results_latest.json")
        print(f"  Validation Report: {pipeline.project_root / pipeline.config['logs_path']}/validation_report.json")
        print(f"  Bias Report: {pipeline.project_root / pipeline.config['logs_path']}/bias_report.json")
        print(f"  Mitigation Report: {pipeline.project_root / pipeline.config['logs_path']}/bias_mitigation_report.json")
        print("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if exec_info['success'] else 1)
        
    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        logger.error(f"Fatal pipeline error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()