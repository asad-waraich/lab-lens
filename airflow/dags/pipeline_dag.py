"""
Lab Lens MIMIC-III Processing Pipeline DAG
Author: Lab Lens Team
Description: Airflow DAG for complete MLOps pipeline orchestration
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import sys
import os
from pathlib import Path

# Default arguments for the DAG
default_args = {
    'owner': 'lab-lens-team',
    'depends_on_past': False,
    'email': ['team@lablens.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 11, 1)
}

# Find project root
def get_project_root():
    """Find the Lab Lens project root directory"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'data-pipeline').exists():
            return str(current)
        current = current.parent
    return str(Path.cwd())

PROJECT_ROOT = get_project_root()
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'data-pipeline', 'scripts')

# Task functions

def run_preprocessing_task(**context):
    """
    Task 1: Run preprocessing pipeline
    Cleans data, removes duplicates, standardizes demographics
    """
    script_path = os.path.join(SCRIPTS_DIR, 'preprocessing.py')
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Preprocessing failed: {result.stderr}")
    
    print(result.stdout)
    return {'status': 'success', 'records_processed': 'See logs'}


def run_validation_task(**context):
    """
    Task 2: Run validation pipeline
    Validates data quality, checks schema, calculates quality score
    """
    script_path = os.path.join(SCRIPTS_DIR, 'validation.py')
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Validation failed: {result.stderr}")
    
    print(result.stdout)
    
    # Parse validation score from output
    import json
    report_path = os.path.join(PROJECT_ROOT, 'data-pipeline/logs/validation_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        validation_score = report.get('overall_score', 0)
        
        # Push score to XCom for downstream tasks
        context['ti'].xcom_push(key='validation_score', value=validation_score)
        
        # Fail if validation score too low
        if validation_score < 70:
            raise Exception(f"Validation score too low: {validation_score}%")
    
    return {'status': 'success', 'validation_score': validation_score}


def run_feature_engineering_task(**context):
    """
    Task 3: Run feature engineering pipeline
    Creates 87 advanced features from processed data
    """
    script_path = os.path.join(SCRIPTS_DIR, 'feature_engineering.py')
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Feature engineering failed: {result.stderr}")
    
    print(result.stdout)
    return {'status': 'success', 'features_created': 87}


def run_bias_detection_task(**context):
    """
    Task 4: Run bias detection pipeline
    Analyzes demographic bias across features
    """
    script_path = os.path.join(SCRIPTS_DIR, 'bias_detection.py')
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Bias detection failed: {result.stderr}")
    
    print(result.stdout)
    
    # Parse bias score and push to XCom
    import json
    report_path = os.path.join(PROJECT_ROOT, 'data-pipeline/logs/bias_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        bias_score = report.get('summary_metrics', {}).get('overall_bias_score', 0)
        
        context['ti'].xcom_push(key='bias_score', value=bias_score)
    
    return {'status': 'success', 'bias_score': bias_score}


def run_bias_mitigation_task(**context):
    """
    Task 5: Run automated bias mitigation pipeline
    Applies mitigation strategies to reduce bias
    """
    script_path = os.path.join(SCRIPTS_DIR, 'automated_bias_handler.py')
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Bias mitigation failed: {result.stderr}")
    
    print(result.stdout)
    
    # Parse mitigation results
    import json
    report_path = os.path.join(PROJECT_ROOT, 'data-pipeline/logs/bias_mitigation_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        mitigation_applied = report.get('mitigation_applied', False)
        context['ti'].xcom_push(key='mitigation_applied', value=mitigation_applied)
    
    return {'status': 'success', 'mitigation_applied': mitigation_applied}


def check_data_availability(**context):
    """
    Task 0: Check if raw data is available before starting pipeline
    """
    data_path = os.path.join(PROJECT_ROOT, 'data-pipeline/data/raw/mimic_discharge_labs.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Raw data not found at {data_path}. "
            "Please run data_acquisition.ipynb first."
        )
    
    # Check file size
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"Data file found: {file_size_mb:.2f} MB")
    
    context['ti'].xcom_push(key='data_size_mb', value=file_size_mb)
    
    return {'status': 'data_ready', 'size_mb': file_size_mb}


def generate_pipeline_summary(**context):
    """
    Task 6: Generate final pipeline summary report
    Combines metrics from all pipeline stages
    """
    ti = context['ti']
    
    # Pull metrics from previous tasks
    validation_score = ti.xcom_pull(task_ids='validate_data', key='validation_score') or 0
    bias_score = ti.xcom_pull(task_ids='detect_bias', key='bias_score') or 0
    mitigation_applied = ti.xcom_pull(task_ids='mitigate_bias', key='mitigation_applied') or False
    data_size = ti.xcom_pull(task_ids='check_data', key='data_size_mb') or 0
    
    summary = {
        'pipeline_execution_time': str(datetime.now()),
        'data_size_mb': data_size,
        'validation_score': validation_score,
        'validation_status': 'PASS' if validation_score >= 80 else 'FAIL',
        'bias_score': bias_score,
        'bias_status': 'ACCEPTABLE' if bias_score <= 10 else 'HIGH',
        'mitigation_applied': mitigation_applied,
        'pipeline_status': 'SUCCESS'
    }
    
    # Save summary
    import json
    summary_path = os.path.join(PROJECT_ROOT, 'data-pipeline/logs/airflow_pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*60)
    
    return summary


# Define the DAG

dag = DAG(
    'lab_lens_mimic_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline for MIMIC-III medical report processing',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['healthcare', 'mlops', 'bias-detection', 'mimic-iii']
)

# Define tasks

# Task 0: Check data availability
check_data = PythonOperator(
    task_id='check_data',
    python_callable=check_data_availability,
    dag=dag
)

# Task 1: Preprocessing
preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing_task,
    dag=dag
)

# Task 2: Validation
validate = PythonOperator(
    task_id='validate_data',
    python_callable=run_validation_task,
    dag=dag
)

# Task 3: Feature Engineering
engineer_features = PythonOperator(
    task_id='engineer_features',
    python_callable=run_feature_engineering_task,
    dag=dag
)

# Task 4: Bias Detection
detect_bias = PythonOperator(
    task_id='detect_bias',
    python_callable=run_bias_detection_task,
    dag=dag
)

# Task 5: Bias Mitigation
mitigate_bias = PythonOperator(
    task_id='mitigate_bias',
    python_callable=run_bias_mitigation_task,
    dag=dag
)

# Task 6: Generate Summary
generate_summary = PythonOperator(
    task_id='generate_summary',
    python_callable=generate_pipeline_summary,
    dag=dag
)

# Define task dependencies (pipeline flow)
check_data >> preprocess >> validate >> engineer_features >> detect_bias >> mitigate_bias >> generate_summary