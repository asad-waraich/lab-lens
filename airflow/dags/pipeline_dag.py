from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os

# Define the base path to your data-pipeline
DATA_PIPELINE_PATH = "/Users/asadullahwaraich/Library/CloudStorage/OneDrive-HigherEducationCommission/Desktop/NEU Fall 2025/MLOps/Project/Github/lab-lens/data-pipeline"

# Define default arguments
default_args = {
    "owner": "lab-lens-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 28),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

# Define the DAG
with DAG(
    dag_id="lab_lens_pipeline",
    default_args=default_args,
    schedule_interval="@daily",  # Run daily
    catchup=False,
    description="Complete MLOps pipeline for MIMIC-III data processing",
    tags=['mlops', 'mimic', 'bias-detection']
) as dag:

    # Task 1: Pull data with DVC
    pull_data = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {DATA_PIPELINE_PATH} && dvc pull",
    )

    # Task 2: Run preprocessing
    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command=f"cd {DATA_PIPELINE_PATH} && python scripts/preprocessing.py",
    )

    # Task 3: Run feature engineering
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd {DATA_PIPELINE_PATH} && python scripts/feature_engineering.py --input data/processed/processed_discharge_summaries.csv --output data/processed/mimic_features_advanced.csv --with_sections",
    )

    # Task 4: Run validation
    validation = BashOperator(
        task_id="validation",
        bash_command=f"cd {DATA_PIPELINE_PATH} && python scripts/validation.py",
    )

    # Task 5: Run bias detection
    bias_detection = BashOperator(
        task_id="bias_detection",
        bash_command=f"cd {DATA_PIPELINE_PATH} && python scripts/bias_detection.py",
    )

    # Task 6: Push results with DVC
    push_results = BashOperator(
        task_id="dvc_push",
        bash_command=f"cd {DATA_PIPELINE_PATH} && dvc push",
    )

    # Define execution order
    pull_data >> preprocessing >> feature_engineering >> validation >> bias_detection >> push_results