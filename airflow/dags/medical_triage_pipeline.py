"""Airflow orchestration for the reproducible medical-triage model pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="medical_triage_training_pipeline",
    description="Prepare data, train, export to ONNX, and benchmark both runtimes.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["medical-triage", "mlops"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_and_validate_data",
        bash_command="python training/prepare_data.py",
        cwd=PROJECT_DIR,
    )

    train_model = BashOperator(
        task_id="train_and_evaluate_model",
        bash_command="python training/train.py",
        cwd=PROJECT_DIR,
    )

    export_onnx = BashOperator(
        task_id="export_model_to_onnx",
        bash_command="python training/export_onnx.py",
        cwd=PROJECT_DIR,
    )

    benchmark_models = BashOperator(
        task_id="benchmark_sklearn_and_onnx",
        bash_command="python training/benchmark.py",
        cwd=PROJECT_DIR,
    )

    prepare_data >> train_model >> export_onnx >> benchmark_models
