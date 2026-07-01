from prefect import flow, task, get_run_logger
from pathlib import Path
import sys
import os

# Fix for emoji/log encoding issues on Windows
sys.stdout.reconfigure(encoding="utf-8")

# 1. Force Python to step inside the 'src' folder automatically
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# 2. Import your code (Notice the "src." is completely gone!)
from kedro_carprice_prediction.run_kedro_pipeline import run_pipeline

@task
def run_kedro_task(pipeline_name: str):
    logger = get_run_logger()
    logger.info(f"Running Kedro pipeline: `{pipeline_name}`")

    try:
        run_pipeline(pipeline_name)
        logger.info(f"Kedro pipeline `{pipeline_name}` finished successfully.")
    except Exception as e:
        logger.error(f"Pipeline `{pipeline_name}` failed: {str(e)}")
        raise

@flow(name="Full Training Processing Flow", description="Runs full preprocessing pipeline")
def flow_full_processing():
    run_kedro_task("production_full_train_process")

@flow(name="Training Flow")
def flow_train():
    run_kedro_task("production_full_prediction_process")

@flow(name="Orchestration Flow")
def full_pipeline():
    flow_full_processing() 
    flow_train() 
