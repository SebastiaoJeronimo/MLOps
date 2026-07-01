from prefect import serve
from prefect.client.schemas.schedules import CronSchedule
from kedro_prefect_flow import full_pipeline, flow_full_processing, flow_train 

if __name__ == "__main__":
    
    print("Building deployments...")


    # Deployment 3: Model Training
    dep_train = flow_train.to_deployment(
        name="model-train-weekly",
        schedule=CronSchedule(cron="0 6 * * 1", timezone="Europe/Lisbon"),
    )

    # Deployment 4: Full Pipeline
    dep_full = full_pipeline.to_deployment(
        name="full_pipeline",
    )

    print("Deployments built successfully! Starting the server...")

    # Run all deployments simultaneously!
    serve(
        dep_train, 
        dep_full
    )