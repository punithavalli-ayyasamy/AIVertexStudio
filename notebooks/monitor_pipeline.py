from google.cloud import aiplatform
import time

def monitor_pipeline_job(job_id):
    """
    Monitor a Vertex AI Pipeline job and print its status.
    
    Args:
        job_id: The full resource name of the pipeline job
    """
    pipeline_job = aiplatform.PipelineJob.get(job_id)
    
    print(f"Pipeline Job Name: {pipeline_job.display_name}")
    print(f"Pipeline Job ID: {pipeline_job.resource_name}")
    print(f"Pipeline Job State: {pipeline_job.state}")
    
    while pipeline_job.state in ['PIPELINE_STATE_RUNNING', 'PIPELINE_STATE_PENDING']:
        print(f"Current state: {pipeline_job.state}")
        print("Pipeline is still running... checking again in 60 seconds")
        time.sleep(60)
        pipeline_job = aiplatform.PipelineJob.get(job_id)
    
    print("\nPipeline completed!")
    print(f"Final state: {pipeline_job.state}")
    
    if pipeline_job.state == 'PIPELINE_STATE_SUCCEEDED':
        print("\nPipeline succeeded! You can now:")
        print("1. Check the deployed model endpoints")
        print("2. Make predictions using the deployed models")
        print("3. Monitor model performance")
    else:
        print("\nPipeline failed or was cancelled. Check the error messages in the Google Cloud Console.")
