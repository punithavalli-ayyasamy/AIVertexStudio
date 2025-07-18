from kfp.v2.dsl import component
from google.cloud import aiplatform

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-aiplatform']
)
def train_tabular(
    project_id: str,
    region: str,
    dataset_uri: str,
    min_accuracy: float
) -> dict:
    """Train AutoML Tabular model for crop yield prediction.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        dataset_uri: URI of the processed tabular dataset
        min_accuracy: Minimum required accuracy
    
    Returns:
        dict with model information
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Create dataset
    dataset = aiplatform.TabularDataset.create(
        display_name="crop_tabular_dataset",
        gcs_source=dataset_uri
    )

    # Train model
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="crop_tabular_model",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse"
    )

    model = job.run(
        dataset=dataset,
        target_column="yield",
        budget_milli_node_hours=83.33,  # 5 minutes
        model_display_name="crop_tabular_model",
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1
    )

    # Evaluate model
    eval_metrics = model.get_model_evaluation()
    if eval_metrics.metrics['rmse'] > min_accuracy:
        raise ValueError(f"Model RMSE {eval_metrics.metrics['rmse']} above threshold {min_accuracy}")

    return {
        'model': model.resource_name,
        'rmse': float(eval_metrics.metrics['rmse'])
    }
