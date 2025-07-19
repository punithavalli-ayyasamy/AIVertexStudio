from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Artifact
from google.cloud import aiplatform
import json

@dsl.component
def train_vision(
    project_id: str,
    region: str,
    dataset: str,
    min_accuracy: float
) -> dict:
    """Train AutoML Vision model for crop analysis.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        dataset_uri: URI of the processed vision dataset
        min_accuracy: Minimum required accuracy
    
    Returns:
        dict with model information
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Create dataset
    ai_dataset = aiplatform.ImageDataset.create(
        display_name="crop_vision_dataset",
        gcs_source=dataset,
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification
    )

    # Train model
    job = aiplatform.AutoMLImageTrainingJob(
        display_name="crop_vision_model",
        prediction_type="image_classification"
    )

    ai_model = job.run(
        dataset=ai_dataset,
        target_column="yield",
        budget_milli_node_hours=83.33,  # 5 minutes
        model_display_name="crop_vision_model",
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1
    )

    # Evaluate model
    eval_metrics = ai_model.get_model_evaluation()
    if eval_metrics.metrics['auRoc'] < min_accuracy:
        raise ValueError(f"Model accuracy {eval_metrics.metrics['auRoc']} below threshold {min_accuracy}")

    return {
        'model': ai_model.resource_name,
        'accuracy': float(eval_metrics.metrics['auRoc'])
    }
