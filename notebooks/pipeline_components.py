from kfp.v2.dsl import component
from google.cloud import storage, aiplatform
from PIL import Image
import io
import numpy as np
import pandas as pd
import json
import os

@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-storage",
        "pandas",
        "numpy",
        "pillow",
        "protobuf"
    ]
)
def preprocess_data(vision_data: str, tabular_data: str, bucket_name: str) -> tuple[str, str]:
    """
    Preprocess vision and tabular data for training
    
    Args:
        vision_data: GCS URI for vision dataset
        tabular_data: GCS URI for tabular dataset
        bucket_name: GCS bucket for processed data
        
    Returns:
        tuple: (vision_dataset_uri, tabular_dataset_uri)
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Process vision data
    def process_image(image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))  # Standard size for many vision models
        return np.array(img)

    # Process tabular data
    def process_tabular(df):
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Feature engineering
        if "planting_date" in df.columns:
            df["planting_date"] = pd.to_datetime(df["planting_date"])
            df["planting_month"] = df["planting_date"].dt.month
            df["planting_day"] = df["planting_date"].dt.day
        
        return df

    # Process and save datasets
    vision_blob = bucket.blob('processed_vision_data.txt')
    vision_blob.upload_from_string(vision_data)
    vision_output_uri = f"gs://{bucket_name}/{vision_blob.name}"

    tabular_blob = bucket.blob('processed_tabular_data.csv')
    tabular_blob.upload_from_string(tabular_data)
    tabular_output_uri = f"gs://{bucket_name}/{tabular_blob.name}"
    
    return vision_output_uri, tabular_output_uri

@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform"
    ]
)
def train_vision_model(project_id: str, region: str, dataset: str, min_accuracy: float) -> dict:
    """
    Train AutoML Vision model for crop analysis
    
    Args:
        project_id: GCP project ID
        region: GCP region
        dataset: URI of the processed vision dataset
        min_accuracy: Minimum required accuracy
        
    Returns:
        dict: Model information including resource name and metrics
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Create dataset
    ai_dataset = aiplatform.ImageDataset.create(
        display_name="crop_vision_dataset",
        gcs_source=dataset
    )

    # Train model
    job = aiplatform.AutoMLImageTrainingJob(
        display_name="crop_vision_model",
        prediction_type="classification",
        budget_milli_node_hours=83,  # Approximately 5 minutes
        model_type="CLOUD",
        base_model=None
    )

    # Run the training job
    ai_model = job.run(
        dataset=ai_dataset,
        budget_milli_node_hours=83,  # 5 minutes for testing
        training_filter_split="",  # No filter
        model_display_name="crop_vision_model",
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1
    )

    # Get model evaluation
    eval_metrics = ai_model.list_model_evaluations()[0]

    # Check if model meets accuracy threshold
    if eval_metrics.metrics['auRoc'] < min_accuracy:
        raise ValueError(f"Model accuracy {eval_metrics.metrics['auRoc']} below threshold {min_accuracy}")

    # Return model info
    model_info = {
        'model': ai_model.resource_name,
        'accuracy': float(eval_metrics.metrics['auRoc'])
    }
    return model_info

@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform"
    ]
)
def train_tabular_model(project_id: str, region: str, dataset: str, min_accuracy: float) -> dict:
    """
    Train AutoML Tabular model for crop yield prediction
    
    Args:
        project_id: GCP project ID
        region: GCP region
        dataset: URI of the processed tabular dataset
        min_accuracy: Minimum required accuracy (RMSE threshold)
        
    Returns:
        dict: Model information including resource name and metrics
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Create dataset
    ai_dataset = aiplatform.TabularDataset.create(
        display_name="crop_tabular_dataset",
        gcs_source=dataset
    )

    # Train model
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="crop_tabular_model",
        optimization_objective="minimize-rmse",
        column_transformations=[
            {"numeric": {"column_name": "field_size"}},
            {"numeric": {"column_name": "rainfall"}},
            {"numeric": {"column_name": "temperature"}},
            {"categorical": {"column_name": "location"}},
            {"categorical": {"column_name": "crop_type"}},
            {"timestamp": {"column_name": "date"}}
        ],
        target_column="yield",
        budget_milli_node_hours=83,  # Approximately 5 minutes
        optimization_prediction_type="regression",
        additional_experiments=["enable_model_compression"]
    )

    # Run the training job
    ai_model = job.run(
        dataset=ai_dataset,
        model_display_name="crop_yield_model",
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1
    )

    # Get model evaluation
    eval_metrics = ai_model.list_model_evaluations()[0]

    # Check if model meets accuracy threshold
    if eval_metrics.metrics['rmse'] > min_accuracy:
        raise ValueError(f"Model RMSE {eval_metrics.metrics['rmse']} above threshold {min_accuracy}")

    # Return model info
    model_info = {
        'model': ai_model.resource_name,
        'rmse': float(eval_metrics.metrics['rmse'])
    }
    return model_info

@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform"
    ]
)
def deploy_models(project_id: str, region: str, vision_model: dict, tabular_model: dict) -> tuple[str, str]:
    """
    Deploy trained models to endpoints
    
    Args:
        project_id: GCP project ID
        region: GCP region
        vision_model: Vision model information
        tabular_model: Tabular model information
        
    Returns:
        tuple: (vision_endpoint_name, tabular_endpoint_name)
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Deploy vision model
    vision_model_resource = aiplatform.Model(vision_model['model'])
    vision_endpoint = vision_model_resource.deploy(
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=1
    )

    # Deploy tabular model
    tabular_model_resource = aiplatform.Model(tabular_model['model'])
    tabular_endpoint = tabular_model_resource.deploy(
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=1
    )

    return vision_endpoint.resource_name, tabular_endpoint.resource_name
