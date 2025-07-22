from kfp import dsl
from kfp.dsl import Output, Input, Dataset, Model, Artifact
import os

@dsl.component(
    packages_to_install=[
        'google-cloud-storage>=2.0.0',
        'google-cloud-aiplatform==1.104.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0'
    ]
)
def preprocess_data(
    tabular_data: str,
    bucket_name: str,
    project_id: str,
    region: str,
    tabular_dataset: Output[Dataset]
):
    """Preprocess tabular data for crop yield prediction."""
    from google.cloud import storage
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Reading data from: %s", tabular_data)
    df = pd.read_csv(tabular_data)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['soil_quality'] = le.fit_transform(df['soil_quality'])
    
    # Save processed data
    output_uri = f"gs://{bucket_name}/processed_data/farming_data_processed.csv"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('processed_data/farming_data_processed.csv')
    blob.upload_from_string(df.to_csv(index=False))
    
    logger.info("Saved processed data to: %s", output_uri)
    
    # Save to the KFP output location
    with open(tabular_dataset.path, 'w') as f:
        f.write(output_uri)

@dsl.component(
    packages_to_install=[
        'google-cloud-aiplatform==1.104.0',
        'google-cloud-storage>=2.0.0'
    ]
)
def train_tabular_model(
    project_id: str,
    region: str,
    dataset: Input[Artifact],
    min_accuracy: float,
    model_info: Output[Model]
):
    """Train AutoML Tabular model for crop yield prediction."""
    from google.cloud import aiplatform
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    logger.info("Creating dataset from: %s", dataset)
    
    # Create dataset
    ai_dataset = aiplatform.TabularDataset.create(
        display_name="crop_tabular_dataset",
        gcs_source=dataset
    )
    
    logger.info("Training AutoML model")
    
    # Train model
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="crop_tabular_model",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse"
    )
    
    model = job.run(
        dataset=ai_dataset,
        target_column="yield",
        budget_milli_node_hours=1000,  # ~1 hour
        model_display_name="crop_tabular_model",
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1
    )
    
    # Evaluate model
    eval_metrics = model.get_model_evaluation()
    rmse = eval_metrics.metrics['rmse']
    logger.info("Model RMSE: %f", rmse)
    
    if rmse > min_accuracy:
        raise ValueError(f"Model RMSE {rmse} above threshold {min_accuracy}")
    
    # Save model info
    model_info_dict = {
        'resource_name': model.resource_name,
        'rmse': float(rmse)
    }
    
    with open(model_info.path, 'w') as f:
        f.write(model.resource_name)
    
    return model_info_dict

@dsl.component(
    packages_to_install=[
        'google-cloud-aiplatform==1.104.0'
    ]
)
def deploy_model(
    project_id: str,
    region: str,
    model: Input[Model],
    endpoint_info: Output[Artifact]
):
    """Deploy the trained model to an endpoint."""
    from google.cloud import aiplatform
    import logging
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Get the model resource name from the artifact
    with open(model.path, 'r') as f:
        model_resource_name = f.read().strip()
    
    # Get the model
    model = aiplatform.Model(model_resource_name)
    
    # Deploy the model
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1
    )
    
    endpoint_info_dict = {
        'endpoint_name': endpoint.resource_name,
        'display_name': endpoint.display_name
    }
    
    # Save endpoint info
    with open(endpoint_info.path, 'w') as f:
        json.dump(endpoint_info_dict, f)
    
    return endpoint_info_dict
