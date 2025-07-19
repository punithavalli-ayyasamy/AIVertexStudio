from kfp.v2.dsl import component, Input, Output, Model, Artifact
from google.cloud import aiplatform
import json

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-aiplatform']
)
def deploy(
    project_id: str,
    region: str,
    vision_model: Input[Model],
    tabular_model: Input[Model],
    endpoint: Output[Artifact]
):
    """Deploy trained models to endpoints.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        vision_model: Vision model resource name
        tabular_model: Tabular model resource name
    
    Returns:
        dict with endpoint information
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Load model info
    with open(vision_model.path, 'r') as f:
        vision_model_info = json.load(f)
    
    # Deploy vision model
    vision_endpoint = aiplatform.Endpoint.create(display_name="crop-vision-endpoint")
    vision_model_obj = aiplatform.Model(vision_model_info['resource_name'])
    vision_deploy = vision_model_obj.deploy(
        endpoint=vision_endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

    # Load model info
    with open(tabular_model.path, 'r') as f:
        tabular_model_info = json.load(f)
    
    # Deploy tabular model
    tabular_endpoint = aiplatform.Endpoint.create(display_name="crop-tabular-endpoint")
    tabular_model_obj = aiplatform.Model(tabular_model_info['resource_name'])
    tabular_deploy = tabular_model_obj.deploy(
        endpoint=tabular_endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

    # Save endpoint info
    endpoint_info = {
        'vision_endpoint': vision_endpoint.resource_name,
        'tabular_endpoint': tabular_endpoint.resource_name
    }
    
    with open(endpoint.path, 'w') as f:
        json.dump(endpoint_info, f)
