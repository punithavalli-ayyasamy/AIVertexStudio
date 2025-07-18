from kfp.v2.dsl import component
from google.cloud import aiplatform

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-aiplatform']
)
def deploy(
    project_id: str,
    region: str,
    vision_model: str,
    tabular_model: str
) -> dict:
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

    # Deploy vision model
    vision_endpoint = aiplatform.Endpoint.create(display_name="crop-vision-endpoint")
    vision_model = aiplatform.Model(vision_model)
    vision_deploy = vision_model.deploy(
        endpoint=vision_endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

    # Deploy tabular model
    tabular_endpoint = aiplatform.Endpoint.create(display_name="crop-tabular-endpoint")
    tabular_model = aiplatform.Model(tabular_model)
    tabular_deploy = tabular_model.deploy(
        endpoint=tabular_endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

    return {
        'vision_endpoint': vision_endpoint.resource_name,
        'tabular_endpoint': tabular_endpoint.resource_name
    }
