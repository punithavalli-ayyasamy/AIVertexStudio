from kfp.dsl import component

@component
def deploy(
    project_id: str,
    region: str,
    vision_model: dict,
    tabular_model: dict
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
    vision_model_obj = aiplatform.Model(vision_model['model'])
    vision_deploy = vision_model_obj.deploy(
        endpoint=vision_endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

    # Deploy tabular model
    tabular_endpoint = aiplatform.Endpoint.create(display_name="crop-tabular-endpoint")
    tabular_model_obj = aiplatform.Model(tabular_model['model'])
    tabular_deploy = tabular_model_obj.deploy(
        endpoint=tabular_endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3
    )

    return {
        'vision_endpoint': vision_endpoint.resource_name,
        'tabular_endpoint': tabular_endpoint.resource_name
    }
