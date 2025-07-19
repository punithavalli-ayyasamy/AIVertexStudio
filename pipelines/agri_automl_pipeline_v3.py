from kfp import dsl, components, compiler
from google.cloud import aiplatform

@dsl.pipeline(
    name='AgriAutoML Pipeline',
    description='End-to-end pipeline for agricultural yield prediction'
)
def agri_automl_pipeline(
    project_id: str,
    region: str,
    bucket_name: str,
    vision_dataset_uri: str,
    tabular_dataset_uri: str,
    min_accuracy: float = 0.8
):
    # Load components from local files
    preprocess_op = components.load_component_from_file('components/preprocess.py')
    train_vision_op = components.load_component_from_file('components/train_vision.py')
    train_tabular_op = components.load_component_from_file('components/train_tabular.py')
    deploy_op = components.load_component_from_file('components/deploy.py')

    # Data preprocessing
    preprocess = preprocess_op(
        vision_data=vision_dataset_uri,
        tabular_data=tabular_dataset_uri,
        bucket_name=bucket_name
    )

    # Train vision model
    train_vision = train_vision_op(
        project_id=project_id,
        region=region,
        dataset_uri=preprocess.outputs['vision_dataset'],
        min_accuracy=min_accuracy
    )

    # Train tabular model
    train_tabular = train_tabular_op(
        project_id=project_id,
        region=region,
        dataset_uri=preprocess.outputs['tabular_dataset'],
        min_accuracy=min_accuracy
    )

    # Deploy models
    deploy = deploy_op(
        project_id=project_id,
        region=region,
        vision_model=train_vision.outputs['model'],
        tabular_model=train_tabular.outputs['model']
    )

if __name__ == "__main__":
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=agri_automl_pipeline,
        package_path='agri_automl_pipeline.json'
    )
