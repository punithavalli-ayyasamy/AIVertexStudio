import os
from datetime import datetime
from kfp import dsl, compiler
from kfp.dsl import Input, Output, OutputPath, InputPath, component, Dataset, Model, Artifact
from kfp.client import Client

# Local configuration
PROJECT_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Create sample data
def create_sample_data():
    """Create sample datasets locally"""
    sample_csv = """date,location,crop_type,field_size,rainfall,temperature,yield
2025-01-01,Iowa,corn,5.0,750,25,150
2025-01-15,Kansas,wheat,3.5,500,22,120
2025-02-01,Nebraska,soybean,4.2,600,24,130"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(DATA_DIR, timestamp)
    os.makedirs(data_path, exist_ok=True)
    
    tabular_path = os.path.join(data_path, 'crop_data.csv')
    with open(tabular_path, 'w') as f:
        f.write(sample_csv)
    
    vision_path = os.path.join(data_path, 'vision_data.txt')
    with open(vision_path, 'w') as f:
        f.write("Dummy vision data for testing")
    
    return vision_path, tabular_path

# Mock components
@component
def mock_preprocess(
    vision_data: str,
    tabular_data: str,
    vision_dataset: Output[Dataset],
    tabular_dataset: Output[Dataset]
):
    print(f"Preprocessing vision data from: {vision_data}")
    print(f"Preprocessing tabular data from: {tabular_data}")
    with open(vision_dataset.path, 'w') as f:
        f.write(vision_data)
    with open(tabular_dataset.path, 'w') as f:
        f.write(tabular_data)

@component
def mock_train_vision(
    dataset: Input[Dataset],
    min_accuracy: float,
    model: Output[Model]
):
    print(f"Training vision model with data from: {dataset.path}")
    print(f"Target accuracy: {min_accuracy}")
    with open(model.path, 'w') as f:
        f.write('mock_vision_model')

@component
def mock_train_tabular(
    dataset: Input[Dataset],
    min_accuracy: float,
    model: Output[Model]
):
    print(f"Training tabular model with data from: {dataset.path}")
    print(f"Target accuracy: {min_accuracy}")
    with open(model.path, 'w') as f:
        f.write('mock_tabular_model')

@component
def mock_deploy(
    vision_model: Input[Model],
    tabular_model: Input[Model],
    endpoint: Output[Artifact]
):
    print(f"Deploying vision model: {vision_model.path}")
    print(f"Deploying tabular model: {tabular_model.path}")
    with open(endpoint.path, 'w') as f:
        f.write('mock_endpoint')

# Define pipeline
@dsl.pipeline(
    name='Local AgriAutoML Pipeline',
    description='Local testing version of agricultural yield prediction pipeline'
)
def local_agri_automl_pipeline(
    vision_dataset_path: str,
    tabular_dataset_path: str,
    min_accuracy: float = 0.8
):
    # Preprocess data
    preprocess = mock_preprocess(
        vision_data=vision_dataset_path,
        tabular_data=tabular_dataset_path
    )

    # Train vision model
    train_vision = mock_train_vision(
        dataset=preprocess.outputs['vision_dataset'],
        min_accuracy=min_accuracy
    )

    # Train tabular model
    train_tabular = mock_train_tabular(
        dataset=preprocess.outputs['tabular_dataset'],
        min_accuracy=min_accuracy
    )

    # Deploy models
    deploy = mock_deploy(
        vision_model=train_vision.outputs['model'],
        tabular_model=train_tabular.outputs['model']
    )

def main():
    # Create sample data
    vision_path, tabular_path = create_sample_data()
    print(f"Vision Dataset Path: {vision_path}")
    print(f"Tabular Dataset Path: {tabular_path}")

    # Create a client (will use local or default config)
    client = Client()

    # Create an experiment
    exp_name = 'local-agri-automl-test'
    try:
        experiment = client.create_experiment(name=exp_name)
    except:
        experiment = client.get_experiment(experiment_name=exp_name)

    # Create a pipeline run
    run = client.create_run_from_pipeline_func(
        pipeline_func=local_agri_automl_pipeline,
        arguments={
            'vision_dataset_path': vision_path,
            'tabular_dataset_path': tabular_path,
            'min_accuracy': 0.8
        },
        experiment_name=exp_name,
        run_name=f'local-test-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    print(f"\nPipeline run created with ID: {run.run_id}")
    print("You can monitor the run in the Kubeflow Pipelines UI")


if __name__ == '__main__':
    main()
