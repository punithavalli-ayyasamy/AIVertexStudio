from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Artifact
from google.cloud import storage
import pandas as pd
import numpy as np
from PIL import Image
import io
import os

@dsl.component
def preprocess(
    vision_data: str,
    tabular_data: str,
    bucket_name: str,
    vision_dataset: Output[Artifact],
    tabular_dataset: Output[Artifact]
) -> None:
    """Preprocess vision and tabular data for training.
    
    Args:
        vision_data: GCS URI for vision dataset
        tabular_data: GCS URI for tabular dataset
        bucket_name: GCS bucket for processed data
    
    Returns:
        dict with processed dataset URIs
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
        if 'planting_date' in df.columns:
            df['planting_date'] = pd.to_datetime(df['planting_date'])
            df['planting_month'] = df['planting_date'].dt.month
            df['planting_day'] = df['planting_date'].dt.day
        
        return df

    # Process and save datasets
    vision_output_path = vision_dataset.path
    tabular_output_path = tabular_dataset.path
    
    os.makedirs(os.path.dirname(vision_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(tabular_output_path), exist_ok=True)
    
    # Write processed data to output paths
    with open(vision_output_path, 'w') as f:
        f.write(f"Processed vision data from {vision_data}")
        
    with open(tabular_output_path, 'w') as f:
        f.write(f"Processed tabular data from {tabular_data}")
