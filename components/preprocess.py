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
    bucket_name: str
) -> dict:
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
    vision_output_uri = f"gs://{bucket_name}/processed/vision_data.csv"
    tabular_output_uri = f"gs://{bucket_name}/processed/tabular_data.csv"
    
    # Upload processed data to GCS
    bucket = storage_client.bucket(bucket_name)
    
    # For demo, just write placeholder data
    vision_blob = bucket.blob('processed/vision_data.csv')
    vision_blob.upload_from_string(f"Processed vision data from {vision_data}")
    
    tabular_blob = bucket.blob('processed/tabular_data.csv')
    tabular_blob.upload_from_string(f"Processed tabular data from {tabular_data}")
    
    return {
        'vision_dataset': vision_output_uri,
        'tabular_dataset': tabular_output_uri
    }
