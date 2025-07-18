# AgriAutoML - Vertex AI Studio Implementation

This project implements the AgriAutoML crop yield prediction system using Vertex AI Studio components.

## Project Overview

The project uses the following Vertex AI components:
1. Custom Training Jobs
2. Model Registry
3. Pipelines
4. Endpoints
5. Feature Store

## Directory Structure

```
AIVertexStudio/
├── pipelines/           # Kubeflow pipeline definitions
├── components/          # Custom pipeline components
├── training/           # Model training code
├── prediction/         # Prediction service code
└── config/            # Configuration files
```

## Setup Instructions

1. **Enable Required APIs:**
   ```bash
   gcloud services enable \
     aiplatform.googleapis.com \
     artifactregistry.googleapis.com \
     containerregistry.googleapis.com
   ```

2. **Set Environment Variables:**
   ```bash
   export PROJECT_ID="your-project-id"
   export REGION="your-region"
   export BUCKET_NAME="your-bucket-name"
   ```

3. **Create Cloud Storage Bucket:**
   ```bash
   gsutil mb -l $REGION gs://$BUCKET_NAME
   ```

## Pipeline Components

1. **Data Processing:**
   - Image preprocessing for satellite data
   - Tabular data preparation
   - Feature engineering

2. **Model Training:**
   - Vision model for satellite imagery
   - Tabular model for crop data
   - Text model for query processing

3. **Model Deployment:**
   - Model versioning
   - Endpoint creation
   - Prediction service deployment

## Usage

1. **Build and Upload Pipeline:**
   ```python
   from pipeline import build_pipeline
   pipeline = build_pipeline()
   pipeline.compile("agri_automl_pipeline.json")
   ```

2. **Run Pipeline:**
   ```python
   from google.cloud import aiplatform
   job = aiplatform.PipelineJob(
       display_name="agri-automl-pipeline",
       template_path="agri_automl_pipeline.json",
       pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root"
   )
   job.run()
   ```
