import os
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the development environment."""
    # Create virtual environment
    print("Creating virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], check=True)

    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_script = "source venv/bin/activate"

    print("Installing requirements...")
    if os.name == 'nt':
        subprocess.run(f"{activate_script} && pip install -r requirements.txt", shell=True, check=True)
    else:
        subprocess.run(["bash", "-c", f"{activate_script} && pip install -r requirements.txt"], check=True)

def create_env_template():
    """Create .env template file."""
    env_template = """# GCP Configuration
PROJECT_ID=agrifingcpflow-465809
REGION=us-central1
BUCKET_NAME=agrifinstorage

# Service Account
GOOGLE_APPLICATION_CREDENTIALS=C:/Users/charl/CascadeProjects/AgriFinGCPFlow/agrifingcpflow.json

# Vertex AI Configuration
PIPELINE_ROOT=gs://${BUCKET_NAME}/pipeline_root
MODEL_DISPLAY_NAME=agri-automl-model
ENDPOINT_NAME=agri-automl-endpoint

# Training Parameters
TRAIN_BUDGET_HOURS=0.0833  # 5 minutes
MIN_ACCURACY=0.8
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)

def main():
    """Main setup function."""
    print("Setting up AgriAutoML Vertex AI environment...")
    
    # Create necessary directories if they don't exist
    directories = ['data', 'models', 'configs']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Set up Python environment
    setup_environment()
    
    # Create .env template
    create_env_template()
    
    print("""
Setup complete! Next steps:
1. Copy .env.template to .env and fill in your configuration
2. Place your service account key file in a secure location
3. Update the GOOGLE_APPLICATION_CREDENTIALS path in .env
4. Open notebooks/pipeline_execution.ipynb to run the pipeline
""")

if __name__ == '__main__':
    main()
