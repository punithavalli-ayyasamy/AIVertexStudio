FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required packages
RUN pip install --no-cache-dir \
    google-cloud-aiplatform \
    google-cloud-storage \
    google-cloud-bigquery \
    google-auth \
    google-auth-httplib2 \
    google-api-python-client \
    pandas \
    numpy \
    pillow \
    scikit-learn \
    tensorflow \
    kfp==1.8.22

# Set working directory
WORKDIR /app

# Copy any additional files if needed
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/key.json
