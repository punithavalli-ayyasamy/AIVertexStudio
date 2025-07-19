FROM python:3.9-slim

# Install required packages
RUN pip install --no-cache-dir \
    google-cloud-aiplatform \
    google-cloud-storage \
    pandas \
    numpy \
    pillow

# Set working directory
WORKDIR /app

# Copy any additional files if needed
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
