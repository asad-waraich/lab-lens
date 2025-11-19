from google.cloud import storage
import tarfile
import os
from datetime import datetime

def push_to_gcp_registry():
    """Push model to GCP Artifact Registry"""
    
    # Your GCP settings
    PROJECT_ID = "your-project-id"
    BUCKET_NAME = "your-model-bucket"
    
    model_path = "../saved_models/content/saved_model"
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create tar archive
    tar_filename = f"llama_medical_v{version}.tar.gz"
    with tarfile.open(tar_filename, 'w:gz') as tar:
        tar.add(model_path, arcname='model')
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{tar_filename}")
    blob.upload_from_filename(tar_filename)
    
    print(f"✅ Model pushed to gs://{BUCKET_NAME}/models/{tar_filename}")
    
    # Clean up
    os.remove(tar_filename)

if __name__ == "__main__":
    push_to_gcp_registry()
