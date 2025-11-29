#!/bin/bash

# Create the models directory if it doesn't exist
mkdir -p models

# Download models from your S3 bucket
echo "Downloading pii_detector model..."
wget https://finetuning-demo-models.s3.amazonaws.com/pii_detector_Q4_K_M.gguf -O models/pii_detector_Q4_K_M.gguf

echo "Downloading tinyllama model..."
wget https://finetuning-demo-models.s3.amazonaws.com/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

echo "Models downloaded successfully!"