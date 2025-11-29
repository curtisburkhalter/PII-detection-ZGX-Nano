# PII Masking Demo - Fine-tuning Comparison

## Overview
This demonstration showcases the effectiveness of fine-tuning large language models for PII (Personally Identifiable Information) detection and masking. The demo compares a base TinyLlama model against a fine-tuned Qwen2.5-32B model, highlighting the dramatic improvement in accuracy and consistency achieved through fine-tuning.

## Purpose
This demo is designed for HP ZGX Nano sales and marketing teams to demonstrate the power of local AI model fine-tuning capabilities at customer events and presentations.

## System Requirements
- HP ZGX Nano 
- Python 3.8+
- CUDA-capable GPU with at least 40GB VRAM (recommended)
- 8GB+ system RAM
- Network access for remote demo access

## Directory Structure
```
gartner-pii-demo/
├── backend/
│   ├── main.py                    # FastAPI server handling model inference
│   ├── requirements.txt           # Python dependencies
│   ├── offline_responses.json     # Fallback responses for offline mode
│   └── models/                    # Directory for GGUF model files (created during setup)
├── frontend/
│   └── index.html                 # Web interface for the demo
├── install.sh                     # Installation script
├── download_models.sh             # Model download script
├── start_demo_remote.sh          # Demo startup script
└── .gitattributes                # Git LFS configuration for large files
```

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-org/gartner-pii-demo.git
cd gartner-pii-demo
```

### Step 2: Set Up Virtual Environment
**IMPORTANT**: The demo expects a virtual environment named `new-ft-env` in the project root directory.

```bash
python3 -m venv new-ft-env
source new-ft-env/bin/activate
```

### Step 3: Run Installation Script
```bash
chmod +x install.sh
./install.sh
```

This script will:
- Verify Python installation
- Install required Python packages
- Create the offline response database
- Set up the backend directory structure

### Step 4: Download Models
```bash
chmod +x download_models.sh
./download_models.sh
```

This downloads the pre-quantized GGUF models from the S3 bucket:
- `pii_detector_Q4_K_M.gguf` - Fine-tuned Qwen2.5-32B model
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` - Base TinyLlama model

## Configuration for Sales Teams

### CRITICAL: Update Model Paths
Sales teams must update the model paths in `backend/main.py` to match their local setup:

1. Open `backend/main.py`
2. Locate lines 24-25:
```python
BASE_MODEL_PATH = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
FINETUNED_MODEL_PATH = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/llama.cpp/models/pii_detector_Q4_K_M.gguf"
```

3. Update these paths to point to your downloaded models:
```python
BASE_MODEL_PATH = "/path/to/your/gartner-pii-demo/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
FINETUNED_MODEL_PATH = "/path/to/your/gartner-pii-demo/models/pii_detector_Q4_K_M.gguf"
```

### Network Configuration
**IMPORTANT**: Sales and marketing teams must update the server IP address to match their demo system's IP address.

The demo is configured for remote access. When running on your demo system:

1. Open `start_demo_remote.sh`
2. Locate the fallback IP address (line 13):
```bash
SERVER_IP="192.168.10.117"  # Fallback to your provided IP
```
3. **Update this to your demo system's actual IP address**

To find your system's IP address:
```bash
hostname -I | awk '{print $1}'
```

Note: The script will attempt to automatically detect the IP address, but the fallback IP must be updated to ensure proper operation if automatic detection fails.

## Running the Demo

### Start the Demo
```bash
chmod +x start_demo_remote.sh
./start_demo_remote.sh
```

The script will:
1. Clean up any existing processes on ports 8000 and 8080
2. Start the FastAPI backend server on port 8000
3. Start the frontend web server on port 8080
4. Display the access URLs for the demo

### Access the Demo
From any browser on the same network:
```
http://[YOUR_SERVER_IP]:8080
```

The terminal will display the actual IP address to use.

### Using the Demo Interface

1. **Load Models**: Click the "Load Models" button first (this takes 1-2 minutes)
2. **Enter Text**: Type or paste text containing PII, or use the sample buttons
3. **Process**: Click "Process & Compare" to see both models' outputs
4. **Compare Results**: Observe the difference between the base and fine-tuned models

## Demo Flow for Presentations

1. **Introduction**: Explain the importance of PII protection in enterprise environments
2. **Show Base Model Performance**: Demonstrate how a generic model struggles with PII detection
3. **Show Fine-tuned Model**: Highlight the accuracy improvement from fine-tuning
4. **Emphasize Local Processing**: Point out that all processing happens locally on the HP ZGX Nano
5. **Performance Metrics**: Show inference time comparisons between models

## Troubleshooting

### Models Not Loading
- Verify model files exist in the correct directory
- Check that paths in `main.py` are absolute and correct
- Ensure sufficient GPU memory is available

### Cannot Access Web Interface
- Verify the server IP address is correct
- Check firewall settings allow connections on ports 8000 and 8080
- Ensure both frontend and backend services are running (check terminal output)

### Virtual Environment Issues
- The demo expects a virtual environment named `new-ft-env`
- If missing, create it as shown in Step 2 of Installation

### Port Already in Use
- The startup script attempts to kill existing processes
- If issues persist, manually check and kill processes:
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

## Stopping the Demo
Press `Ctrl+C` in the terminal running the demo to gracefully shutdown all services.

## Offline Mode
The demo includes an offline test endpoint that uses regex-based masking if models fail to load. This ensures basic functionality for demonstrations even without GPU resources.

## Notes for Sales Teams

- Allow 3-5 minutes for initial model loading during setup so it's best to run this a few minutes before you have to present
- Test the demo before customer presentations
- Have sample PII text ready (the interface includes examples)
- The fine-tuned model will consistently outperform the base model
- Emphasize the local processing capability of the HP ZGX Nano
- All processing happens on-device, ensuring data privacy

## Support
For issues or questions about this demo, please contact Curtis Burkhalter at curtis.burkhalter@hp.com
