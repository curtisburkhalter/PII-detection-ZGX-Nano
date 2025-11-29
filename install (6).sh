#!/bin/bash

echo "======================================"
echo "PII Masking Demo Installer"
echo "======================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install with: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

echo "✔ Python 3 found: $(python3 --version)"

# Use existing virtual environment
echo ""
echo "Using existing virtual environment: new-ft-env"
if [ ! -d "new-ft-env" ]; then
    echo "❌ Virtual environment 'new-ft-env' not found!"
    echo "Please ensure 'new-ft-env' exists in the current directory"
    exit 1
else
    echo "✔ Found virtual environment 'new-ft-env'"
fi

# Activate virtual environment
source new-ft-env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install backend dependencies
echo ""
echo "Installing dependencies..."
cd backend
pip install -r requirements.txt

echo ""
echo "======================================"
echo "⚠️  Model Setup Information"
echo "======================================"
echo ""
echo "This demo uses two large language models:"
echo "1. Base Model: Qwen/Qwen2.5-32B-Instruct"
echo "2. Finetuned Model: pii_detector_Qwen32B_FTmerged"
echo ""
echo "IMPORTANT:"
echo "- Models will be loaded when you click 'Load Models' in the web interface"
echo "- Each model is ~32GB, so loading may take several minutes"
echo "- Ensure you have sufficient GPU memory (recommended: 40GB+ VRAM)"
echo "- Models will use 8-bit quantization to reduce memory requirements"
echo ""
echo "If your finetuned model is in a local directory, update the path in:"
echo "  backend/main.py -> FINETUNED_MODEL_NAME variable"
echo ""

cd ..

# Create offline_responses.json for fallback
echo ""
echo "Creating offline response database..."
cd backend
cat > offline_responses.json <<'EOF'
{
    "name": "Names should be replaced with [NAME] to protect identity",
    "phone": "Phone numbers should be replaced with [PHONE]",
    "email": "Email addresses should be replaced with [EMAIL]",
    "ssn": "Social Security Numbers should be replaced with [SSN]",
    "address": "Physical addresses should be replaced with [ADDRESS]",
    "date": "Dates of birth should be replaced with [DATE]",
    "credit": "Credit card numbers should be replaced with [CREDIT_CARD]",
    "id": "ID numbers should be replaced with [ID]",
    "default": "This text contains PII that should be masked for privacy protection"
}
EOF
echo "✔ Offline response database created"

cd ..

echo ""
echo "======================================"
echo "✅ Installation Complete!"
echo "======================================"
echo ""
echo "To start the demo:"
echo "  ./start_demo_remote.sh"
echo ""
echo "Then access from your Windows laptop:"
echo "  http://YOUR_SERVER_IP:8080"
echo ""
echo "Note: Update the IP address in start_demo_remote.sh to match your server"
echo ""