#!/bin/bash

clear
echo "======================================"
echo "🔒 PII Masking Demo (Remote Access)"
echo "======================================"
echo ""

# Get the hostname/IP of the Linux server
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "Server Information:"
echo "  Hostname/IP: $SERVER_IP"
echo ""

# Kill any existing processes on the ports
echo "Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 2

# Start backend
echo "Starting backend API server..."
cd backend
source ../new-ft-env/bin/activate

# Export environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

python3 main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Test backend connection
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "200"; then
    echo "✔ Backend API is running"
else
    echo "⚠️  Backend may not be fully initialized yet"
fi

# Start frontend server
echo "Starting frontend web server..."
cd frontend

# Update the API_URL in index.html to use the actual server IP
sed -i "s|const API_URL = .*|const API_URL = 'http://${SERVER_IP}:8000';|" index.html

python3 -m http.server 8080 --bind 0.0.0.0 &
FRONTEND_PID=$!
cd ..

# Wait for frontend
sleep 2

echo ""
echo "======================================"
echo "✅ Demo is running!"
echo "======================================"
echo ""
echo "Access the demo from your Windows laptop:"
echo "👉 http://${SERVER_IP}:8080"
echo ""
echo "Backend API endpoints:"
echo "  - Status: http://${SERVER_IP}:8000/"
echo "  - Load Models: http://${SERVER_IP}:8000/load_models"
echo "  - Process Text: http://${SERVER_IP}:8000/mask_pii"
echo ""
echo "Instructions:"
echo "1. Open the web interface in your browser"
echo "2. Click 'Load Models' button (this will take a few minutes)"
echo "3. Enter text with PII or use sample buttons"
echo "4. Click 'Process & Compare' to see both models in action"
echo ""
echo "⚠️  Note: Models use ~32GB each. Ensure sufficient GPU memory!"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "======================================"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    
    # Restore original index.html
    cd frontend
    sed -i "s|const API_URL = .*|const API_URL = 'http://${SERVER_IP}:8000';|" index.html
    cd ..
    
    echo "✔ Demo stopped"
    exit 0
}

# Set trap for cleanup on Ctrl+C
trap cleanup INT

# Keep script running
while true; do
    sleep 1
done
