# Uninstall current CPU-only version
pip uninstall llama-cpp-python -y

# Rebuild with CUDA for Grace Blackwell (compute capability 100)
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --no-cache-dir --break-system-packages
