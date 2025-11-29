from llama_cpp import Llama
import time

MODEL_PATH = "/home/curtburk/Desktop/gartner-pii-demo/models/pii_detector_Q4_K_M.gguf"

print("Loading model with optimized settings...")
model = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=1024,
    n_batch=2048,
    flash_attn=True,
    verbose=True
)

prompt = """<|im_start|>system
You are a PII masking assistant.<|im_end|>
<|im_start|>user
Mask PII: John Smith, SSN 123-45-6789, phone 555-123-4567<|im_end|>
<|im_start|>assistant
"""

print("\n" + "="*50)
print("Running inference benchmark...")
print("="*50)

start = time.time()
response = model(prompt, max_tokens=100, temperature=0.1)
elapsed = time.time() - start

print(f"\nOutput: {response['choices'][0]['text']}")
print(f"\nTime: {elapsed:.2f} seconds")
print(f"Tokens generated: {response['usage']['completion_tokens']}")
print(f"Tokens/second: {response['usage']['completion_tokens']/elapsed:.1f}")
