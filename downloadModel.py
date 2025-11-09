from huggingface_hub import hf_hub_download

# TinyLlama 1.1B Chat model (smaller & faster than Mistral)
model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Download the model file into your local 'models/' directory
model_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    cache_dir="models/"
)

print(f"✅ Model downloaded successfully!")
print(f"📍 Local path: {model_path}")