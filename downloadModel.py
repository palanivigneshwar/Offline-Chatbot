from huggingface_hub import hf_hub_download

repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
filename = "Phi-3-mini-4k-instruct-q4.gguf"

model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir="models/"
)

print("Model downloaded:", model_path)
