from transformers import AutoModel, AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2" # Or your desired model
local_model_path = "./Embeddings"

# Download and save the model and tokenizer
model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
