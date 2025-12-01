from sentence_transformers import SentenceTransformer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
local_model_path = "./Embeddings"

model = SentenceTransformer(model_id)
model.save(local_model_path)

print("Model saved correctly to:", local_model_path)
