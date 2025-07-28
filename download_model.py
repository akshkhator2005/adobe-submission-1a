from pathlib import Path
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L12-v2"
save_directory = Path("./english_minilm_model/")

save_directory.mkdir(exist_ok=True)

print(f"Downloading model: '{model_name}'...")
model = SentenceTransformer(model_name)
model.save(str(save_directory))
print(f"✅ Model downloaded successfully and saved to '{save_directory}'")