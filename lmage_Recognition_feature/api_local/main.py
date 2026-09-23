import os
import json
import shutil
import tempfile

import torch
import numpy as np
import faiss
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor

# Configuration

MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_PATH = "places_index.faiss"
MAPPING_PATH = "places_mapping.json"
PLACES_INFO_PATH = "places_info.json" 

CONFIDENCE_THRESHOLD = 0.75
TOP_K = 5

# App + CORS 


app = FastAPI(title="Landmark Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything ONCE at startup 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(
        f"'{INDEX_PATH}' and '{MAPPING_PATH}' must be in this folder. "
        "Copy them here from your Colab session before starting the API."
    )

print("Loading FAISS index and mapping...")
faiss_index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)


places_info = {}
if os.path.exists(PLACES_INFO_PATH):
    with open(PLACES_INFO_PATH, "r", encoding="utf-8") as f:
        places_info = json.load(f)

print("API ready.")


# Core embedding function (identical logic to the notebook)

def get_image_embedding(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        pooled_output = vision_outputs.pooler_output
        features = model.visual_projection(pooled_output)
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32").flatten()


def search_place(image_path: str):
    query_embedding = get_image_embedding(image_path).reshape(1, -1)
    scores, indices = faiss_index.search(query_embedding, TOP_K)

    place_scores = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        place = mapping[str(idx)]["place"]
        place_scores[place] = max(place_scores.get(place, 0), float(score))

    if not place_scores:
        return None, 0.0, {}

    best_place = max(place_scores, key=place_scores.get)
    best_score = place_scores[best_place]
    return best_place, best_score, place_scores


# Endpoints-

@app.get("/health")
def health_check():
    return {"status": "ok", "places_indexed": len(set(v["place"] for v in mapping.values()))}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Save the upload to a temp file so PIL can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        best_place, best_score, all_scores = search_place(tmp_path)
    finally:
        os.remove(tmp_path)

    if best_place is None or best_score < CONFIDENCE_THRESHOLD:
        return {
            "recognized": False,
            "confidence": round(best_score, 4),
            "message": "Could not confidently identify this place.",
        }

    info = places_info.get(best_place, {})

    return {
        "recognized": True,
        "place": best_place,
        "confidence": round(best_score, 4),
        "description": info.get("description"),
        "category": info.get("category"),
        "candidates": [
            {"place": p, "score": round(s, 4)}
            for p, s in sorted(all_scores.items(), key=lambda x: -x[1])
        ],
    }
