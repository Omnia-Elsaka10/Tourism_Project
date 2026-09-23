import json

import spaces  
import torch
import faiss
import gradio as gr
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Configuration

MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_PATH = "places_index.faiss"
MAPPING_PATH = "places_mapping.json"

CONFIDENCE_THRESHOLD = 0.75
TOP_K = 5


GPU_DURATION_SECONDS = 20


# Load model on CPU at startup 

print("Loading CLIP model (on CPU, moved to GPU only during requests)...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

print("Loading FAISS index and mapping (CPU-only, unaffected by GPU quota)...")
faiss_index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

print("Ready.")


# GPU-backed embedding function

@spaces.GPU(duration=GPU_DURATION_SECONDS)
def get_image_embedding(image: Image.Image):
    """
    Runs on an actual GPU for the duration of this call only, thanks to the
    @spaces.GPU decorator. Outside this function, there is no GPU attached.
    """
    model.to("cuda")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        pooled_output = vision_outputs.pooler_output
        features = model.visual_projection(pooled_output)
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32").flatten()


def recognize(image: Image.Image):
    """
    This function is the API. Whatever it returns (a JSON-serializable dict)
    is exactly what the backend receives when it calls this endpoint.
    The FAISS search itself stays on CPU - only get_image_embedding touches the GPU.
    """
    if image is None:
        return {"recognized": False, "confidence": 0.0, "message": "No image provided."}

    embedding = get_image_embedding(image).reshape(1, -1)
    scores, indices = faiss_index.search(embedding, TOP_K)

    place_scores = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        place = mapping[str(idx)]["place"]
        place_scores[place] = max(place_scores.get(place, 0), float(score))

    if not place_scores:
        return {"recognized": False, "confidence": 0.0, "message": "No match found."}

    best_place = max(place_scores, key=place_scores.get)
    best_score = place_scores[best_place]

    if best_score < CONFIDENCE_THRESHOLD:
        return {
            "recognized": False,
            "confidence": round(best_score, 4),
            "message": "Could not confidently identify this place.",
        }

    return {
        "recognized": True,
        "place_id": best_place,
        "confidence": round(best_score, 4),
        "candidates": [
            {"place_id": p, "score": round(s, 4)}
            for p, s in sorted(place_scores.items(), key=lambda x: -x[1])
        ],
    }


# UI 

with gr.Blocks(title="Egypt Landmark Recognition") as demo:
    gr.Markdown(
        "# Egypt Landmark Recognition\n"
        "Upload a photo to test the recognition model. "
        "This same function is available as an API endpoint (`recognize`) "
        "for the backend to call directly."
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload a photo")
        output_json = gr.JSON(label="Result")

    recognize_btn = gr.Button("Recognize")
    recognize_btn.click(
        fn=recognize,
        inputs=image_input,
        outputs=output_json,
        api_name="recognize",  # this name is what the backend will call
    )

demo.launch()
