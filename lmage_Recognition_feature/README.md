# Landmark Recognition API

Wraps the CLIP + FAISS pipeline from the Colab notebook into a real HTTP API
that a mobile app (or anything else) can call.

## Setup

1. **Copy your index files here.** From your Colab session, download:
   - `places_index.faiss`
   - `places_mapping.json`

   and place them in this same folder (next to `main.py`).

2. **(Optional) Add place descriptions.** Copy `places_info.example.json` to
   `places_info.json` and fill in real descriptions for each place. The key
   for each entry must match the place names from your dataset (visible in
   `places_mapping.json`). If you skip this, the API still works — it just
   won't return a `description` field.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Testing it

Check it's alive:
```bash
curl http://localhost:8000/health
```

Send a photo:
```bash
curl -X POST "http://localhost:8000/recognize" -F "file=@test_photo.jpg"
```

Example response:
```json
{
  "recognized": true,
  "place": "luxor_temple",
  "confidence": 0.8421,
  "description": "One of the largest ancient sites in Egypt...",
  "category": "temple",
  "candidates": [
    {"place": "luxor_temple", "score": 0.8421},
    {"place": "karnak_temple", "score": 0.7103}
  ]
}
```

You can also open `http://localhost:8000/docs` in a browser — FastAPI
auto-generates an interactive test page where you can upload a photo directly
without curl.

## Next steps

- **Ratings & nearby activities**: this API currently only returns what you
  put in `places_info.json`. To get live ratings and nearby activities, call
  the Google Places API from inside the `/recognize` endpoint once you have
  the matched place name, and add the results to the response before
  returning it.
- **Deployment**: once this works locally, deploy it on a free tier like
  Render or Railway so your mobile app teammates can call a real URL instead
  of `localhost`.
- **Mobile integration**: the mobile app just needs to send a `multipart/form-data`
  POST request with the photo to `/recognize` — any HTTP client library works.
