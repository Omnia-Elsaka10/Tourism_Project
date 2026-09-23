# AI Recognition Service — Handoff to Backend

This is what the AI part delivers, and what the backend needs to do with it.

## What the AI service does (and only this)

Given a photo, it answers one question: **"which place is this?"** It does
NOT know about ratings, nearby activities, descriptions, or anything else in
the app's database. That's intentionally the backend's job — it keeps this
service simple, testable, and independent from your main app's database
schema or business logic.

## Where it's hosted

This service runs as a **Hugging Face Space** (Gradio SDK, free tier), not
a traditional REST API. It's reachable at:

```
https://<your-username>-<your-space-name>.hf.space
```

(the exact URL is shown on the Space's page once deployed)

## How the backend calls it

Because this runs on Gradio, the cleanest way to call it is with Hugging
Face's official `gradio_client` library rather than raw HTTP requests —
it handles the request format for you.

**Python:**
```python
from gradio_client import Client, handle_file

client = Client("your-username/your-space-name")
result = client.predict(
    handle_file("photo.jpg"),
    api_name="/recognize"
)
print(result)
```

**JavaScript / Node.js** (if the backend isn't Python):
```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("your-username/your-space-name");
const result = await client.predict("/recognize", {
  image: photoFileBlob,
});
console.log(result.data);
```

Install with `pip install gradio_client` (Python) or `npm install @gradio/client` (JS).

## The response format

**Success** (confidence above threshold):
```json
{
  "recognized": true,
  "place_id": "luxor_temple",
  "confidence": 0.8421,
  "candidates": [
    {"place_id": "luxor_temple", "score": 0.8421},
    {"place_id": "karnak_temple", "score": 0.7103}
  ]
}
```

**Uncertain / no match**:
```json
{
  "recognized": false,
  "confidence": 0.42,
  "message": "Could not confidently identify this place."
}
```

## What `place_id` actually is

It's the raw folder name from the training dataset (e.g. `luxor_temple`,
`abu_simbel`). It is **not** a display name and should never be shown to the
user directly. The backend should maintain its own table mapping each
`place_id` to:
- Display name and description (in whatever language the app supports)
- A Google Places `place_id` or search query, for fetching live ratings and
  nearby activities via the Google Places API
- Any other app-specific data (photos, opening hours, ticket prices, etc.)

This keeps content management (editing descriptions, adding new places to
the *display* layer) entirely in the backend/database, without ever needing
to touch or redeploy the AI service.

## What the backend is responsible for

1. Receiving the photo from the mobile app
2. Forwarding it to the AI Space using `gradio_client` as shown above
3. Taking the returned `place_id` and looking up the full place record in
   the app's own database
4. Calling the Google Places API (or your own cached data) for live rating
   and nearby activities
5. Combining everything into the final response the mobile app receives
6. Handling auth, rate limiting, logging, and error cases (e.g. AI service
   down, unrecognized place, low confidence)

## What happens on "not recognized" or low confidence

The backend should decide the UX for this — e.g. show "we couldn't identify
this place, try a clearer photo" or offer a manual search fallback. The AI
service just reports its confidence honestly; it does not guess.

## Important: cold starts on the free tier

The free Hugging Face Space sleeps after ~48 hours of no traffic. The first
request after it wakes up will take longer (the model has to reload into
memory). The backend should set a reasonably generous timeout (e.g. 30-60
seconds) on its call to the AI service to avoid failing on a cold start.
