# AI Voice Detection API

Backend API for detecting AI-generated voices using `facebook/wav2vec2-large-xlsr-53`.

## Setup & Run

### 1. Local (Mac)

**Prerequisites:**
- Python 3.10+
- `ffmpeg` (install via `brew install ffmpeg`)
- `libsndfile` (install via `brew install libsndfile`)

**Steps:**
```bash
# 1. Create virtual env
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --reload
```

### 2. Docker

```bash
# Build
docker build -t voice-detector .

# Run (Port 8000)
docker run -p 8000:8000 voice-detector
```

## Usage API

**Headers:**
- `X-API-Key`: `demo_key_123` (or set via `API_KEY` env var)

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Detect Voice (via URL)
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo_key_123" \
  -d '{
    "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
  }'
```

### Detect Voice (via Base64)
*(Truncated for validity, replace `...` with real base64 mp3 data)*
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo_key_123" \
  -d '{
    "audio_base64": "SUQzBAAAAAAA..."
  }'
```

## Notes
- **Model Download**: On first run, the app will download ~1.2GB model weights. This may take a few minutes.
- **Detector**: Currently uses a heuristic based on embedding variance. This is a baseline MVP.
