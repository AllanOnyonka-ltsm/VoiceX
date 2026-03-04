# VoiceX Quick Start Guide

Get your VoiceX system running in 5 minutes.

## Prerequisites

- Python 3.8+
- pip
- (Optional) Your trained AST model

## Step 1: Install

```bash
# Clone/navigate to the Python API directory
cd voicex-python

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Start the API Server

### Option A: Without Model (Demo Mode)

```bash
python api_server.py --port 8000
```

The server will run with fallback predictions for testing.

### Option B: With Your ONNX Model

```bash
# First, export your PyTorch model to ONNX
python export_model.py --checkpoint your_model.pth --output tb_classifier.onnx

# Then start the server
python api_server.py --model tb_classifier.onnx --port 8000
```

## Step 3: Test the API

```bash
# Generate test audio and run prediction
python test_api.py --generate-audio --api http://localhost:8000
```

Expected output:
```
============================================================
Testing /predict endpoint
============================================================
Status: 200
Response time: 245.32ms

Record ID: rec_a1b2c3d4
Timestamp: 2024-02-19T12:00:00Z

--- TB Risk Assessment ---
Risk Score: 0.780
Confidence: 0.920
Urgency Tier: High
Cough Quality: Good
Key Features: High frequency energy, Prolonged exhalation

--- Voice Pathology ---
Pathology Detected: true
Types: Dysphonia
Jitter: 1.24%
Shimmer: 0.15%
HNR: 8.50 dB
```

## Step 4: View Geographic Dashboard

1. Open the web dashboard: `https://your-deployment-url`
2. Scroll to the "Geographic Surveillance" section
3. Click "Open Dashboard" to see the interactive map

## API Usage Examples

### Python Client

```python
import requests

# Analyze audio
with open('cough.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'audio': f},
        data={'lat': 19.0760, 'lon': 72.8777}
    )

result = response.json()
print(f"Risk: {result['tb_risk']['risk_score']:.2f}")
print(f"Urgency: {result['tb_risk']['urgency_tier']}")
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "audio=@cough.wav" \
  -F "lat=19.0760" \
  -F "lon=72.8777"

# Get clusters
curl "http://localhost:8000/clusters?days=30&eps_km=5.0"

# Get heatmap
curl "http://localhost:8000/heatmap?days=30"
```

## Data Format

### Input

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | WAV/MP3/OGG | Yes | Audio file |
| `lat` | float | No | GPS latitude |
| `lon` | float | No | GPS longitude |
| `device_id` | string | No | Device identifier |

### Output

```json
{
  "record_id": "rec_abc123",
  "timestamp": "2024-02-19T12:00:00Z",
  "location": {"latitude": 19.076, "longitude": 72.877},
  "tb_risk": {
    "risk_score": 0.78,
    "confidence": 0.92,
    "urgency_tier": "High",
    "cough_quality": "Good",
    "key_features": ["High frequency energy"]
  },
  "voice_pathology": {
    "pathology_detected": true,
    "pathology_types": ["Dysphonia"],
    "jitter_percent": 1.24,
    "shimmer_percent": 0.15,
    "hnr_db": 8.5,
    "cpp_db": 12.3
  },
  "sound_events": [
    {"event_type": "Cough", "confidence": 0.95}
  ],
  "audio_metadata": {
    "duration_secs": 3.5,
    "sample_rate": 16000,
    "channels": 1
  }
}
```

## Connecting Real Data

### 1. Android App → API

Your Android app sends audio to the API:

```kotlin
// Kotlin example
val requestBody = MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("audio", "cough.wav", audioFile.asRequestBody())
    .addFormDataPart("lat", location.latitude.toString())
    .addFormDataPart("lon", location.longitude.toString())
    .build()

val request = Request.Builder()
    .url("http://your-api-server:8000/predict")
    .post(requestBody)
    .build()

client.newCall(request).execute()
```

### 2. API → Database

Results are stored in SQLite (or your preferred database):

```python
# The API server automatically stores results
# Access via /stats, /clusters, /heatmap endpoints
```

### 3. Dashboard ← API

The web dashboard fetches data from the API:

```javascript
// Fetch clusters
const clusters = await fetch('/clusters?days=30').then(r => r.json());

// Display on map
clusters.forEach(cluster => {
    L.circleMarker([cluster.center.latitude, cluster.center.longitude])
        .addTo(map);
});
```

## Training Your Own Model

```bash
# 1. Prepare dataset
# Place audio files in:
#   data/coughs/positive/ (TB positive)
#   data/coughs/negative/ (TB negative)

# 2. Train model
python example_train_model.py

# 3. Export to ONNX
python export_model.py --checkpoint best_model.pth --output tb_classifier.onnx

# 4. Start API with your model
python api_server.py --model tb_classifier.onnx
```

## Troubleshooting

### "No module named 'librosa'"

```bash
pip install -r requirements.txt
```

### "Model loading failed"

```bash
# Verify ONNX model
python -c "import onnxruntime as ort; ort.InferenceSession('model.onnx')"
```

### "Port already in use"

```bash
# Use different port
python api_server.py --port 8001
```

## Next Steps

1. **Train your AST model** using your dataset
2. **Deploy the API** to a cloud server
3. **Connect your Android app** to the API
4. **Customize the dashboard** for your needs

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed documentation.
