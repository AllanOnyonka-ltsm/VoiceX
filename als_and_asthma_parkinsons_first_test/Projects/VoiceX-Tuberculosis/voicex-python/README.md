# VoiceX Python Integration

Python API server and tools for integrating your ML models with VoiceX edge system.

## Quick Start

### 1. Install Dependencies

```bash
cd voicex-python
pip install -r requirements.txt
```

### 2. Export Your Model

```bash
# Export your PyTorch model to ONNX
python export_model.py --checkpoint your_model.pth --output tb_classifier.onnx
```

### 3. Start the API Server

```bash
# With your model
python api_server.py --model tb_classifier.onnx --port 8000

# Without model (uses fallback predictions)
python api_server.py --port 8000
```

### 4. Test the API

```bash
# Generate test audio and run prediction
python test_api.py --generate-audio --api http://localhost:8000

# Test with your own audio
python test_api.py --audio your_cough.wav --lat 19.0760 --lon 72.8777

# Run all tests
python test_api.py --test-all
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Analyze audio for TB/voice pathology |
| `/batch_predict` | POST | Batch process multiple files |
| `/clusters` | GET | Get geographic clusters |
| `/heatmap` | GET | Get heatmap data |
| `/stats` | GET | System statistics |
| `/health` | GET | Health check |

## Example Usage

### Single Prediction

```python
import requests

with open('cough.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'audio': f},
        data={'lat': 19.0760, 'lon': 72.8777}
    )

result = response.json()
print(f"Risk Score: {result['tb_risk']['risk_score']}")
print(f"Urgency: {result['tb_risk']['urgency_tier']}")
```

### Get Clusters

```python
import requests

response = requests.get(
    'http://localhost:8000/clusters',
    params={'days': 30, 'eps_km': 5.0}
)

clusters = response.json()['clusters']
for cluster in clusters:
    print(f"Cluster at {cluster['center']}: {cluster['detection_count']} detections")
```

## Integrating Your Model

### Option 1: ONNX Export (Recommended for Edge)

```python
import torch

# Load your model
model = YourASTModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Export
dummy_input = torch.randn(1, 1, 128, 224)  # mel spectrogram shape
torch.onnx.export(
    model,
    dummy_input,
    "tb_classifier.onnx",
    input_names=["mel_spectrogram"],
    output_names=["risk_score", "confidence"],
    opset_version=11
)
```

### Option 2: TorchScript (For Python API)

```python
import torch

# Trace and save
traced = torch.jit.trace(model, dummy_input)
traced.save("tb_model.pt")
```

### Option 3: Direct Python Integration

Modify `api_server.py` to load your model directly:

```python
# In api_server.py, replace TBClassifier with your model

class YourModelWrapper:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
    
    def predict(self, mel_spectrogram):
        with torch.no_grad():
            input_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0)
            output = self.model(input_tensor)
            risk_score = float(torch.sigmoid(output[0]))
            return {
                'risk_score': risk_score,
                'confidence': 0.9,
                'urgency_tier': get_urgency_tier(risk_score),
                'cough_quality': 'Good',
                'key_features': []
            }

# Use your wrapper
tb_classifier = YourModelWrapper('your_model.pth')
```

## Data Flow

```
Audio File → Feature Extraction → Your Model → API Response
                ↓
         Mel Spectrogram (128x224)
                ↓
         ONNX Runtime Inference
                ↓
         Risk Score + Voice Metrics
                ↓
         JSON Response
```

## Feature Extraction Pipeline

The API extracts these features from audio:

1. **Mel Spectrogram** (128 bins, 224 frames) → Input to AST model
2. **MFCCs** (13 coefficients) → Voice pathology analysis
3. **Zero Crossing Rate** → Sound event detection
4. **Spectral Centroid/Rolloff** → Frequency characteristics
5. **RMS Energy** → Amplitude envelope
6. **Chromagram** → Pitch class profile
7. **F0 (Fundamental Frequency)** → Voice metrics
8. **HNR, Jitter, Shimmer, CPP** → Clinical voice parameters

## Configuration

Environment variables:

```bash
export VOICEX_MODEL_PATH="tb_classifier.onnx"
export VOICEX_DB_PATH="data/voicex.db"
export VOICEX_PORT=8000
```

## Testing

```bash
# Run all API tests
python test_api.py --test-all --generate-audio

# Test specific endpoints
python test_api.py --health
python test_api.py --stats
python test_api.py --clusters
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api_server:app
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api_server.py", "--model", "tb_classifier.onnx"]
```

## Connecting to Rust Edge

The Rust edge client can sync with this API:

```rust
// In your Rust code
let client = reqwest::Client::new();
let response = client
    .post("http://api-server:8000/sync")
    .json(&sync_payload)
    .send()
    .await?;
```

## Troubleshooting

### Model Loading Issues

```bash
# Verify ONNX model
python -c "import onnxruntime as ort; print(ort.InferenceSession('model.onnx'))"
```

### Audio Processing Issues

```bash
# Test feature extraction
python -c "
from api_server import FeatureExtractor
import numpy as np
fe = FeatureExtractor()
features = fe.extract(open('test.wav', 'rb').read())
print(features.keys())
"
```

## License

MIT License - See LICENSE file
