# VoiceX Integration Guide

## Overview

This guide explains how to connect your Python-based ML models to the VoiceX edge system and integrate real data flows.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Android    │────▶│  Rust Edge   │────▶│   SQLite     │   │
│  │   Device     │     │   (onnx)     │     │   (local)    │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                    │                      │          │
│         │                    │                      ▼          │
│         │                    │              ┌──────────────┐   │
│         │                    │              │  Sync Queue  │   │
│         │                    │              └──────────────┘   │
│         │                    │                      │          │
│         ▼                    ▼                      ▼          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Python API Server (Cloud)                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │  │
│  │  │  Your    │  │  Model   │  │  Geo     │  │  Data   │ │  │
│  │  │  AST     │  │  Retrain │  │  Cluster │  │  Export │ │  │
│  │  │  Model   │  │  Pipeline│  │  Engine  │  │  API    │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Web Dashboard (React/Map)                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (Python Developer)

### 1. Install Python API Server

```bash
cd voicex-python
pip install -r requirements.txt
```

### 2. Export Your Model to ONNX

```python
import torch
import torch.onnx

# Your AST model
model = YourASTModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Dummy input matching your audio features
dummy_input = torch.randn(1, 1, 224, 224)  # batch, channels, height, width

# Export
torch.onnx.export(
    model,
    dummy_input,
    "tb_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
```

### 3. Start the API Server

```bash
python api_server.py --model tb_classifier.onnx --port 8000
```

### 4. Test with Sample Audio

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@sample_cough.wav" \
  -F "lat=19.0760" \
  -F "lon=72.8777"
```

---

## Data Formats

### Audio Input

| Field | Type | Description |
|-------|------|-------------|
| `audio_file` | WAV/MP3/OGG | Raw audio file |
| `sample_rate` | int | Target sample rate (default: 16000) |
| `lat` | float | GPS latitude (optional) |
| `lon` | float | GPS longitude (optional) |
| `device_id` | string | Unique device identifier |
| `timestamp` | ISO8601 | Recording timestamp |

### Analysis Result (JSON)

```json
{
  "record_id": "rec_1708368000000000000",
  "timestamp": "2024-02-19T12:00:00Z",
  "location": {
    "latitude": 19.0760,
    "longitude": 72.8777,
    "accuracy_meters": 10.5
  },
  "tb_risk": {
    "risk_score": 0.78,
    "confidence": 0.92,
    "urgency_tier": "High",
    "cough_quality": "Good",
    "key_features": ["High frequency energy", "Prolonged exhalation"]
  },
  "voice_pathology": {
    "pathology_detected": true,
    "pathology_types": ["Dysphonia", "Roughness"],
    "confidence": 0.85,
    "jitter_percent": 1.24,
    "shimmer_percent": 0.15,
    "hnr_db": 8.5,
    "cpp_db": 12.3,
    "f0_mean": 125.0,
    "f0_std": 15.2
  },
  "sound_events": [
    {
      "event_type": "Cough",
      "confidence": 0.95,
      "start_time_ms": 500,
      "end_time_ms": 1200,
      "frequency_range": [100, 8000]
    }
  ],
  "audio_metadata": {
    "duration_secs": 3.5,
    "sample_rate": 16000,
    "channels": 1
  }
}
```

---

## API Endpoints

### POST /predict
Analyze audio file for TB risk and voice pathology.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@cough.wav" \
  -F "lat=19.0760" \
  -F "lon=72.8777" \
  -F "device_id=dev_001"
```

**Response:**
```json
{
  "status": "success",
  "data": { /* AnalysisResult */ },
  "processing_time_ms": 245
}
```

### POST /batch_predict
Batch process multiple audio files.

**Request:**
```bash
curl -X POST http://localhost:8000/batch_predict \
  -F "audio_files=@batch1.wav" \
  -F "audio_files=@batch2.wav"
```

### GET /clusters
Get geographic clusters of detections.

**Request:**
```bash
curl "http://localhost:8000/clusters?days=30&eps_km=5.0&min_points=3"
```

**Response:**
```json
{
  "clusters": [
    {
      "cluster_id": "cluster_a1b2c3d4",
      "center": {"latitude": 19.076, "longitude": 72.877},
      "radius_km": 2.5,
      "detection_count": 156,
      "high_risk_count": 23,
      "first_detection": "2024-01-15T08:00:00Z",
      "last_detection": "2024-02-19T14:30:00Z"
    }
  ]
}
```

### GET /heatmap
Get heatmap data for visualization.

**Request:**
```bash
curl "http://localhost:8000/heatmap?bounds=19.0,72.8,19.1,72.9&resolution_km=1.0"
```

### POST /sync
Upload anonymized data to cloud.

**Request:**
```bash
curl -X POST http://localhost:8000/sync \
  -H "Content-Type: application/json" \
  -d @sync_payload.json
```

---

## Connecting Your AST Model

### Option 1: ONNX Export (Recommended)

```python
# train_model.py
import torch
import torch.nn as nn

class TBASTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Your AST architecture
        self.ast = ASTModel(
            label_dim=num_classes,
            fshape=16,  # patch size
            tshape=16,
            fstride=10,
            tstride=10,
            input_fdim=128,  # mel bins
            input_tdim=224,  # time frames
            model_size='base'
        )
    
    def forward(self, x):
        # x: (batch, 1, 128, 224) - mel spectrogram
        return self.ast(x)

# Export to ONNX
model = TBASTModel()
model.load_state_dict(torch.load('tb_ast_model.pth'))
model.eval()

dummy_input = torch.randn(1, 1, 128, 224)
torch.onnx.export(
    model,
    dummy_input,
    "tb_classifier.onnx",
    input_names=["mel_spectrogram"],
    output_names=[["risk_score", "confidence"]],
    opset_version=11
)
```

### Option 2: Python API Bridge

If you want to keep your model in Python:

```python
# api_server.py
from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
from feature_extraction import extract_mel_spectrogram

app = FastAPI()
model = torch.jit.load('tb_model_scripted.pt')  # TorchScript

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    # Read audio
    audio_data = await audio.read()
    
    # Extract features (in Python)
    mel_spec = extract_mel_spectrogram(audio_data)
    
    # Run model
    with torch.no_grad():
        input_tensor = torch.from_numpy(mel_spec).unsqueeze(0)
        output = model(input_tensor)
    
    risk_score = float(torch.sigmoid(output[0]))
    confidence = float(output[1])
    
    return {
        "tb_risk": {
            "risk_score": risk_score,
            "confidence": confidence,
            "urgency_tier": get_urgency_tier(risk_score)
        }
    }
```

---

## Database Schema

### SQLite Tables

```sql
-- Main analysis results
CREATE TABLE analysis_results (
    record_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    accuracy_meters REAL,
    tb_risk_score REAL NOT NULL,
    tb_confidence REAL NOT NULL,
    urgency_tier TEXT NOT NULL,
    cough_quality TEXT,
    pathology_detected INTEGER NOT NULL,
    jitter_percent REAL,
    shimmer_percent REAL,
    hnr_db REAL,
    cpp_db REAL,
    f0_mean REAL,
    f0_std REAL,
    audio_duration_secs REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    synced INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Sound events detected
CREATE TABLE sound_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    start_time_ms INTEGER,
    end_time_ms INTEGER,
    FOREIGN KEY (record_id) REFERENCES analysis_results(record_id)
);

-- Key features for explainability
CREATE TABLE key_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES analysis_results(record_id)
);
```

---

## Feature Extraction Pipeline

### Python Implementation

```python
# feature_extraction.py
import librosa
import numpy as np

def extract_features(audio_path, sr=16000):
    """Extract all features needed for VoiceX models."""
    
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # 1. Mel Spectrogram (for AST model)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, 
        n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. MFCCs (for voice pathology)
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=2048, hop_length=512
    )
    
    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    
    # 4. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=512
    )[0]
    
    # 5. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=512
    )[0]
    
    # 6. RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    
    # 7. Chromagram (for pitch analysis)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=512
    )
    
    return {
        'mel_spectrogram': mel_spec_db,
        'mfccs': mfccs,
        'zero_crossing_rate': zcr,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'rms_energy': rms,
        'chromagram': chroma,
        'duration': len(y) / sr
    }

def compute_jitter(f0_values):
    """Compute jitter (frequency perturbation)."""
    diffs = np.diff(f0_values)
    return np.mean(np.abs(diffs)) / np.mean(f0_values) * 100

def compute_shimmer(amplitude_values):
    """Compute shimmer (amplitude perturbation)."""
    diffs = np.diff(amplitude_values)
    return np.mean(np.abs(diffs)) / np.mean(amplitude_values) * 100

def compute_hnr(y, sr):
    """Compute Harmonics-to-Noise Ratio."""
    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    
    harmonic_rms = np.sqrt(np.mean(harmonic**2))
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-10
    
    return 20 * np.log10(harmonic_rms / noise_rms)
```

---

## Geographic Clustering

### Python Implementation

```python
# geo_clustering.py
from sklearn.cluster import DBSCAN
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Detection:
    lat: float
    lon: float
    risk_score: float
    timestamp: str

@dataclass
class Cluster:
    center_lat: float
    center_lon: float
    radius_km: float
    detection_count: int
    high_risk_count: int

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def cluster_detections(detections: List[Detection], 
                       eps_km: float = 5.0, 
                       min_samples: int = 3) -> List[Cluster]:
    """Cluster detections using DBSCAN."""
    
    if len(detections) < min_samples:
        return []
    
    # Convert to numpy array
    coords = np.array([[d.lat, d.lon] for d in detections])
    
    # Custom metric for haversine distance
    def haversine_metric(a, b):
        return haversine_distance(a[0], a[1], b[0], b[1])
    
    # Run DBSCAN
    clustering = DBSCAN(
        eps=eps_km, 
        min_samples=min_samples,
        metric=haversine_metric
    ).fit(coords)
    
    # Build clusters
    clusters = []
    unique_labels = set(clustering.labels_)
    
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
        
        mask = clustering.labels_ == label
        cluster_points = coords[mask]
        cluster_detections = [d for d, m in zip(detections, mask) if m]
        
        center_lat = np.mean(cluster_points[:, 0])
        center_lon = np.mean(cluster_points[:, 1])
        
        # Calculate radius (max distance from center)
        distances = [
            haversine_distance(center_lat, center_lon, p[0], p[1])
            for p in cluster_points
        ]
        radius = max(distances) if distances else 0
        
        # Count high-risk detections
        high_risk = sum(1 for d in cluster_detections if d.risk_score > 0.7)
        
        clusters.append(Cluster(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius,
            detection_count=len(cluster_detections),
            high_risk_count=high_risk
        ))
    
    return clusters
```

---

## Cloud Sync Specification

### Sync Payload Format

```json
{
  "device_id": "dev_001",
  "export_timestamp": "2024-02-19T12:00:00Z",
  "records": [
    {
      "record_id": "rec_1708368000000000000",
      "timestamp": "2024-02-19T10:30:00Z",
      "location": {
        "latitude": 19.0760,
        "longitude": 72.8777
      },
      "tb_risk_score": 0.78,
      "urgency_tier": "High",
      "pathology_detected": true,
      "sound_event_types": ["Cough", "Wheeze"],
      "audio_features_hash": "sha256:abc123..."
    }
  ]
}
```

### Sync API

```python
# sync_client.py
import requests
import sqlite3
import json
from datetime import datetime

class VoiceXSyncClient:
    def __init__(self, db_path: str, api_endpoint: str, api_key: str):
        self.db_path = db_path
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    def export_unsynced(self) -> dict:
        """Export unsynced records from local database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT record_id, timestamp, latitude, longitude,
                   tb_risk_score, urgency_tier, pathology_detected
            FROM analysis_results
            WHERE synced = 0
            LIMIT 1000
        """)
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "record_id": row[0],
                "timestamp": row[1],
                "location": {
                    "latitude": row[2],
                    "longitude": row[3]
                },
                "tb_risk_score": row[4],
                "urgency_tier": row[5],
                "pathology_detected": bool(row[6]),
                "sound_event_types": []  # Would join with sound_events table
            })
        
        conn.close()
        
        return {
            "device_id": self.get_device_id(),
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "records": records
        }
    
    def sync(self) -> dict:
        """Sync unsynced records to cloud."""
        payload = self.export_unsynced()
        
        response = requests.post(
            f"{self.api_endpoint}/sync",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload
        )
        
        if response.status_code == 200:
            # Mark records as synced
            record_ids = [r["record_id"] for r in payload["records"]]
            self.mark_synced(record_ids)
        
        return response.json()
    
    def mark_synced(self, record_ids: list):
        """Mark records as synced in local database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for record_id in record_ids:
            cursor.execute(
                "UPDATE analysis_results SET synced = 1 WHERE record_id = ?",
                (record_id,)
            )
        
        conn.commit()
        conn.close()
```

---

## Testing Your Integration

### Unit Tests

```python
# test_integration.py
import unittest
import numpy as np
from feature_extraction import extract_features
from geo_clustering import cluster_detections, Detection

class TestFeatureExtraction(unittest.TestCase):
    def test_mel_spectrogram_shape(self):
        features = extract_features('test_audio.wav')
        self.assertEqual(features['mel_spectrogram'].shape[0], 128)
    
    def test_hnr_computation(self):
        y = np.random.randn(16000)  # 1 second of noise
        hnr = compute_hnr(y, 16000)
        self.assertLess(hnr, 0)  # Noise should have negative HNR

class TestGeoClustering(unittest.TestCase):
    def test_dbscan_clustering(self):
        detections = [
            Detection(19.076, 72.877, 0.8, "2024-01-01"),
            Detection(19.077, 72.878, 0.7, "2024-01-02"),
            Detection(19.078, 72.879, 0.9, "2024-01-03"),
            Detection(28.614, 77.209, 0.5, "2024-01-01"),  # Different city
        ]
        
        clusters = cluster_detections(detections, eps_km=5.0, min_samples=2)
        self.assertEqual(len(clusters), 1)  # Should find one cluster
        self.assertEqual(clusters[0].detection_count, 3)

if __name__ == '__main__':
    unittest.main()
```

---

## Next Steps

1. **Export your AST model to ONNX** using the provided script
2. **Set up the Python API server** with your model
3. **Test with sample audio files** using the `/predict` endpoint
4. **Connect your Android app** to the API (or use the Rust edge client)
5. **Deploy the web dashboard** to visualize clusters

For questions or issues, check the `examples/` directory or open an issue on GitHub.
