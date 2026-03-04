#!/usr/bin/env python3
"""
VoiceX Python API Server

This server provides HTTP endpoints for:
- Audio analysis (TB risk, voice pathology)
- Geographic clustering
- Data sync
- Model management

Usage:
    python api_server.py --model tb_classifier.onnx --port 8000
"""

import argparse
import hashlib
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
import uvicorn

# =============================================================================
# Data Models
# =============================================================================

class GeoLocation(BaseModel):
    latitude: float
    longitude: float
    accuracy_meters: Optional[float] = None

class TBRiskResult(BaseModel):
    risk_score: float
    confidence: float
    urgency_tier: str
    cough_quality: str
    key_features: List[str]

class VoicePathologyResult(BaseModel):
    pathology_detected: bool
    pathology_types: List[str]
    confidence: float
    jitter_percent: float
    shimmer_percent: float
    hnr_db: float
    cpp_db: float
    f0_mean: float
    f0_std: float

class SoundEvent(BaseModel):
    event_type: str
    confidence: float
    start_time_ms: int
    end_time_ms: int
    frequency_range: List[float]

class AudioMetadata(BaseModel):
    duration_secs: float
    sample_rate: int
    channels: int

class AnalysisResult(BaseModel):
    record_id: str
    timestamp: str
    location: Optional[GeoLocation]
    tb_risk: TBRiskResult
    voice_pathology: VoicePathologyResult
    sound_events: List[SoundEvent]
    audio_metadata: AudioMetadata

class Cluster(BaseModel):
    cluster_id: str
    center: GeoLocation
    radius_km: float
    detection_count: int
    high_risk_count: int
    first_detection: str
    last_detection: str

class HeatmapPoint(BaseModel):
    latitude: float
    longitude: float
    intensity: float
    count: int
    risk_score: float

class SyncPayload(BaseModel):
    device_id: str
    export_timestamp: str
    records: List[dict]

# =============================================================================
# Feature Extraction
# =============================================================================

class FeatureExtractor:
    """Extract audio features for ML models."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 13
    
    def extract(self, audio_bytes: bytes) -> dict:
        """Extract all features from audio bytes."""
        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True)
        
        # 1. Mel Spectrogram (for AST model)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=self.hop_length
        )[0]
        
        # 5. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # 6. Chromagram
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # 7. Fundamental frequency (for voice metrics)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        
        # 8. HNR estimation
        hnr = self._compute_hnr(y)
        
        # 9. Jitter and Shimmer
        jitter = self._compute_jitter(f0_clean) if len(f0_clean) > 1 else 0.0
        shimmer = self._compute_shimmer(rms) if len(rms) > 1 else 0.0
        
        # 10. CPP (Cepstral Peak Prominence)
        cpp = self._compute_cpp(y, sr)
        
        return {
            'mel_spectrogram': mel_spec_db,
            'mfccs': mfccs,
            'zero_crossing_rate': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'rms_energy': rms,
            'chromagram': chroma,
            'f0_values': f0_clean,
            'jitter': jitter,
            'shimmer': shimmer,
            'hnr': hnr,
            'cpp': cpp,
            'duration': len(y) / sr,
            'sample_rate': sr,
            'channels': 1
        }
    
    def _compute_hnr(self, y: np.ndarray) -> float:
        """Estimate Harmonics-to-Noise Ratio."""
        try:
            harmonic = librosa.effects.harmonic(y)
            noise = y - harmonic
            
            harmonic_rms = np.sqrt(np.mean(harmonic**2))
            noise_rms = np.sqrt(np.mean(noise**2)) + 1e-10
            
            return 20 * np.log10(harmonic_rms / noise_rms)
        except:
            return 0.0
    
    def _compute_jitter(self, f0_values: np.ndarray) -> float:
        """Compute jitter (frequency perturbation)."""
        if len(f0_values) < 2:
            return 0.0
        diffs = np.diff(f0_values)
        return np.mean(np.abs(diffs)) / np.mean(f0_values) * 100
    
    def _compute_shimmer(self, amplitude: np.ndarray) -> float:
        """Compute shimmer (amplitude perturbation)."""
        if len(amplitude) < 2:
            return 0.0
        diffs = np.diff(amplitude)
        return np.mean(np.abs(diffs)) / np.mean(amplitude) * 100
    
    def _compute_cpp(self, y: np.ndarray, sr: int) -> float:
        """Compute Cepstral Peak Prominence."""
        try:
            # Compute cepstrum
            spectrum = np.abs(np.fft.rfft(y))
            log_spectrum = np.log(spectrum + 1e-10)
            cepstrum = np.fft.irfft(log_spectrum)
            
            # Find peak in quefrency range (2-20 ms)
            start_idx = int(sr * 0.002)
            end_idx = int(sr * 0.020)
            
            if end_idx > len(cepstrum):
                end_idx = len(cepstrum)
            
            if start_idx >= end_idx:
                return 0.0
            
            peak_idx = start_idx + np.argmax(cepstrum[start_idx:end_idx])
            peak_value = cepstrum[peak_idx]
            
            # Compute prominence
            local_mean = np.mean(cepstrum[start_idx:end_idx])
            local_std = np.std(cepstrum[start_idx:end_idx])
            
            if local_std > 0:
                return (peak_value - local_mean) / local_std
            return 0.0
        except:
            return 0.0

# =============================================================================
# Model Inference
# =============================================================================

class TBClassifier:
    """TB Risk Classifier using ONNX model."""
    
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.target_size = (128, 224)  # mel bins x time frames
    
    def predict(self, mel_spectrogram: np.ndarray) -> dict:
        """Run inference on mel spectrogram."""
        # Resize to target size
        resized = self._resize_spectrogram(mel_spectrogram)
        
        # Normalize
        normalized = (resized - resized.mean()) / (resized.std() + 1e-8)
        
        # Add batch and channel dimensions: (1, 1, 128, 224)
        input_tensor = normalized[np.newaxis, np.newaxis, ...].astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Parse outputs (adjust based on your model)
        risk_score = float(outputs[0][0][0]) if len(outputs) > 0 else 0.5
        confidence = float(outputs[0][0][1]) if len(outputs[0][0]) > 1 else 0.8
        
        # Determine urgency tier
        if risk_score < 0.3:
            urgency = "Low"
        elif risk_score < 0.6:
            urgency = "Moderate"
        elif risk_score < 0.8:
            urgency = "High"
        else:
            urgency = "Critical"
        
        # Extract key features (simplified)
        key_features = self._extract_key_features(mel_spectrogram)
        
        return {
            'risk_score': risk_score,
            'confidence': confidence,
            'urgency_tier': urgency,
            'cough_quality': 'Good',  # Would need separate model
            'key_features': key_features
        }
    
    def _resize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """Resize spectrogram to target dimensions."""
        from scipy.ndimage import zoom
        
        current_h, current_w = spec.shape
        target_h, target_w = self.target_size
        
        zoom_h = target_h / current_h
        zoom_w = target_w / current_w
        
        return zoom(spec, (zoom_h, zoom_w), order=1)
    
    def _extract_key_features(self, mel_spec: np.ndarray) -> List[str]:
        """Extract explainable features from spectrogram."""
        features = []
        
        # High frequency energy
        high_freq_energy = np.mean(mel_spec[64:, :])  # Upper half
        if high_freq_energy > -40:
            features.append("High frequency energy")
        
        # Spectral spread
        mean_freq = np.mean(np.argmax(mel_spec, axis=0))
        if mean_freq > 64:
            features.append("Elevated frequency centroid")
        
        # Temporal characteristics
        temporal_var = np.var(np.mean(mel_spec, axis=0))
        if temporal_var > 100:
            features.append("Variable temporal pattern")
        
        return features if features else ["Normal spectral characteristics"]

class VoiceAnalyzer:
    """Voice Pathology Analyzer."""
    
    def analyze(self, features: dict) -> dict:
        """Analyze voice for pathology."""
        jitter = features.get('jitter', 0)
        shimmer = features.get('shimmer', 0)
        hnr = features.get('hnr', 0)
        cpp = features.get('cpp', 0)
        f0_values = features.get('f0_values', [])
        
        f0_mean = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
        f0_std = float(np.std(f0_values)) if len(f0_values) > 0 else 0.0
        
        # Detect pathologies based on thresholds
        pathologies = []
        
        if jitter > 1.04:
            pathologies.append("Dysphonia")
        if shimmer > 0.12:
            pathologies.append("Hoarseness")
        if hnr < 7.0:
            pathologies.append("Breathiness")
        if f0_std > 10.0:
            pathologies.append("Tremor")
        if cpp < 5.0:
            pathologies.append("Roughness")
        
        confidence = 0.5 + min(len(pathologies) * 0.1, 0.4)
        
        return {
            'pathology_detected': len(pathologies) > 0,
            'pathology_types': pathologies,
            'confidence': confidence,
            'jitter_percent': round(jitter, 2),
            'shimmer_percent': round(shimmer, 2),
            'hnr_db': round(hnr, 2),
            'cpp_db': round(cpp, 2),
            'f0_mean': round(f0_mean, 2),
            'f0_std': round(f0_std, 2)
        }

class SoundDetector:
    """Detect sound events in audio."""
    
    def detect(self, features: dict) -> List[dict]:
        """Detect sound events."""
        events = []
        
        # Detect cough based on characteristics
        mel_spec = features.get('mel_spectrogram', np.array([]))
        rms = features.get('rms_energy', [])
        zcr = features.get('zero_crossing_rate', [])
        
        if len(mel_spec) > 0 and len(rms) > 0:
            # Cough detection heuristic
            max_rms = np.max(rms)
            mean_zcr = np.mean(zcr)
            
            if max_rms > 0.3 and mean_zcr > 0.05:
                events.append({
                    'event_type': 'Cough',
                    'confidence': min(max_rms * 2, 0.95),
                    'start_time_ms': 0,
                    'end_time_ms': int(features.get('duration', 0) * 1000),
                    'frequency_range': [100.0, 8000.0]
                })
            
            # Speech detection
            spectral_centroid = features.get('spectral_centroid', [])
            if len(spectral_centroid) > 0:
                mean_centroid = np.mean(spectral_centroid)
                if 200 < mean_centroid < 4000:
                    events.append({
                        'event_type': 'Speech',
                        'confidence': 0.85,
                        'start_time_ms': 0,
                        'end_time_ms': int(features.get('duration', 0) * 1000),
                        'frequency_range': [80.0, 8000.0]
                    })
        
        return events

# =============================================================================
# Geographic Clustering
# =============================================================================

class GeoClustering:
    """Cluster detections geographically."""
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points."""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    @classmethod
    def cluster(cls, detections: List[dict], eps_km: float = 5.0, min_samples: int = 3) -> List[Cluster]:
        """Cluster detections using DBSCAN."""
        if len(detections) < min_samples:
            return []
        
        # Extract coordinates
        coords = np.array([[d['lat'], d['lon']] for d in detections])
        
        # Custom distance metric
        def metric(a, b):
            return cls.haversine_distance(a[0], a[1], b[0], b[1])
        
        # Run DBSCAN
        clustering = DBSCAN(eps=eps_km, min_samples=min_samples, metric=metric).fit(coords)
        
        # Build clusters
        clusters = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = clustering.labels_ == label
            cluster_coords = coords[mask]
            cluster_detections = [d for d, m in zip(detections, mask) if m]
            
            center_lat = float(np.mean(cluster_coords[:, 0]))
            center_lon = float(np.mean(cluster_coords[:, 1]))
            
            # Calculate radius
            distances = [
                cls.haversine_distance(center_lat, center_lon, c[0], c[1])
                for c in cluster_coords
            ]
            radius = max(distances) if distances else 0.0
            
            # Count high risk
            high_risk = sum(1 for d in cluster_detections if d.get('risk_score', 0) > 0.7)
            
            # Get time range
            timestamps = [d.get('timestamp', '') for d in cluster_detections]
            timestamps.sort()
            
            clusters.append(Cluster(
                cluster_id=f"cluster_{uuid.uuid4().hex[:8]}",
                center=GeoLocation(latitude=center_lat, longitude=center_lon),
                radius_km=round(radius, 2),
                detection_count=len(cluster_detections),
                high_risk_count=high_risk,
                first_detection=timestamps[0] if timestamps else '',
                last_detection=timestamps[-1] if timestamps else ''
            ))
        
        return clusters

# =============================================================================
# In-Memory Store (Replace with database in production)
# =============================================================================

class DataStore:
    """Simple in-memory data store."""
    
    def __init__(self):
        self.records: List[dict] = []
    
    def add(self, record: dict):
        self.records.append(record)
    
    def get_recent(self, days: int = 30) -> List[dict]:
        cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        return [
            r for r in self.records
            if datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')).timestamp() > cutoff
        ]
    
    def get_all(self) -> List[dict]:
        return self.records.copy()

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="VoiceX API",
    description="AI Audio Triage for TB Risk and Voice Pathology",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
feature_extractor = FeatureExtractor()
store = DataStore()
tb_classifier: Optional[TBClassifier] = None
voice_analyzer = VoiceAnalyzer()
sound_detector = SoundDetector()

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/predict", response_model=AnalysisResult)
async def predict(
    audio: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    device_id: Optional[str] = Form(None)
):
    """Analyze audio for TB risk and voice pathology."""
    start_time = time.time()
    
    # Read audio
    audio_bytes = await audio.read()
    
    # Extract features
    features = feature_extractor.extract(audio_bytes)
    
    # Run TB classification
    if tb_classifier:
        tb_result = tb_classifier.predict(features['mel_spectrogram'])
    else:
        # Fallback without model
        tb_result = {
            'risk_score': 0.5,
            'confidence': 0.8,
            'urgency_tier': 'Moderate',
            'cough_quality': 'Good',
            'key_features': ['No model loaded']
        }
    
    # Analyze voice
    voice_result = voice_analyzer.analyze(features)
    
    # Detect sound events
    sound_events = sound_detector.detect(features)
    
    # Build result
    result = AnalysisResult(
        record_id=f"rec_{uuid.uuid4().hex}",
        timestamp=datetime.utcnow().isoformat() + "Z",
        location=GeoLocation(latitude=lat, longitude=lon) if lat and lon else None,
        tb_risk=TBRiskResult(**tb_result),
        voice_pathology=VoicePathologyResult(**voice_result),
        sound_events=[SoundEvent(**e) for e in sound_events],
        audio_metadata=AudioMetadata(
            duration_secs=features['duration'],
            sample_rate=features['sample_rate'],
            channels=features['channels']
        )
    )
    
    # Store record
    store.add({
        'record_id': result.record_id,
        'timestamp': result.timestamp,
        'lat': lat,
        'lon': lon,
        'risk_score': tb_result['risk_score'],
        'urgency_tier': tb_result['urgency_tier'],
        'pathology_detected': voice_result['pathology_detected']
    })
    
    processing_time = (time.time() - start_time) * 1000
    
    return result

@app.post("/batch_predict")
async def batch_predict(
    audio_files: List[UploadFile] = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None)
):
    """Batch process multiple audio files."""
    results = []
    
    for audio in audio_files:
        result = await predict(audio, lat, lon)
        results.append(result)
    
    return {
        "status": "success",
        "processed": len(results),
        "results": results
    }

@app.get("/clusters")
async def get_clusters(
    days: int = 30,
    eps_km: float = 5.0,
    min_points: int = 3
):
    """Get geographic clusters of detections."""
    recent = store.get_recent(days)
    
    clusters = GeoClustering.cluster(recent, eps_km, min_points)
    
    return {
        "status": "success",
        "clusters": clusters
    }

@app.get("/heatmap")
async def get_heatmap(
    days: int = 30,
    resolution_km: float = 1.0
):
    """Get heatmap data."""
    recent = store.get_recent(days)
    
    # Simple grid-based heatmap
    grid = {}
    for r in recent:
        if r.get('lat') and r.get('lon'):
            # Bin to grid
            grid_x = int(r['lon'] / (resolution_km / 111.0))
            grid_y = int(r['lat'] / (resolution_km / 111.0))
            key = (grid_x, grid_y)
            
            if key not in grid:
                grid[key] = {'count': 0, 'risk_sum': 0, 'lats': [], 'lons': []}
            
            grid[key]['count'] += 1
            grid[key]['risk_sum'] += r.get('risk_score', 0)
            grid[key]['lats'].append(r['lat'])
            grid[key]['lons'].append(r['lon'])
    
    points = []
    for key, data in grid.items():
        center_lat = np.mean(data['lats'])
        center_lon = np.mean(data['lons'])
        avg_risk = data['risk_sum'] / data['count'] if data['count'] > 0 else 0
        intensity = min((data['count'] * 0.3 + avg_risk * 0.7), 1.0)
        
        points.append(HeatmapPoint(
            latitude=center_lat,
            longitude=center_lon,
            intensity=round(intensity, 2),
            count=data['count'],
            risk_score=round(avg_risk, 2)
        ))
    
    return {
        "status": "success",
        "points": points
    }

@app.post("/sync")
async def sync_data(payload: SyncPayload):
    """Receive sync data from edge devices."""
    # In production, store in database
    for record in payload.records:
        store.add(record)
    
    return {
        "status": "success",
        "received": len(payload.records),
        "synced_ids": [r.get('record_id') for r in payload.records]
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    all_records = store.get_all()
    
    total = len(all_records)
    high_risk = sum(1 for r in all_records if r.get('risk_score', 0) > 0.7)
    pathology = sum(1 for r in all_records if r.get('pathology_detected', False))
    
    return {
        "status": "success",
        "stats": {
            "total_records": total,
            "high_risk_count": high_risk,
            "pathology_detected_count": pathology,
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tb_classifier is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VoiceX API Server")
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # Load model if provided
    global tb_classifier
    if args.model and Path(args.model).exists():
        print(f"Loading model from {args.model}")
        tb_classifier = TBClassifier(args.model)
        print("Model loaded successfully")
    else:
        print("Warning: No model loaded. Using fallback predictions.")
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
