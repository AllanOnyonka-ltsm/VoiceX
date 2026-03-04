#!/usr/bin/env python3
"""
Test script for VoiceX API

Usage:
    python test_api.py --audio sample.wav --api http://localhost:8000
"""

import argparse
import json
import time
from pathlib import Path

import requests


def test_predict(api_url: str, audio_path: str, lat: float = None, lon: float = None):
    """Test the /predict endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing /predict endpoint")
    print(f"{'='*60}")
    
    url = f"{api_url}/predict"
    
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        data = {}
        if lat is not None:
            data['lat'] = lat
        if lon is not None:
            data['lon'] = lon
        data['device_id'] = 'test_device_001'
        
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    print(f"Response time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nRecord ID: {result['record_id']}")
        print(f"Timestamp: {result['timestamp']}")
        
        print(f"\n--- TB Risk Assessment ---")
        tb = result['tb_risk']
        print(f"Risk Score: {tb['risk_score']:.3f}")
        print(f"Confidence: {tb['confidence']:.3f}")
        print(f"Urgency Tier: {tb['urgency_tier']}")
        print(f"Cough Quality: {tb['cough_quality']}")
        print(f"Key Features: {', '.join(tb['key_features'])}")
        
        print(f"\n--- Voice Pathology ---")
        voice = result['voice_pathology']
        print(f"Pathology Detected: {voice['pathology_detected']}")
        if voice['pathology_detected']:
            print(f"Types: {', '.join(voice['pathology_types'])}")
        print(f"Jitter: {voice['jitter_percent']:.2f}%")
        print(f"Shimmer: {voice['shimmer_percent']:.2f}%")
        print(f"HNR: {voice['hnr_db']:.2f} dB")
        print(f"CPP: {voice['cpp_db']:.2f} dB")
        print(f"F0 Mean: {voice['f0_mean']:.2f} Hz")
        print(f"F0 Std: {voice['f0_std']:.2f} Hz")
        
        print(f"\n--- Sound Events ---")
        for event in result['sound_events']:
            print(f"  {event['event_type']}: {event['confidence']:.2f} confidence")
        
        print(f"\n--- Audio Metadata ---")
        meta = result['audio_metadata']
        print(f"Duration: {meta['duration_secs']:.2f}s")
        print(f"Sample Rate: {meta['sample_rate']} Hz")
        print(f"Channels: {meta['channels']}")
        
        if result.get('location'):
            loc = result['location']
            print(f"\nLocation: {loc['latitude']:.4f}, {loc['longitude']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    return response


def test_clusters(api_url: str, days: int = 30):
    """Test the /clusters endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing /clusters endpoint")
    print(f"{'='*60}")
    
    url = f"{api_url}/clusters"
    params = {'days': days, 'eps_km': 5.0, 'min_points': 3}
    
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        clusters = data.get('clusters', [])
        print(f"Found {len(clusters)} clusters")
        
        for i, cluster in enumerate(clusters[:5], 1):
            print(f"\nCluster {i}: {cluster['cluster_id']}")
            print(f"  Center: {cluster['center']['latitude']:.4f}, {cluster['center']['longitude']:.4f}")
            print(f"  Radius: {cluster['radius_km']:.2f} km")
            print(f"  Detections: {cluster['detection_count']}")
            print(f"  High Risk: {cluster['high_risk_count']}")
    else:
        print(f"Error: {response.text}")
    
    return response


def test_heatmap(api_url: str, days: int = 30):
    """Test the /heatmap endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing /heatmap endpoint")
    print(f"{'='*60}")
    
    url = f"{api_url}/heatmap"
    params = {'days': days, 'resolution_km': 1.0}
    
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        points = data.get('points', [])
        print(f"Found {len(points)} heatmap points")
        
        # Show top 5 by intensity
        sorted_points = sorted(points, key=lambda p: p['intensity'], reverse=True)
        for i, point in enumerate(sorted_points[:5], 1):
            print(f"\nPoint {i}:")
            print(f"  Location: {point['latitude']:.4f}, {point['longitude']:.4f}")
            print(f"  Intensity: {point['intensity']:.2f}")
            print(f"  Count: {point['count']}")
            print(f"  Avg Risk: {point['risk_score']:.2f}")
    else:
        print(f"Error: {response.text}")
    
    return response


def test_stats(api_url: str):
    """Test the /stats endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing /stats endpoint")
    print(f"{'='*60}")
    
    url = f"{api_url}/stats"
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        stats = data.get('stats', {})
        print(f"Total Records: {stats.get('total_records', 0)}")
        print(f"High Risk Count: {stats.get('high_risk_count', 0)}")
        print(f"Pathology Detected: {stats.get('pathology_detected_count', 0)}")
        print(f"Last Updated: {stats.get('last_updated', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    
    return response


def test_health(api_url: str):
    """Test the /health endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing /health endpoint")
    print(f"{'='*60}")
    
    url = f"{api_url}/health"
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Model Loaded: {data.get('model_loaded')}")
        print(f"Timestamp: {data.get('timestamp')}")
    else:
        print(f"Error: {response.text}")
    
    return response


def generate_test_audio(output_path: str, duration: float = 3.0):
    """Generate a test audio file."""
    import numpy as np
    import soundfile as sf
    
    # Generate synthetic cough-like sound
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create cough-like pattern (short bursts)
    signal = np.zeros_like(t)
    
    # Add some bursts
    for start in [0.5, 1.5, 2.5]:
        burst_duration = 0.2
        burst_start = int(start * sample_rate)
        burst_end = int((start + burst_duration) * sample_rate)
        
        if burst_end < len(signal):
            burst = np.random.randn(burst_end - burst_start) * 0.5
            burst *= np.hanning(len(burst))
            signal[burst_start:burst_end] = burst
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save
    sf.write(output_path, signal, sample_rate)
    print(f"Generated test audio: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test VoiceX API")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument("--lat", type=float, help="GPS latitude")
    parser.add_argument("--lon", type=float, help="GPS longitude")
    parser.add_argument("--generate-audio", action="store_true", help="Generate test audio")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Generate test audio if requested
    if args.generate_audio:
        args.audio = generate_test_audio("test_cough.wav")
    
    # Run tests
    if args.test_all or args.audio:
        if not args.audio:
            print("Error: No audio file provided. Use --audio or --generate-audio")
            return
        
        test_predict(args.api, args.audio, args.lat, args.lon)
    
    if args.test_all:
        test_health(args.api)
        test_stats(args.api)
        test_clusters(args.api)
        test_heatmap(args.api)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
