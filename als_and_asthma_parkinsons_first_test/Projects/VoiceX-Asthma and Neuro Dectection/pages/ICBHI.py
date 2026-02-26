import streamlit as st
import numpy as np
import librosa
import pickle
import io
import tempfile
import os
import time
import warnings
import pandas as pd
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import noisereduce as nr
from datetime import datetime
import sklearn  # Fixed: was missing space
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import plotly.express as px
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import threading
from collections import deque
import gc
st.title("Breath Sound Classification")
st.write("Record your breath sound to get a classification.")

# Constants for Mel spectrogram generation
SR = 16000  # Reduced from 22050 for faster processing
N_MELS = 64  # Reduced from 128
N_FFT = 1024  # Reduced from 2048
HOP = 256   # Reduced from 512
FMIN, FMAX = 20, 8000
IMG_SIZE = (224, 224)

# Cache compiled functions and models
@st.cache_resource
def load_model():
    """Load and cache the model"""
    num_classes = 4
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load("/workspaces/Projects/VoiceX/Models/best_model.pth", map_location=device))
        model.to(device)
        model.eval()
        
        # Optimize model for inference
        if hasattr(torch.jit, 'script'):
            try:
                model = torch.jit.script(model)
            except:
                pass  # Fall back to regular model if scripting fails
                
        return model, device
    except FileNotFoundError:
        st.error("Error: Model file 'best_model.pth' not found.")
        return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

@st.cache_resource
def get_transforms():
    """Cache transforms"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE[0], IMG_SIZE[1]), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Optimized mel spectrogram function
@st.cache_data
def mel_spectrogram_cached(y_hash, sr):
    """Cached mel spectrogram generation"""
    # This won't actually cache due to array input, but shows the pattern
    pass

def mel_spectrogram_optimized(y, sr):
    """Optimized Mel spectrogram generation"""
    # Use power spectrogram directly (faster than magnitude + power conversion)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0  # Direct power spectrogram
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def spec_db_to_uint8_vectorized(spec_db, min_db=-80.0, max_db=0.0):
    """Vectorized conversion to uint8"""
    spec_clipped = np.clip(spec_db, min_db, max_db)
    spec_normalized = (spec_clipped - min_db) / (max_db - min_db)
    return (spec_normalized * 255.0).astype(np.uint8)

# Load model once at startup
model, device = load_model()
data_transforms = get_transforms()
class_names = ['both', 'crackle', 'normal', 'wheeze']

# Pre-allocate arrays for better memory management
MAX_BUFFER_SIZE = SR * 10  # 10 seconds max

class OptimizedAudioRecorder(AudioProcessorBase):
    def __init__(self):
        # Use deque for efficient append operations
        self.audio_buffer = deque(maxlen=MAX_BUFFER_SIZE)
        self.spectrogram_image = None
        self.prediction = None
        self.lock = threading.Lock()
        
    def recv(self, frame):
        """Efficiently append audio frames"""
        with self.lock:
            audio_chunk = frame.to_ndarray().flatten()
            self.audio_buffer.extend(audio_chunk)
        return frame

    def process_audio_and_generate_spectrogram(self):
        """Optimized audio processing"""
        if model is None:
            return "Model not loaded. Cannot make prediction.", None
            
        with self.lock:
            if len(self.audio_buffer) == 0:
                return "No audio recorded. Please record for 2-3 seconds and try again.", None
            
            # Convert deque to numpy array efficiently
            audio_data = np.array(self.audio_buffer, dtype=np.float32)
        
        try:
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Trim to reasonable length (5 seconds max) for faster processing
            max_samples = SR * 5
            if len(audio_data) > max_samples:
                # Take middle portion
                start_idx = (len(audio_data) - max_samples) // 2
                audio_data = audio_data[start_idx:start_idx + max_samples]
            
            # Generate spectrogram with optimized function
            spec_db = mel_spectrogram_optimized(audio_data, SR)
            
            # Vectorized conversion
            spec_u8 = spec_db_to_uint8_vectorized(spec_db)
            
            # Efficient padding/cropping
            current_width = spec_u8.shape[1]
            target_width = IMG_SIZE[1]
            
            if current_width < target_width:
                # Pad
                padding = target_width - current_width
                spec_u8 = np.pad(spec_u8, ((0, 0), (0, padding)), mode='constant')
            elif current_width > target_width:
                # Crop center
                start_col = (current_width - target_width) // 2
                spec_u8 = spec_u8[:, start_col:start_col + target_width]
            
            # Create image with optimized settings
            img = Image.fromarray(spec_u8, mode="L")
            img = img.resize(IMG_SIZE, Image.LANCZOS)  # LANCZOS is faster than BICUBIC
            img = img.convert("RGB")
            
            self.spectrogram_image = img
            
            # Efficient prediction
            img_tensor = data_transforms(img).unsqueeze(0).to(device, non_blocking=True)
            
            with torch.no_grad():
                # Use half precision if supported for faster inference
                if device.type == 'cuda':
                    img_tensor = img_tensor.half()
                    model_half = model.half() if hasattr(model, 'half') else model
                    outputs = model_half(img_tensor)
                else:
                    outputs = model(img_tensor)
                
                predicted_class_index = torch.argmax(outputs, dim=1)
            
            prediction = class_names[predicted_class_index.item()]
            self.prediction = prediction
            
            # Clear GPU cache if using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return f"Predicted Class: **{prediction}**", img
            
        except Exception as e:
            return f"Processing error: {str(e)}", None
        finally:
            # Force garbage collection
            gc.collect()

# Optimized WebRTC configuration
rtc_config = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Use optimized processor
ctx = webrtc_streamer(
    key="audio-recorder",
    audio_processor_factory=OptimizedAudioRecorder,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "audio": {
            "sampleRate": SR,  # Match our processing sample rate
            "channelCount": 1,  # Mono for efficiency
            "echoCancellation": True,
            "noiseSuppression": True
        }, 
        "video": False
    }
)

# Streamlined UI
col1, col2 = st.columns([2, 1])

with col1:
    st.info("""
        **Quick Recording**:
        1. Click 'Start' → Record 2-3 seconds → Click 'Stop'
        2. Click 'Process' for instant analysis
    """)

with col2:
    if st.button("Process Recording", type="primary") and ctx.audio_processor:
        # Use spinner for better UX without fake delays
        with st.spinner("Analyzing..."):
            result, spec_img = ctx.audio_processor.process_audio_and_generate_spectrogram()
        
        # Display results immediately
        if spec_img is not None:
            st.image(spec_img, caption="Spectrogram", width=300)
        
        if "Predicted Class" in result:
            st.success(result)
        elif "error" in result.lower():
            st.error(result)
        else:
            st.warning(result)

# Performance tips
with st.expander("Performance Tips"):
    st.markdown("""
    - **GPU Acceleration**: Using CUDA if available
    - **Reduced Resolution**: Optimized spectrogram parameters
    - **Memory Management**: Efficient buffering and cleanup
    - **Model Optimization**: JIT compilation when possible
    - **Vectorized Operations**: NumPy optimizations throughout
    """)

# System info
if st.checkbox("Show System Info"):
    st.code(f"""
    Device: {device}
    PyTorch Version: {torch.__version__}
    CUDA Available: {torch.cuda.is_available()}
    Sample Rate: {SR} Hz
    Spectrogram Size: {N_MELS}x{IMG_SIZE[0]}
    """)