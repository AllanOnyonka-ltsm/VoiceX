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
# Suppress warnings for cleaner UI (after handling critical ones)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="ALS Audio Detection",
    layout="wide"
)

# Constants
SAMPLE_RATE = 16000
MIN_DURATION = 3.0
MAX_DURATION = 5.0
FEATURE_NAMES = [
    'PVI_a', 'PFR_a', 'CCi_3', 'CCi_2', 'CCi_6', 
    'PVI_i', 'PPE_a', 'Hi_8_sd', 'Ha_8_rel', 'dCCi_6'
]

def check_sklearn_version():
    """Check if scikit-learn version matches training environment"""
    try:
        # Get current version
        current_version = sklearn.__version__
        
        # Check if we can detect the training version
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Try to load a dummy model to trigger warning
                try:
                    with open('Models/als_model.pkl', 'rb') as f:
                        pickle.load(f)
                except:
                    pass
                
                # Check for version warning
                for warning in w:
                    if "InconsistentVersionWarning" in str(warning.message):
                        original_version = str(warning.message).split("version ")[1].split(" ")[0]
                        return current_version, original_version
            
            return current_version, None
            
        except Exception:
            return current_version, None
            
    except Exception as e:
        return "unknown", None

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler with version consistency checks"""
    current_version, training_version = check_sklearn_version()
    
    # Check if there's a version mismatch
    if training_version and current_version != training_version:
        warning_msg = (
            f"Version Mismatch Detected!\n\n"
            f"• Model trained with: scikit-learn {training_version}\n"
            f"• Current environment: scikit-learn {current_version}\n\n"
            f"This can lead to invalid predictions. You should either:\n"
            f"1. Downgrade scikit-learn: `pip install scikit-learn=={training_version}`\n"
            f"2. Retrain your model with current scikit-learn version"
        )
        st.warning(warning_msg)
    
    try:
        # Load your trained model and scaler
        with open('/workspaces/Projects/VoiceX/Models/als_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('/workspaces/Projects/VoiceX/Models/als_scaler.pkl', 'rb') as f:  # Fixed filename to match error message
            scaler = pickle.load(f)
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, f"Model files not found: {e}"
    except Exception as e:
        return None, None, f"Error loading model: {e}"

def denoise_audio(audio_data, sr=SAMPLE_RATE):
    """Apply noise reduction to improve signal quality"""
    try:
        # Simple noise reduction - only process first 500ms for noise profile
        noise = audio_data[:int(sr * 0.5)]
        reduced = nr.reduce_noise(y=audio_data, y_noise=noise, sr=sr, stationary=True)
        return reduced
    except Exception as e:
        st.warning(f"Noise reduction failed: {e}. Continuing with original audio.")
        return audio_data

def validate_audio(audio_data, sr=SAMPLE_RATE):
    """Validate audio quality and duration"""
    duration = len(audio_data) / sr
    
    # Check duration
    if duration < MIN_DURATION:
        return False, f"Audio too short ({duration:.1f}s). Please record for {MIN_DURATION}-{MAX_DURATION} seconds."
    if duration > MAX_DURATION + 1:  # Allow 1s buffer
        # Trim to MAX_DURATION
        audio_data = audio_data[:int(MAX_DURATION * sr)]
        duration = MAX_DURATION
    
    # Check for silence
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < 0.001:
        return False, "Audio too quiet. Please speak louder and closer to the microphone."
    
    return True, audio_data

def extract_minsk_features(audio_data, sr=SAMPLE_RATE):
    """
    Extract MINSK dataset features from audio data with optimized performance
    Returns features in the order expected by your model
    """
    features = {}
    
    try:
        # Basic audio properties
        duration = len(audio_data) / sr
        if duration < MIN_DURATION:  # Too short
            return None
            
        # Denoise audio before analysis
        audio_data = denoise_audio(audio_data, sr)
        
        # Pitch-based features using more efficient method than pyin
        f0 = librosa.yin(
            audio_data, 
            fmin=80, 
            fmax=300, 
            sr=sr, 
            frame_length=int(sr * 0.02),  # 20ms frames
            win_length=int(sr * 0.04)      # 40ms window
        )
        
        # Create voiced/unvoiced mask
        voiced_flag = (f0 < 300) & (f0 > 80)
        f0_clean = f0[voiced_flag]
        
        if len(f0_clean) < 10:  # Not enough voiced segments
            return None
        
        # PVI_a (Pairwise Variability Index for vowel /a/)
        if len(f0_clean) > 1:
            pvi_a = np.mean(np.abs(np.diff(f0_clean)) / (f0_clean[:-1] + f0_clean[1:] + 1e-8)) * 200
        else:
            pvi_a = 0
        features['PVI_a'] = pvi_a
        
        # PFR_a (Pitch Feature for /a/) - pitch variation
        pfr_a = np.std(f0_clean) / (np.mean(f0_clean) + 1e-8)
        features['PFR_a'] = pfr_a
        
        # PPE_a (Pitch Period Entropy for /a/)
        pitch_periods = 1.0 / (f0_clean + 1e-8)
        hist, _ = np.histogram(pitch_periods, bins=20, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        ppe_a = -np.sum(hist * np.log2(hist))
        features['PPE_a'] = ppe_a
        
        # Compute STFT once and reuse
        stft = librosa.stft(audio_data, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        
        # CCi features (Cross-correlation indices) - approximated using spectral correlation
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=13)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)
        
        # Cross-correlation approximations
        if mfcc.shape[1] > 10:
            cc_3 = np.corrcoef(mfcc[2], mfcc[3])[0, 1] if mfcc.shape[0] > 3 else 0
            cc_2 = np.corrcoef(mfcc[1], mfcc[2])[0, 1] if mfcc.shape[0] > 2 else 0
            cc_6 = np.corrcoef(mfcc[5], mfcc[6])[0, 1] if mfcc.shape[0] > 6 else 0
            
            # Differential cross-correlation
            mfcc_diff = np.diff(mfcc, axis=1)
            if mfcc_diff.shape[1] > 5 and mfcc_diff.shape[0] > 6:
                dcc_6 = np.corrcoef(mfcc_diff[5], mfcc_diff[6])[0, 1]
            else:
                dcc_6 = 0
        else:
            cc_3 = cc_2 = cc_6 = dcc_6 = 0
            
        features['CCi_3'] = cc_3
        features['CCi_2'] = cc_2  
        features['CCi_6'] = cc_6
        features['dCCi_6'] = dcc_6
        
        # PVI_i (for vowel /i/ - using higher frequency analysis)
        f0_high = librosa.yin(audio_data, fmin=150, fmax=500, sr=sr, frame_length=int(sr * 0.02))
        voiced_high = (f0_high < 500) & (f0_high > 150)
        f0_high_clean = f0_high[voiced_high]
        
        if len(f0_high_clean) > 1:
            pvi_i = np.mean(np.abs(np.diff(f0_high_clean)) / (f0_high_clean[:-1] + f0_high_clean[1:] + 1e-8)) * 200
        else:
            pvi_i = pvi_a  # Fallback
        features['PVI_i'] = pvi_i
        
        # Harmonic features Hi(8)_sd and Ha(8)_rel
        harmonics = librosa.effects.harmonic(audio_data)
        harmonic_spec = np.abs(librosa.stft(harmonics))
        
        # Hi(8)_sd - 8th harmonic standard deviation approximation
        if harmonic_spec.shape[0] > 8:
            hi_8_sd = np.std(harmonic_spec[7])  # 8th row (0-indexed)
        else:
            hi_8_sd = np.std(harmonic_spec[-1])  # Use last available
        features['Hi_8_sd'] = hi_8_sd
        
        # Ha(8)_rel - 8th harmonic relative energy
        total_harmonic_energy = np.sum(harmonic_spec**2)
        if harmonic_spec.shape[0] > 8 and total_harmonic_energy > 0:
            ha_8_rel = np.sum(harmonic_spec[7]**2) / total_harmonic_energy
        else:
            ha_8_rel = 0.1  # Default value
        features['Ha_8_rel'] = ha_8_rel
        
        return features
        
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def process_audio_bytes(audio_bytes, progress_bar, status_text):
    """Convert audio bytes to numpy array with progress tracking"""
    try:
        # Update progress
        status_text.text("Preparing audio for analysis...")
        progress_bar.progress(10)
        
        # Save to temporary file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Load with soundfile directly (more reliable than librosa.load)
        status_text.text("Loading audio data...")
        progress_bar.progress(20)
        
        try:
            # Try to use soundfile directly
            audio_data, sr = sf.read(tmp_path)
            
            # If stereo, convert to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Resample if needed
            if sr != SAMPLE_RATE:
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sr, 
                    target_sr=SAMPLE_RATE,
                    res_type='soxr_hq'
                )
                sr = SAMPLE_RATE
        except Exception as e:
            # Fallback to librosa if soundfile fails
            st.warning(f"SoundFile failed: {e}. Using librosa as fallback.")
            audio_data, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Validate audio
        status_text.text("Validating audio quality...")
        progress_bar.progress(30)
        is_valid, result = validate_audio(audio_data, sr)
        if not is_valid:
            st.error(result)
            return None, None
        
        return result, sr  # Return validated audio
        
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None, None

def create_waveform_plot(audio_data, sr):
    """Create interactive waveform visualization"""
    time = np.arange(len(audio_data)) / sr
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, 
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        showlegend=False
    )
    
    return fig

def create_feature_importance_plot(model, feature_vector_scaled):
    """Create feature importance visualization"""
    try:
        # For tree-based models, we can get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fig = go.Figure(data=[
                go.Bar(
                    x=FEATURE_NAMES, 
                    y=importances,
                    marker_color=['#e74c3c' if i in [0, 1, 5] else '#1f77b4' for i in range(len(FEATURE_NAMES))]
                )
            ])
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance",
                height=300
            )
            return fig
        
        # For other models, create a coefficient-based plot if possible
        elif hasattr(model, 'coef_'):
            # Get absolute values for importance
            coef = np.abs(model.coef_[0])
            fig = go.Figure(data=[
                go.Bar(
                    x=FEATURE_NAMES, 
                    y=coef,
                    marker_color=['#e74c3c' if i in [0, 1, 5] else '#1f77b4' for i in range(len(FEATURE_NAMES))]
                )
            ])
            fig.update_layout(
                title="Feature Influence",
                xaxis_title="Features",
                yaxis_title="Coefficient Magnitude",
                height=300
            )
            return fig
            
        return None
    except:
        return None

def main():
    st.title("ALS Audio Detection System")
    st.markdown("Advanced analysis of voice patterns for early ALS detection")
    st.markdown("---")
    
    # Check scikit-learn version before loading model
    current_version, training_version = check_sklearn_version()
    
    if training_version and current_version != training_version:
        st.warning(f"Version Mismatch: Model trained with scikit-learn {training_version}, current version is {current_version}")
        st.info("This can lead to invalid predictions. Consider downgrading: `pip install scikit-learn=={training_version}`")
    
    # Load model and scaler
    model, scaler, error = load_model_and_scaler()
    
    if error:
        st.error(error)
        st.info("Please ensure 'models/als_model.pkl' and 'models/als_scaler.pkl' exist in your project directory.")
        st.stop()
    
    # Initialize session state
    if 'vowel_a_audio' not in st.session_state:
        st.session_state.vowel_a_audio = None
    if 'vowel_i_audio' not in st.session_state:
        st.session_state.vowel_i_audio = None
    if 'recording_start' not in st.session_state:
        st.session_state.recording_start = None
    if 'recording_duration' not in st.session_state:
        st.session_state.recording_duration = 0.0
    
    # Sidebar instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        **Recording Guidelines:**
        
        1. **Vowel /a/**: Record 5 seconds of sustained "ahhhhh"
        2. **Vowel /i/**: Record 5 seconds of sustained "eeeeee"
        3. **Quality**: Speak clearly in a quiet environment
        4. **Distance**: Stay 6-12 inches from microphone
        
        **Analysis Process:**
        - Extract MINSK dataset features
        - Normalize using trained scaler  
        - Predict ALS likelihood
        - Display confidence scores
        """)
        
        st.markdown("---")
        st.subheader("System Status")
        st.info(f"• Model loaded: {type(model).__name__}\n"
                f"• Audio sample rate: {SAMPLE_RATE} Hz\n"
                f"• Required duration: {MIN_DURATION}-{MAX_DURATION} seconds\n"
                f"• scikit-learn: {sklearn.__version__}")
        
        st.markdown("---")
        st.subheader("System Requirements")
        st.markdown("""
        - Python 3.8+
        - FFmpeg installed (`sudo apt-get install ffmpeg` on Linux)
        - SoundFile library (`pip install soundfile`)
        - libsndfile system dependency (`sudo apt-get install libsndfile1`)
        - noisereduce (`pip install noisereduce`)
        """)

    # Main interface
    col1, col2 = st.columns(2)
    
    # Function to handle recording with duration feedback
    def handle_recording(column, vowel, color, key):
        with column:
            st.subheader(f"Vowel /{vowel}/ Recording")
            st.info(f"Say '{'ahhhhh' if vowel == 'a' else 'eeeeee'}' for {MIN_DURATION}-{MAX_DURATION} seconds")
            
            # Recording status
            status_col1, status_col2 = st.columns([2, 1])
            with status_col1:
                recording_status = st.empty()
            with status_col2:
                timer_display = st.empty()
            
            # Start recording
            audio_bytes = audio_recorder(
                text=f"Record /{vowel}/",
                recording_color=color,
                neutral_color="#95a5a6",
                icon_name="microphone",
                icon_size="2x",
                key=key
            )
            
            # Process recording
            if audio_bytes:
                # Save to session state
                st.session_state[f"vowel_{vowel}_audio"] = audio_bytes
                st.success(f"/{vowel}/ recording captured!")
                
                # Process and visualize
                audio_data, sr = process_audio_bytes(audio_bytes, st.empty(), st.empty())
                if audio_data is not None:
                    duration = len(audio_data) / sr
                    st.metric("Duration", f"{duration:.1f} seconds")
                    
                    # Waveform visualization
                    fig = create_waveform_plot(audio_data, sr)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Audio playback
                    st.audio(audio_bytes, format='audio/wav')
                    
                    # Quality assessment
                    rms = np.sqrt(np.mean(audio_data**2))
                    if rms < 0.01:
                        st.warning("Low audio volume detected. Results may be less accurate.")
                    if duration < MIN_DURATION:
                        st.warning(f"Recording too short ({duration:.1f}s). Aim for {MIN_DURATION}-{MAX_DURATION} seconds.")
                    elif duration > MAX_DURATION:
                        st.warning(f"Recording trimmed to {MAX_DURATION} seconds.")
            
            return audio_bytes
    
    # Handle both recordings with proper status feedback
    audio_bytes_a = handle_recording(col1, "a", "#e74c3c", "vowel_a")
    audio_bytes_i = handle_recording(col2, "i", "#3498db", "vowel_i")
    
    # Clear recordings button
    if st.button("Clear All Recordings", type="secondary"):
        st.session_state.vowel_a_audio = None
        st.session_state.vowel_i_audio = None
        st.rerun()
    
    # Analysis section
    st.markdown("---")
    st.header("ALS Analysis")
    
    if st.session_state.vowel_a_audio and st.session_state.vowel_i_audio:
        
        if st.button("Analyze Audio for ALS Detection", type="primary", use_container_width=True):
            # Create progress tracking elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_display = st.empty()
            start_time = time.time()
            
            try:
                # Step 1: Process audio files
                status_text.text("Step 1/6: Processing audio files...")
                time_display.text("Estimated time: 3-5 seconds")
                progress_bar.progress(10)
                
                audio_a, sr_a = process_audio_bytes(
                    st.session_state.vowel_a_audio, 
                    progress_bar, 
                    status_text
                )
                audio_i, sr_i = process_audio_bytes(
                    st.session_state.vowel_i_audio, 
                    progress_bar, 
                    status_text
                )
                
                if audio_a is None or audio_i is None:
                    st.error("Failed to process audio recordings")
                    return
                
                # Step 2: Extract features from /a/
                status_text.text("Step 2/6: Extracting features from /a/ sound...")
                progress_bar.progress(25)
                features_a = extract_minsk_features(audio_a, sr_a)
                
                if features_a is None:
                    st.error("Failed to extract features from /a/ recording")
                    return
                
                # Step 3: Extract features from /i/
                status_text.text("Step 3/6: Extracting features from /i/ sound...")
                progress_bar.progress(40)
                features_i = extract_minsk_features(audio_i, sr_i)
                
                if features_i is None:
                    st.error("Failed to extract features from /i/ recording")
                    return
                
                # Step 4: Combine features
                status_text.text("Step 4/6: Combining and validating features...")
                progress_bar.progress(60)
                
                # Combine features
                combined_features = {
                    'PVI_a': features_a['PVI_a'],
                    'PFR_a': features_a['PFR_a'],
                    'PPE_a': features_a['PPE_a'],
                    'CCi_3': features_a['CCi_3'],
                    'CCi_2': features_a['CCi_2'],
                    'CCi_6': features_a['CCi_6'],
                    'dCCi_6': features_a['dCCi_6'],
                    'Hi_8_sd': features_a['Hi_8_sd'],
                    'Ha_8_rel': features_a['Ha_8_rel'],
                    'PVI_i': features_i['PVI_i']
                }
                
                # Step 5: Scale features
                status_text.text("Step 5/6: Normalizing feature values...")
                progress_bar.progress(80)
                
                # Create feature vector
                feature_vector = np.array([combined_features[name] for name in FEATURE_NAMES]).reshape(1, -1)
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Step 6: Make prediction
                status_text.text("Step 6/6: Running ALS prediction model...")
                progress_bar.progress(90)
                
                # Make prediction
                prediction = model.predict(feature_vector_scaled)[0]
                prediction_proba = model.predict_proba(feature_vector_scaled)[0]
                
                # Complete progress
                progress_bar.progress(100)
                elapsed_time = time.time() - start_time
                time_display.text(f"Analysis completed in {elapsed_time:.1f} seconds")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:  # Assuming 1 = ALS positive
                        st.error("ALS Risk Detected")
                        confidence = prediction_proba[1] * 100
                    else:
                        st.success("No ALS Risk Detected")
                        confidence = prediction_proba[0] * 100
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    risk_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
                    color = "#e74c3c" if risk_level == "High" else "#f39c12" if risk_level == "Medium" else "#2ecc71"
                    st.markdown(f'<p style="color:{color};font-size:24px;font-weight:bold;text-align:center;">{risk_level}</p>', 
                               unsafe_allow_html=True)
                    st.caption("Risk Level")
                
                # Feature importance
                st.subheader("Model Insights")
                importance_plot = create_feature_importance_plot(model, feature_vector_scaled)
                if importance_plot:
                    st.plotly_chart(importance_plot, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
                
                # Feature breakdown
                st.subheader("Feature Analysis")
                
                # Create comparison DataFrame
                feature_df = pd.DataFrame({
                    'Feature': FEATURE_NAMES,
                    'Description': [
                        'Pitch Variability Index (/a/)',
                        'Pitch Fluctuation Rate',
                        'Cross-Correlation Index 3',
                        'Cross-Correlation Index 2',
                        'Cross-Correlation Index 6',
                        'Pitch Variability Index (/i/)',
                        'Pitch Period Entropy',
                        'Harmonic Irregularity (8th)',
                        'Harmonic Amplitude Relative',
                        'Differential CC Index 6'
                    ],
                    'Value': [combined_features[name] for name in FEATURE_NAMES],
                    'Status': ['High' if (name in ['PVI_a', 'PFR_a', 'PVI_i'] and combined_features[name] > 0.5) 
                              else 'Normal' for name in FEATURE_NAMES]
                })
                
                # Apply styling to DataFrame
                def color_status(val):
                    color = '#e74c3c' if val == 'High' else '#27ae60'
                    return f'color: {color}'
                
                styled_df = feature_df.style.applymap(
                    color_status, 
                    subset=['Status']
                ).format({
                    'Value': '{:.4f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Probability breakdown
                st.subheader("Prediction Confidence")
                
                prob_df = pd.DataFrame({
                    'Class': ['Normal', 'ALS Risk'],
                    'Probability': prediction_proba
                })
                
                fig_prob = go.Figure(data=[
                    go.Bar(x=prob_df['Class'], y=prob_df['Probability'],
                          marker_color=['#2ecc71', '#e74c3c'])
                ])
                fig_prob.update_layout(
                    title="Classification Probabilities", 
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Clinical interpretation
                st.subheader("Clinical Interpretation")
                
                if prediction == 1:
                    st.markdown("""
                    **Interpretation:** The voice analysis shows patterns associated with ALS.
                    
                    **Key indicators:**
                    - Elevated pitch variability (PVI)
                    - Increased pitch fluctuation rate
                    - Abnormal harmonic patterns
                    
                    **Recommendation:** 
                    This result should not be considered as part of a comprehensive clinical evaluation. 
                    Please consult with a neurologist for further assessment and diagnostic testing.
                    """)
                else:
                    st.markdown("""
                    **Interpretation:** The voice analysis shows normal patterns without ALS indicators.
                    
                    **Note:** 
                    This screening tool does not rule out all neuromuscular conditions. 
                    If symptoms persist or worsen, please consult with a healthcare provider.
                    """)
                
                # Save results option
                st.markdown("---")
                st.subheader("Save Analysis Results")
                
                if st.button("Save Results for Medical Review", type="primary", use_container_width=True):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create results dictionary
                    results = {
                        'timestamp': timestamp,
                        'prediction': int(prediction),
                        'confidence': float(confidence),
                        'risk_level': risk_level,
                        'features': combined_features,
                        'probabilities': prediction_proba.tolist(),
                        'analysis_time': elapsed_time
                    }
                    
                    # Save to session state for download
                    st.session_state.analysis_results = results
                    st.success("Results saved! You can download them below.")
                    
                    # Auto-download after save
                    st.download_button(
                        label="Download Analysis Report (JSON)",
                        data=pd.Series(results).to_json(),
                        file_name=f"als_analysis_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                progress_bar.progress(0)
                status_text.empty()
                time_display.empty()
                import traceback
                st.code(traceback.format_exc())
    
    else:
        st.info("Please record both vowel sounds (/a/ and /i/) to proceed with analysis")
        
        missing = []
        if not st.session_state.vowel_a_audio:
            missing.append("Vowel /a/")
        if not st.session_state.vowel_i_audio:
            missing.append("Vowel /i/")
        
        st.warning(f"Missing: {', '.join(missing)}")
        
        # Add visual indicator of what's missing
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.vowel_a_audio:
                st.markdown("**/a/ recording not completed**")
            else:
                st.markdown("/a/ recording completed")
        with col2:
            if not st.session_state.vowel_i_audio:
                st.markdown("**/i/ recording not completed**")
            else:
                st.markdown("/i/ recording completed")

if __name__ == "__main__":
    main()