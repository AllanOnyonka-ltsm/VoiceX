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
# Set page config
st.set_page_config(
    page_title="Voice Pathology Detection System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS - Clean, Modern, Accessible
st.markdown("""
<style>
    /* Clean typography and layout */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 2.5rem 0 1rem 0;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }

    /* Cards with subtle hover */
    .card {
        background: #fff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 5px solid transparent;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }

    .card.als { border-left-color: #e74c3c; }
    .card.parkinsons { border-left-color: #3498db; }
    .card.lung { border-left-color: #f39c12; }

    .card h3 {
        margin-top: 0;
        color: #2c3e50;
    }

    /* Stats cards */
    .stats-card {
        background: #fff;
        padding: 1.3rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid #eee;
    }
    .stats-card h2 {
        margin: 0;
        font-size: 2rem;
    }
    .stats-card h4 {
        margin: 0.5rem 0 0 0;
        color: #2c3e50;
        font-weight: 600;
    }
    .stats-card p {
        color: #7f8c8d;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Callout boxes */
    .highlight-box {
        background: #f0f7ff;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 2rem 0;
        border: 1px solid #d0e7ff;
        text-align: center;
    }
    .highlight-box h2 {
        color: #1a1a1a;
        margin-bottom: 1rem;
    }

    .benefit-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.2rem;
        margin-top: 1.5rem;
    }
    .benefit-item {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .benefit-item h3 {
        margin-top: 0;
        color: #2c3e50;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-header { font-size: 2.2rem; }
        .sub-header { font-size: 1.6rem; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    # --------------------------
    # Hero Section
    # --------------------------
    st.markdown("<h1 class='main-header'>Voice Feature Analysis for Early Pathology Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #34495e; margin-bottom: 2rem;">
        AI-powered voice analysis for early detection of neurological and respiratory diseases.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/800x200?text=Voice+Analysis+Dashboard", use_column_width=True)

    # Key Stats - Simplified
    st.markdown("""
    <div class="highlight-box">
        <div style="display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap;">
            <div>
                <h2 style="color: #27ae60; margin: 0;">85–95%</h2>
                <p style="margin: 0.25rem 0; font-weight: 500;">Detection Accuracy</p>
            </div>
            <div>
                <h2 style="color: #3498db; margin: 0;">5–10 yrs</h2>
                <p style="margin: 0.25rem 0; font-weight: 500;">Earlier Diagnosis</p>
            </div>
            <div>
                <h2 style="color: #f39c12; margin: 0;">Remote</h2>
                <p style="margin: 0.25rem 0; font-weight: 500;">Non-Invasive Screening</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # --------------------------
    # Why Early Detection Matters
    # --------------------------
    st.markdown("<h2 class='sub-header'>Why Early Detection Matters</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="benefit-grid">
        <div class="benefit-item">
            <h3>Earlier Intervention</h3>
            <p>Voice changes can appear <strong>5–10 years</strong> before clinical symptoms:</p>
            <ul>
                <li>Preventive treatments to slow progression</li>
                <li>Preserve quality of life</li>
                <li>Enable early therapeutic strategies</li>
            </ul>
        </div>
        <div class="benefit-item">
            <h3>Healthcare Cost Reduction</h3>
            <p>Early detection reduces long-term burden:</p>
            <ul>
                <li>Fewer emergency interventions</li>
                <li>Lower long-term care costs</li>
                <li>More effective outpatient management</li>
            </ul>
        </div>
        <div class="benefit-item">
            <h3>Objective Biomarkers</h3>
            <p>Voice provides measurable, repeatable data:</p>
            <ul>
                <li>Reduces subjective bias</li>
                <li>Enables precise monitoring</li>
                <li>Supports clinical decisions</li>
            </ul>
        </div>
        <div class="benefit-item">
            <h3>Global Accessibility</h3>
            <p>Democratizes screening:</p>
            <ul>
                <li>No specialized equipment needed</li>
                <li>Works in remote areas</li>
                <li>Scalable to populations</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # --------------------------
    # Global Impact
    # --------------------------
    st.markdown("<h2 class='sub-header'>Global Health Impact</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    stats = [
        ("450,000+", "ALS Cases Worldwide", "Growing by 5% annually", "#e74c3c"),
        ("10M+", "Parkinson’s Patients", "2nd most common neurological disorder", "#3498db"),
        ("3.23M", "COPD Deaths/Year", "3rd leading cause of death globally", "#f39c12"),
        ("$150B+", "Annual Healthcare Costs", "For these conditions in US alone", "#27ae60"),
    ]

    for i, (value, label, sublabel, color) in enumerate(stats):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="stats-card">
                <h2 style="color: {color};">{value}</h2>
                <h4>{label}</h4>
                <p>{sublabel}</p>
            </div>
            """, unsafe_allow_html=True)


    # --------------------------
    # Target Pathologies
    # --------------------------
    st.markdown("<h2 class='sub-header'>Target Pathologies & Voice Signs</h2>", unsafe_allow_html=True)

    pathologies = [
        {
            "class": "als",
            "title": "ALS (Lou Gehrig's Disease)",
            "changes": [
                "Progressive speech deterioration",
                "Reduced speech rate",
                "Altered speech rhythm",
                "Decreased vocal intensity"
            ],
            "features": [
                "Jitter/Shimmer: Voice instability",
                "Formant frequencies: Articulation precision",
                "Speech timing: Pause patterns",
                "Spectral analysis: Vocal tract changes"
            ],
            "note": "Voice changes appear 18–24 months before visible symptoms."
        },
        {
            "class": "parkinsons",
            "title": "Parkinson's Disease",
            "changes": [
                "Hypophonia (reduced volume)",
                "Monotone speech",
                "Vocal tremor",
                "Breathy voice quality"
            ],
            "features": [
                "F0 variations: Pitch control",
                "HNR: Voice quality",
                "Prosodic features: Rhythm & intonation",
                "Vocal dynamics: Amplitude modulation"
            ],
            "note": "Voice symptoms in 90% of patients — often the first sign."
        },
        {
            "class": "lung",
            "title": "Respiratory Diseases (COPD/Asthma)",
            "changes": [
                "Breathiness during speech",
                "Shorter sustained vowels",
                "Vocal fatigue",
                "Cough-related irritation"
            ],
            "features": [
                "Spectral energy: Frequency distribution",
                "Breathing patterns: Speech-respiration sync",
                "Voice onset time: Coordination",
                "Airflow dynamics: Respiratory support"
            ],
            "note": "Can detect airway changes before spirometry shows abnormalities."
        }
    ]

    cols = st.columns(3)
    for col, p in zip(cols, pathologies):
        with col:
            st.markdown(f"""
            <div class="card {p['class']}">
                <h3>{p['title']}</h3>
                <strong>Early Voice Changes:</strong>
                <ul>
                    {''.join(f'<li>{c}</li>' for c in p['changes'])}
                </ul>
                <strong>Key Detection Features:</strong>
                <ul>
                    {''.join(f'<li>{f}</li>' for f in p['features'])}
                </ul>
                <div style="font-size: 0.9rem; color: #555; font-style: italic;">
                    {p['note']}
                </div>
            </div>
            """, unsafe_allow_html=True)


    # --------------------------
    # Research Evidence
    # --------------------------
    st.markdown("<h2 class='sub-header'>Scientific Evidence</h2>", unsafe_allow_html=True)

    research_data = {
        'Study': ['Tsanas et al. (2012)', 'Rusz et al. (2013)', 'Godino-Llorente et al. (2006)',
                  'Verde et al. (2019)', 'Little et al. (2009)', 'Arora et al. (2014)'],
        'Pathology': ['Parkinson’s', 'Parkinson’s', 'Voice Disorders', 'ALS', 'Parkinson’s', 'Parkinson’s'],
        'Accuracy (%)': [99.5, 92.3, 94.6, 89.7, 86.8, 98.5],
        'Sample Size': [263, 84, 668, 64, 23, 50],
        'Innovation': ['Dysphonia measures', 'Prosodic analysis', 'Spectral features',
                       'Articulation timing', 'Sustained vowels', 'Mobile app detection']
    }

    df = pd.DataFrame(research_data)
    st.dataframe(df, use_container_width=True)

    fig = px.scatter(
        df,
        x='Sample Size',
        y='Accuracy (%)',
        size='Accuracy (%)',
        color='Pathology',
        hover_name='Study',
        hover_data={'Innovation': True, 'Sample Size': True},
        title='Voice Pathology Detection: Accuracy vs. Sample Size',
        labels={'Sample Size': 'Sample Size (Participants)', 'Accuracy (%)': 'Detection Accuracy (%)'},
        log_x=True,
        height=500
    )
    fig.update_layout(title_x=0, font=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)


    # --------------------------
    # Technology Overview
    # --------------------------
    st.markdown("<h2 class='sub-header'>Technology Overview</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>Voice Signal Processing</h3>
            <ol>
                <li><strong>Audio Acquisition:</strong> 16kHz+ recording</li>
                <li><strong>Preprocessing:</strong> Noise reduction, normalization</li>
                <li><strong>Feature Extraction:</strong> 200+ acoustic parameters</li>
                <li><strong>Selection:</strong> Disease-specific biomarkers</li>
                <li><strong>Classification:</strong> ML for detection</li>
                <li><strong>Reporting:</strong> Clinician-friendly insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>Machine Learning Approach</h3>
            <ul>
                <li><strong>Deep Learning:</strong> CNNs, LSTMs for temporal patterns</li>
                <li><strong>Ensemble Models:</strong> Random Forest, XGBoost</li>
                <li><strong>Feature Engineering:</strong> Domain-guided selection</li>
                <li><strong>Cross-Validation:</strong> Rigorous testing</li>
                <li><strong>Interpretability:</strong> SHAP, feature importance</li>
                <li><strong>Deployment:</strong> Real-time on edge devices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    # --------------------------
    # Call to Action
    # --------------------------
    st.markdown("<h2 class='sub-header'>Next Steps</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
        <p style="font-size: 1.2rem; margin-bottom: 1.5rem;">
            Ready to build the system? Key implementation phases:
        </p>
        <div class="benefit-grid">
            <div class="benefit-item">
                <h4>Feature Extraction</h4>
                <p>Implement advanced signal processing</p>
            </div>
            <div class="benefit-item">
                <h4>ML Model Training</h4>
                <p>Train and validate detection algorithms</p>
            </div>
            <div class="benefit-item">
                <h4>Clinical Dashboard</h4>
                <p>Design clinician-facing interface</p>
            </div>
            <div class="benefit-item">
                <h4>Compliance</h4>
                <p>Integrate HIPAA, FDA, GDPR standards</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #7f8c8d; font-size: 0.9rem;">
        © 2025 Voice Pathology Detection System | For research and clinical support
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()