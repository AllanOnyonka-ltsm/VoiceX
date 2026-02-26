# VoiceX - AI-Powered Voice Pathology Detection System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

## 🎯 Overview

VoiceX is an AI-powered voice analysis system designed for early detection of neurological and respiratory pathologies through voice feature analysis. The system leverages deep learning and advanced audio processing to identify biomarkers associated with ALS, Parkinson's Disease, and respiratory conditions (Asthma/COPD).

## ✨ Features

- **Multi-Condition Detection**
  - ALS (Amyotrophic Lateral Sclerosis) detection
  - Parkinson's Disease detection
  - Respiratory condition detection (ICBHI database)
  
- **Advanced Audio Processing**
  - Real-time audio recording and analysis
  - Noise reduction and audio enhancement
  - Multiple audio format support (WAV, MP3, OGG, FLAC)
  
- **Interactive Web Interface**
  - Built with Streamlit
  - Real-time visualization of audio features
  - User-friendly interface for healthcare professionals
  - Live audio recording capability

- **Robust ML Pipeline**
  - PyTorch-based deep learning models
  - Feature extraction using librosa
  - Trained on clinical datasets (ICBHI, Parkinson's Voice Dataset)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AllanOnyonka-ltsm/VoiceX.git
   cd VoiceX
   ```

2. **Navigate to the application directory**
   ```bash
   cd "als_and_asthma_parkinsons_first_test/Projects/VoiceX-Asthma and Neuro Dectection"
   ```

3. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run Home.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
VoiceX/
├── README.md
└── als_and_asthma_parkinsons_first_test/
    └── Projects/
        ├── Reverse_Engineer_Model.ipynb    # Model development notebook
        ├── VoiceX-Asthma and Neuro Dectection/
        │   ├── Home.py                     # Main application entry
        │   ├── Requirements.txt            # Python dependencies
        │   ├── str.toml                    # Streamlit configuration
        │   ├── Models/
        │   │   └── best_model.pth          # Pre-trained model weights
        │   └── pages/
        │       ├── ALS_APP.py              # ALS detection interface
        │       ├── ICBHI.py                # Respiratory detection interface
        │       └── Parkinson_app.py        # Parkinson's detection interface
        └── solutions/
            └── python/
```

## 🔬 Technology Stack

- **Frontend**: Streamlit, Plotly, Streamlit-WebRTC
- **Audio Processing**: librosa, soundfile, noisereduce
- **Machine Learning**: PyTorch, scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly Express

## 📊 Supported Analyses

### 1. ALS Detection
Analyzes voice features to identify early indicators of Amyotrophic Lateral Sclerosis, including:
- Voice tremor patterns
- Vocal cord dysfunction markers
- Speech articulation changes

### 2. Parkinson's Disease Detection
Detects Parkinson's-related voice characteristics:
- Vocal fold bowing
- Reduced loudness
- Monotone pitch patterns
- Voice tremor

### 3. Respiratory Condition Detection (ICBHI)
Identifies respiratory abnormalities through breath sounds:
- Wheezing detection
- Crackle identification
- Abnormal breathing patterns

## 🎤 Usage

1. **Select a Detection Module**
   - Navigate to ALS, Parkinson's, or ICBHI page from the sidebar

2. **Record or Upload Audio**
   - Use the built-in recorder for real-time analysis
   - Upload pre-recorded audio files (WAV, MP3, OGG, FLAC)

3. **View Results**
   - Instant analysis with confidence scores
   - Visual representations of audio features
   - Detailed diagnostic insights

## 🛠️ Development

### Model Training
Reference the `Reverse_Engineer_Model.ipynb` notebook for model development and training procedures.

### Adding New Features
1. Create a new page in the `pages/` directory
2. Follow the existing page structure
3. Update the main `Home.py` if needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

**This system is intended for research and educational purposes only.** It should not be used as a sole diagnostic tool. Always consult with qualified healthcare professionals for medical diagnosis and treatment.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Allan Onyonka (@AllanOnyonka-ltsm)

## 🙏 Acknowledgments

- ICBHI 2017 Challenge Dataset
- Parkinson's Voice Dataset contributors
- Open-source community for the amazing tools and libraries

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: February 2026
