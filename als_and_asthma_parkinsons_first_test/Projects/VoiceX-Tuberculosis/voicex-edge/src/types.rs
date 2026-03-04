//! Core types for VoiceX Edge

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main error type for VoiceX operations
#[derive(Debug, thiserror::Error)]
pub enum VoiceXError {
    #[error("Audio processing error: {0}")]
    AudioError(String),
    
    #[error("ML inference error: {0}")]
    InferenceError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Geolocation error: {0}")]
    GeoError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Geographic coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub accuracy_meters: Option<f64>,
    pub altitude: Option<f64>,
}

impl GeoLocation {
    /// Calculate distance to another point in kilometers (Haversine formula)
    pub fn distance_to(&self, other: &GeoLocation) -> f64 {
        const R: f64 = 6371.0; // Earth's radius in km
        
        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let delta_lat = (other.latitude - self.latitude).to_radians();
        let delta_lon = (other.longitude - self.longitude).to_radians();
        
        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        
        R * c
    }
}

/// Audio metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub duration_secs: f32,
    pub sample_rate: u32,
    pub channels: u16,
}

/// TB Risk classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TBRiskResult {
    pub risk_score: f32,           // 0.0 - 1.0
    pub confidence: f32,           // Model confidence
    pub urgency_tier: UrgencyTier, // Triage priority
    pub cough_quality: CoughQuality,
    pub key_features: Vec<String>, // Explainable features
}

impl TBRiskResult {
    pub fn is_high_risk(&self) -> bool {
        self.risk_score > 0.7 && self.confidence > 0.6
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UrgencyTier {
    Low,      // Score < 0.3
    Moderate, // Score 0.3 - 0.6
    High,     // Score 0.6 - 0.8
    Critical, // Score > 0.8
}

impl UrgencyTier {
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s < 0.3 => UrgencyTier::Low,
            s if s < 0.6 => UrgencyTier::Moderate,
            s if s < 0.8 => UrgencyTier::High,
            _ => UrgencyTier::Critical,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CoughQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Invalid,
}

/// Voice pathology analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAnalysisResult {
    pub pathology_detected: bool,
    pub pathology_types: Vec<PathologyType>,
    pub confidence: f32,
    
    // Clinical metrics
    pub jitter_percent: f32,      // Frequency perturbation
    pub shimmer_percent: f32,     // Amplitude perturbation
    pub hnr_db: f32,              // Harmonics-to-Noise Ratio
    pub cpp_db: f32,              // Cepstral Peak Prominence
    pub f0_mean: f32,             // Fundamental frequency mean
    pub f0_std: f32,              // Fundamental frequency std dev
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PathologyType {
    Dysphonia,
    Hoarseness,
    Breathiness,
    Roughness,
    Asthenia,
    Strain,
    Tremor,
}

/// Sound event detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundEvent {
    pub event_type: SoundEventType,
    pub confidence: f32,
    pub start_time_ms: u32,
    pub end_time_ms: u32,
    pub frequency_range: (f32, f32), // Hz
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SoundEventType {
    // Respiratory sounds
    Cough,
    Wheeze,
    Crackles,
    Stridor,
    
    // Voice sounds
    Speech,
    SustainedVowel,
    
    // Environmental (for noise filtering)
    BackgroundNoise,
    SpeechOverlap,
    Music,
    Traffic,
    
    // Other medical sounds
    Sneeze,
    ThroatClear,
}

/// Comprehensive analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub timestamp: DateTime<Utc>,
    pub tb_risk: TBRiskResult,
    pub voice_pathology: VoiceAnalysisResult,
    pub sound_events: Vec<SoundEvent>,
    pub location: Option<GeoLocation>,
    pub audio_metadata: AudioMetadata,
}

/// Detection mode for targeted analysis
#[derive(Debug, Clone, Copy)]
pub enum DetectionMode {
    TBScreening,
    VoicePathology,
    SoundEvent(SoundEventType),
    Comprehensive,
}

/// Targeted analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetedResult {
    TB(TBRiskResult),
    Voice(VoiceAnalysisResult),
    Sound(Vec<SoundEvent>),
    Comprehensive {
        tb: TBRiskResult,
        voice: VoiceAnalysisResult,
        sounds: Vec<SoundEvent>,
    },
}

/// Geographic cluster of detections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoCluster {
    pub cluster_id: String,
    pub center: GeoLocation,
    pub radius_km: f64,
    pub detection_count: u32,
    pub high_risk_count: u32,
    pub first_detection: DateTime<Utc>,
    pub last_detection: DateTime<Utc>,
    pub risk_distribution: HashMap<UrgencyTier, u32>,
    pub sound_type_distribution: HashMap<SoundEventType, u32>,
}

/// Sync payload for cloud upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPayload {
    pub device_id: String,
    pub export_timestamp: DateTime<Utc>,
    pub records: Vec<AnonymizedRecord>,
}

/// Anonymized record for cloud sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizedRecord {
    pub record_id: String,
    pub timestamp: DateTime<Utc>,
    pub location: GeoLocation,
    pub tb_risk_score: f32,
    pub urgency_tier: UrgencyTier,
    pub pathology_detected: bool,
    pub sound_event_types: Vec<SoundEventType>,
    pub audio_features_hash: String, // For deduplication
}

/// Audio feature vector for ML inference
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub spectrogram: ndarray::Array2<f32>,
    pub mfccs: ndarray::Array2<f32>,
    pub mel_spectrogram: ndarray::Array2<f32>,
    pub chromagram: ndarray::Array2<f32>,
    pub zero_crossing_rate: Vec<f32>,
    pub spectral_centroid: Vec<f32>,
    pub spectral_rolloff: Vec<f32>,
    pub rms_energy: Vec<f32>,
    pub duration_samples: usize,
    pub sample_rate: u32,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub sample_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
    pub n_mels: usize,
    pub n_mfcc: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 224,
            output_size: 2,
            sample_rate: 16000,
            hop_length: 512,
            n_fft: 2048,
            n_mels: 128,
            n_mfcc: 13,
        }
    }
}
