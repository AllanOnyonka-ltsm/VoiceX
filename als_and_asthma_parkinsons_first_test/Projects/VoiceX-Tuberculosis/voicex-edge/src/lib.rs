//! VoiceX Edge - Offline AI Audio Triage System
//! 
//! A Rust-based edge computing solution for:
//! - TB risk detection from cough audio
//! - Voice pathology analysis
//! - General sound event detection
//! - Geographic clustering of detections

pub mod audio;
pub mod ml;
pub mod geo;
pub mod storage;
pub mod types;

use std::path::Path;
use tracing::{info, debug, warn};

pub use types::*;

/// Main VoiceX Edge engine
pub struct VoiceXEngine {
    audio_processor: audio::AudioProcessor,
    tb_classifier: ml::TBClassifier,
    voice_analyzer: ml::VoiceAnalyzer,
    sound_detector: ml::SoundDetector,
    geo_tracker: geo::GeoTracker,
    storage: storage::LocalStorage,
}

impl VoiceXEngine {
    /// Initialize the VoiceX engine with model paths
    pub fn new(
        tb_model_path: &Path,
        voice_model_path: &Path,
        sound_model_path: &Path,
        db_path: &Path,
    ) -> Result<Self, VoiceXError> {
        info!("Initializing VoiceX Edge engine...");
        
        let audio_processor = audio::AudioProcessor::new()?;
        let tb_classifier = ml::TBClassifier::new(tb_model_path)?;
        let voice_analyzer = ml::VoiceAnalyzer::new(voice_model_path)?;
        let sound_detector = ml::SoundDetector::new(sound_model_path)?;
        let geo_tracker = geo::GeoTracker::new()?;
        let storage = storage::LocalStorage::new(db_path)?;
        
        info!("VoiceX Edge engine initialized successfully");
        
        Ok(Self {
            audio_processor,
            tb_classifier,
            voice_analyzer,
            sound_detector,
            geo_tracker,
            storage,
        })
    }
    
    /// Process audio file and return comprehensive analysis
    pub fn process_audio(&self, audio_path: &Path) -> Result<AnalysisResult, VoiceXError> {
        debug!("Processing audio: {:?}", audio_path);
        
        // Load and preprocess audio
        let audio_data = self.audio_processor.load_audio(audio_path)?;
        let features = self.audio_processor.extract_features(&audio_data)?;
        
        // Run ML inferences
        let tb_result = self.tb_classifier.predict(&features)?;
        let voice_result = self.voice_analyzer.analyze(&features)?;
        let sound_events = self.sound_detector.detect(&features)?;
        
        // Get current location
        let location = self.geo_tracker.get_current_location().ok();
        
        let result = AnalysisResult {
            timestamp: chrono::Utc::now(),
            tb_risk: tb_result,
            voice_pathology: voice_result,
            sound_events,
            location,
            audio_metadata: AudioMetadata {
                duration_secs: audio_data.duration(),
                sample_rate: audio_data.sample_rate,
                channels: audio_data.channels,
            },
        };
        
        // Store result locally
        self.storage.store_analysis(&result)?;
        
        Ok(result)
    }
    
    /// Process audio with specific detection modes
    pub fn process_audio_targeted(
        &self,
        audio_path: &Path,
        mode: DetectionMode,
    ) -> Result<TargetedResult, VoiceXError> {
        let audio_data = self.audio_processor.load_audio(audio_path)?;
        let features = self.audio_processor.extract_features(&audio_data)?;
        
        let result = match mode {
            DetectionMode::TBScreening => {
                TargetedResult::TB(self.tb_classifier.predict(&features)?)
            }
            DetectionMode::VoicePathology => {
                TargetedResult::Voice(self.voice_analyzer.analyze(&features)?)
            }
            DetectionMode::SoundEvent(event_type) => {
                let events = self.sound_detector.detect_by_type(&features, event_type)?;
                TargetedResult::Sound(events)
            }
            DetectionMode::Comprehensive => {
                let tb = self.tb_classifier.predict(&features)?;
                let voice = self.voice_analyzer.analyze(&features)?;
                let sounds = self.sound_detector.detect(&features)?;
                TargetedResult::Comprehensive { tb, voice, sounds }
            }
        };
        
        Ok(result)
    }
    
    /// Get geographic cluster analysis
    pub fn get_geo_clusters(&self, days: u32) -> Result<Vec<GeoCluster>, VoiceXError> {
        self.storage.get_clusters_by_timeframe(days)
    }
    
    /// Export data for sync to cloud
    pub fn export_for_sync(&self) -> Result<SyncPayload, VoiceXError> {
        self.storage.export_unsynced_records()
    }
    
    /// Mark records as synced
    pub fn mark_synced(&self, record_ids: &[String]) -> Result<(), VoiceXError> {
        self.storage.mark_records_synced(record_ids)
    }
}

/// WASM-compatible entry point
#[cfg(feature = "wasm")]
pub mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;
    
    #[wasm_bindgen]
    pub struct VoiceXWasm {
        engine: VoiceXEngine,
    }
    
    #[wasm_bindgen]
    impl VoiceXWasm {
        #[wasm_bindgen(constructor)]
        pub fn new(
            tb_model_bytes: &[u8],
            voice_model_bytes: &[u8],
            sound_model_bytes: &[u8],
        ) -> Result<VoiceXWasm, JsValue> {
            // Initialize engine with in-memory models for WASM
            todo!("WASM initialization")
        }
        
        #[wasm_bindgen]
        pub fn process_audio_buffer(&self, audio_buffer: &[f32]) -> Result<JsValue, JsValue> {
            todo!("Process audio buffer in WASM")
        }
    }
}
