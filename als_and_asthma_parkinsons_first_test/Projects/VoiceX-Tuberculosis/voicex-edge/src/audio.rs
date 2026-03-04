//! Audio preprocessing and feature extraction

use crate::types::{AudioFeatures, VoiceXError, AudioMetadata};
use hound::{WavReader, WavSpec};
use ndarray::{Array1, Array2, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use tracing::{debug, trace};

/// Audio data container
pub struct AudioData {
    samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioData {
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32 / self.channels as f32
    }
    
    pub fn as_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }
        
        // Mix down to mono
        self.samples
            .chunks(self.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / self.channels as f32)
            .collect()
    }
}

/// Audio processor for preprocessing and feature extraction
pub struct AudioProcessor {
    config: crate::types::ModelConfig,
    fft_planner: FftPlanner<f32>,
}

impl AudioProcessor {
    pub fn new() -> Result<Self, VoiceXError> {
        Ok(Self {
            config: crate::types::ModelConfig::default(),
            fft_planner: FftPlanner::new(),
        })
    }
    
    /// Load audio from file (WAV format)
    pub fn load_audio(&self, path: &std::path::Path) -> Result<AudioData, VoiceXError> {
        debug!("Loading audio from {:?}", path);
        
        let reader = WavReader::open(path)
            .map_err(|e| VoiceXError::AudioError(format!("Failed to open WAV: {}", e)))?;
        
        let spec = reader.spec();
        
        // Convert to f32 samples
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1i32 << (bits - 1)) as f32;
                reader.into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
        };
        
        let audio = AudioData {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        };
        
        debug!("Loaded audio: {}s, {}Hz, {}ch", audio.duration(), audio.sample_rate, audio.channels);
        Ok(audio)
    }
    
    /// Load audio from raw buffer (for WASM)
    pub fn load_from_buffer(&self, samples: Vec<f32>, sample_rate: u32) -> Result<AudioData, VoiceXError> {
        Ok(AudioData {
            samples,
            sample_rate,
            channels: 1,
        })
    }
    
    /// Preprocess audio: normalize, denoise, segment
    pub fn preprocess(&self, audio: &AudioData) -> Result<AudioData, VoiceXError> {
        let mut mono = audio.as_mono();
        
        // Peak normalization
        let max_amp = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if max_amp > 0.0 {
            mono.iter_mut().for_each(|s| *s /= max_amp);
        }
        
        // Simple noise gate
        let noise_floor = 0.01;
        mono.iter_mut().for_each(|s| {
            if s.abs() < noise_floor {
                *s = 0.0;
            }
        });
        
        // Energy-based segmentation (trim silence)
        let window_size = (audio.sample_rate as f32 * 0.025) as usize; // 25ms windows
        let hop_size = (audio.sample_rate as f32 * 0.010) as usize;    // 10ms hop
        
        let energies: Vec<f32> = mono
            .windows(window_size)
            .step_by(hop_size)
            .map(|w| w.iter().map(|s| s * s).sum::<f32>().sqrt())
            .collect();
        
        let energy_threshold = energies.iter().sum::<f32>() / energies.len() as f32 * 0.1;
        
        // Find start and end of speech
        let start_idx = energies
            .iter()
            .position(|&e| e > energy_threshold)
            .unwrap_or(0) * hop_size;
        
        let end_idx = energies
            .iter()
            .rposition(|&e| e > energy_threshold)
            .map(|i| (i + 1) * hop_size)
            .unwrap_or(mono.len())
            .min(mono.len());
        
        let trimmed = mono[start_idx..end_idx].to_vec();
        
        debug!("Preprocessed: {} samples -> {} samples", mono.len(), trimmed.len());
        
        Ok(AudioData {
            samples: trimmed,
            sample_rate: audio.sample_rate,
            channels: 1,
        })
    }
    
    /// Extract comprehensive audio features
    pub fn extract_features(&self, audio: &AudioData) -> Result<AudioFeatures, VoiceXError> {
        let preprocessed = self.preprocess(audio)?;
        let samples = preprocessed.samples;
        let sr = preprocessed.sample_rate as usize;
        
        // Compute STFT
        let spectrogram = self.compute_stft(&samples)?;
        
        // Compute mel spectrogram
        let mel_spectrogram = self.compute_mel_spectrogram(&spectrogram, sr)?;
        
        // Compute MFCCs
        let mfccs = self.compute_mfccs(&mel_spectrogram)?;
        
        // Compute chromagram
        let chromagram = self.compute_chromagram(&spectrogram, sr)?;
        
        // Compute temporal features
        let zero_crossing_rate = self.compute_zcr(&samples);
        let spectral_centroid = self.compute_spectral_centroid(&spectrogram, sr);
        let spectral_rolloff = self.compute_spectral_rolloff(&spectrogram, sr);
        let rms_energy = self.compute_rms(&samples);
        
        Ok(AudioFeatures {
            spectrogram,
            mfccs,
            mel_spectrogram,
            chromagram,
            zero_crossing_rate,
            spectral_centroid,
            spectral_rolloff,
            rms_energy,
            duration_samples: samples.len(),
            sample_rate: sr as u32,
        })
    }
    
    /// Compute Short-Time Fourier Transform
    fn compute_stft(&self, samples: &[f32]) -> Result<Array2<f32>, VoiceXError> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        
        let fft = self.fft_planner.plan_fft_forward(n_fft);
        
        let num_frames = (samples.len() - n_fft) / hop_length + 1;
        let mut spectrogram = Array2::zeros((n_fft / 2 + 1, num_frames));
        
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let x = 2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
                0.54 - 0.46 * x.cos() // Hamming window
            })
            .collect();
        
        for (frame_idx, frame) in samples.windows(n_fft).step_by(hop_length).enumerate() {
            if frame_idx >= num_frames {
                break;
            }
            
            // Apply window and create complex buffer
            let mut buffer: Vec<Complex<f32>> = frame
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            
            // Pad to n_fft if needed
            buffer.resize(n_fft, Complex::new(0.0, 0.0));
            
            // Perform FFT
            fft.process(&mut buffer);
            
            // Store magnitude spectrum
            for (bin_idx, complex) in buffer.iter().take(n_fft / 2 + 1).enumerate() {
                spectrogram[[bin_idx, frame_idx]] = complex.norm();
            }
        }
        
        Ok(spectrogram)
    }
    
    /// Compute mel spectrogram from linear spectrogram
    fn compute_mel_spectrogram(
        &self,
        spectrogram: &Array2<f32>,
        sample_rate: usize,
    ) -> Result<Array2<f32>, VoiceXError> {
        let n_mels = self.config.n_mels;
        let n_fft = self.config.n_fft;
        
        // Create mel filterbank
        let mel_fb = self.create_mel_filterbank(sample_rate, n_fft, n_mels);
        
        // Apply filterbank
        let mel_spec = mel_fb.dot(spectrogram);
        
        // Convert to dB
        let mel_spec_db = mel_spec.map(|&x| {
            let x = x.max(1e-10);
            10.0 * x.log10()
        });
        
        Ok(mel_spec_db)
    }
    
    /// Create mel filterbank
    fn create_mel_filterbank(
        &self,
        sample_rate: usize,
        n_fft: usize,
        n_mels: usize,
    ) -> Array2<f32> {
        let f_min = 0.0f32;
        let f_max = sample_rate as f32 / 2.0;
        
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);
        
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        
        let freq_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let fft_freqs: Vec<f32> = (0..=n_fft / 2)
            .map(|i| sample_rate as f32 * i as f32 / n_fft as f32)
            .collect();
        
        let mut filterbank = Array2::zeros((n_mels, n_fft / 2 + 1));
        
        for (mel_idx, freqs) in freq_points.windows(3).enumerate() {
            let f_left = freqs[0];
            let f_center = freqs[1];
            let f_right = freqs[2];
            
            for (bin_idx, &fft_freq) in fft_freqs.iter().enumerate() {
                let weight = if fft_freq >= f_left && fft_freq <= f_center {
                    (fft_freq - f_left) / (f_center - f_left)
                } else if fft_freq > f_center && fft_freq <= f_right {
                    (f_right - fft_freq) / (f_right - f_center)
                } else {
                    0.0
                };
                filterbank[[mel_idx, bin_idx]] = weight;
            }
        }
        
        filterbank
    }
    
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10f32.powf(mel / 2595.0) - 1.0)
    }
    
    /// Compute MFCCs from mel spectrogram
    fn compute_mfccs(&self, mel_spectrogram: &Array2<f32>) -> Result<Array2<f32>, VoiceXError> {
        let n_mfcc = self.config.n_mfcc;
        
        // Simple DCT-based MFCC (simplified)
        let (n_mels, n_frames) = mel_spectrogram.dim();
        let mut mfccs = Array2::zeros((n_mfcc, n_frames));
        
        for frame in 0..n_frames {
            for coeff in 0..n_mfcc {
                let sum: f32 = (0..n_mels)
                    .map(|mel| {
                        mel_spectrogram[[mel, frame]]
                            * ((std::f32::consts::PI * coeff as f32 * (2 * mel + 1) as f32)
                                / (2 * n_mels) as f32)
                                .cos()
                    })
                    .sum();
                mfccs[[coeff, frame]] = sum;
            }
        }
        
        Ok(mfccs)
    }
    
    /// Compute chromagram (pitch class profile)
    fn compute_chromagram(
        &self,
        spectrogram: &Array2<f32>,
        sample_rate: usize,
    ) -> Result<Array2<f32>, VoiceXError> {
        let n_chroma = 12; // 12 pitch classes
        let (n_bins, n_frames) = spectrogram.dim();
        
        let mut chromagram = Array2::zeros((n_chroma, n_frames));
        
        for frame in 0..n_frames {
            for bin in 0..n_bins {
                let freq = bin as f32 * sample_rate as f32 / (2 * n_bins) as f32;
                if freq > 0.0 {
                    let midi_note = 69.0 + 12.0 * (freq / 440.0).log2();
                    let pitch_class = ((midi_note.round() as i32).rem_euclid(12)) as usize;
                    chromagram[[pitch_class, frame]] += spectrogram[[bin, frame]];
                }
            }
        }
        
        // Normalize
        for frame in 0..n_frames {
            let sum: f32 = chromagram.column(frame).sum();
            if sum > 0.0 {
                for pc in 0..n_chroma {
                    chromagram[[pc, frame]] /= sum;
                }
            }
        }
        
        Ok(chromagram)
    }
    
    /// Compute zero crossing rate
    fn compute_zcr(&self, samples: &[f32]) -> Vec<f32> {
        let window_size = self.config.hop_length;
        
        samples
            .windows(window_size)
            .step_by(window_size)
            .map(|w| {
                w.windows(2)
                    .filter(|pair| pair[0].signum() != pair[1].signum())
                    .count() as f32
                    / window_size as f32
            })
            .collect()
    }
    
    /// Compute spectral centroid
    fn compute_spectral_centroid(&self, spectrogram: &Array2<f32>, sample_rate: usize) -> Vec<f32> {
        let (n_bins, n_frames) = spectrogram.dim();
        let bin_freqs: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * sample_rate as f32 / (2 * n_bins) as f32)
            .collect();
        
        (0..n_frames)
            .map(|frame| {
                let sum: f32 = spectrogram.column(frame).sum();
                if sum > 0.0 {
                    spectrogram
                        .column(frame)
                        .iter()
                        .zip(bin_freqs.iter())
                        .map(|(&mag, &freq)| mag * freq)
                        .sum::<f32>()
                        / sum
                } else {
                    0.0
                }
            })
            .collect()
    }
    
    /// Compute spectral rolloff
    fn compute_spectral_rolloff(&self, spectrogram: &Array2<f32>, sample_rate: usize) -> Vec<f32> {
        let (n_bins, n_frames) = spectrogram.dim();
        let bin_freqs: Vec<f32> = (0..n_bins)
            .map(|i| i as f32 * sample_rate as f32 / (2 * n_bins) as f32)
            .collect();
        
        (0..n_frames)
            .map(|frame| {
                let total: f32 = spectrogram.column(frame).sum();
                let threshold = total * 0.85;
                
                let mut cumsum = 0.0;
                for (bin, (&mag, &freq)) in spectrogram
                    .column(frame)
                    .iter()
                    .zip(bin_freqs.iter())
                    .enumerate()
                {
                    cumsum += mag;
                    if cumsum >= threshold {
                        return freq;
                    }
                }
                bin_freqs.last().copied().unwrap_or(0.0)
            })
            .collect()
    }
    
    /// Compute RMS energy
    fn compute_rms(&self, samples: &[f32]) -> Vec<f32> {
        let window_size = self.config.hop_length;
        
        samples
            .windows(window_size)
            .step_by(window_size)
            .map(|w| {
                (w.iter().map(|&s| s * s).sum::<f32>() / window_size as f32).sqrt()
            })
            .collect()
    }
}
