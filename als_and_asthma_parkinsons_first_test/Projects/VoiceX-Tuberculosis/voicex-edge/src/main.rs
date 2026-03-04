//! VoiceX Edge CLI

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};

#[derive(Parser)]
#[command(name = "voicex")]
#[command(about = "VoiceX Edge - Offline AI Audio Triage")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Path to TB detection model
    #[arg(long, default_value = "models/tb_classifier.onnx")]
    tb_model: PathBuf,
    
    /// Path to voice pathology model
    #[arg(long, default_value = "models/voice_analyzer.onnx")]
    voice_model: PathBuf,
    
    /// Path to sound detection model
    #[arg(long, default_value = "models/sound_detector.onnx")]
    sound_model: PathBuf,
    
    /// Path to database
    #[arg(long, default_value = "data/voicex.db")]
    database: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Process audio file
    Process {
        /// Path to audio file
        audio_path: PathBuf,
        
        /// Detection mode
        #[arg(short, long, default_value = "comprehensive")]
        mode: String,
        
        /// GPS latitude
        #[arg(long)]
        lat: Option<f64>,
        
        /// GPS longitude
        #[arg(long)]
        lon: Option<f64>,
    },
    
    /// Batch process directory
    Batch {
        /// Directory containing audio files
        input_dir: PathBuf,
        
        /// Output directory for results
        #[arg(short, long, default_value = "output")]
        output: PathBuf,
    },
    
    /// Show geographic clusters
    Clusters {
        /// Time window in days
        #[arg(short, long, default_value = "30")]
        days: u32,
        
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    
    /// Export data for cloud sync
    Export {
        /// Output file path
        #[arg(short, long, default_value = "sync_payload.json")]
        output: PathBuf,
    },
    
    /// Show storage statistics
    Stats,
    
    /// Start real-time monitoring mode
    Monitor {
        /// Audio device index
        #[arg(short, long, default_value = "0")]
        device: u32,
        
        /// Detection interval in seconds
        #[arg(short, long, default_value = "5")]
        interval: u32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    // Ensure data directory exists
    if let Some(parent) = cli.database.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    match cli.command {
        Commands::Process { audio_path, mode, lat, lon } => {
            info!("Processing audio: {:?}", audio_path);
            
            // Initialize engine
            let engine = voicex_edge::VoiceXEngine::new(
                &cli.tb_model,
                &cli.voice_model,
                &cli.sound_model,
                &cli.database,
            )?;
            
            // Process audio
            let result = engine.process_audio(&audio_path)?;
            
            // Print results
            println!("\n=== Analysis Results ===");
            println!("Timestamp: {}", result.timestamp);
            println!("\nTB Risk Assessment:");
            println!("  Risk Score: {:.2}", result.tb_risk.risk_score);
            println!("  Confidence: {:.2}", result.tb_risk.confidence);
            println!("  Urgency: {:?}", result.tb_risk.urgency_tier);
            println!("  Cough Quality: {:?}", result.tb_risk.cough_quality);
            
            if !result.tb_risk.key_features.is_empty() {
                println!("  Key Features: {}", result.tb_risk.key_features.join(", "));
            }
            
            println!("\nVoice Pathology:");
            println!("  Detected: {}", result.voice_pathology.pathology_detected);
            if result.voice_pathology.pathology_detected {
                let types: Vec<String> = result.voice_pathology.pathology_types
                    .iter()
                    .map(|p| format!("{:?}", p))
                    .collect();
                println!("  Types: {}", types.join(", "));
            }
            println!("  Jitter: {:.2}%", result.voice_pathology.jitter_percent);
            println!("  Shimmer: {:.2}%", result.voice_pathology.shimmer_percent);
            println!("  HNR: {:.2} dB", result.voice_pathology.hnr_db);
            println!("  CPP: {:.2} dB", result.voice_pathology.cpp_db);
            
            println!("\nSound Events:");
            for event in &result.sound_events {
                println!("  {:?} (confidence: {:.2})", event.event_type, event.confidence);
            }
            
            if let Some(loc) = result.location {
                println!("\nLocation: {:.6}, {:.6}", loc.latitude, loc.longitude);
            }
        }
        
        Commands::Batch { input_dir, output } => {
            info!("Batch processing: {:?} -> {:?}", input_dir, output);
            
            std::fs::create_dir_all(&output)?;
            
            let engine = voicex_edge::VoiceXEngine::new(
                &cli.tb_model,
                &cli.voice_model,
                &cli.sound_model,
                &cli.database,
            )?;
            
            let entries = std::fs::read_dir(&input_dir)?;
            let mut processed = 0;
            
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                
                if path.extension().map(|e| e == "wav").unwrap_or(false) {
                    match engine.process_audio(&path) {
                        Ok(result) => {
                            let output_path = output.join(
                                path.file_stem().unwrap().to_str().unwrap().to_string() + ".json"
                            );
                            let json = serde_json::to_string_pretty(&result)?;
                            std::fs::write(&output_path, json)?;
                            println!("Processed: {:?} -> {:?}", path, output_path);
                            processed += 1;
                        }
                        Err(e) => {
                            error!("Failed to process {:?}: {}", path, e);
                        }
                    }
                }
            }
            
            println!("\nBatch complete: {} files processed", processed);
        }
        
        Commands::Clusters { days, format } => {
            info!("Fetching clusters for last {} days", days);
            
            let engine = voicex_edge::VoiceXEngine::new(
                &cli.tb_model,
                &cli.voice_model,
                &cli.sound_model,
                &cli.database,
            )?;
            
            let clusters = engine.get_geo_clusters(days)?;
            
            match format.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&clusters)?);
                }
                "table" => {
                    println!("\n{:<12} {:<15} {:<15} {:<10} {:<12} {:<20}",
                        "Cluster ID", "Latitude", "Longitude", "Radius(km)", "Detections", "High Risk");
                    println!("{}", "-".repeat(100));
                    
                    for cluster in &clusters {
                        println!("{:<12} {:<15.6} {:<15.6} {:<10.2} {:<12} {:<20}",
                            &cluster.cluster_id[..12.min(cluster.cluster_id.len())],
                            cluster.center.latitude,
                            cluster.center.longitude,
                            cluster.radius_km,
                            cluster.detection_count,
                            cluster.high_risk_count
                        );
                    }
                    
                    println!("\nTotal clusters: {}", clusters.len());
                }
                _ => {
                    error!("Unknown format: {}", format);
                }
            }
        }
        
        Commands::Export { output } => {
            info!("Exporting unsynced records to {:?}", output);
            
            let engine = voicex_edge::VoiceXEngine::new(
                &cli.tb_model,
                &cli.voice_model,
                &cli.sound_model,
                &cli.database,
            )?;
            
            let payload = engine.export_for_sync()?;
            let json = serde_json::to_string_pretty(&payload)?;
            std::fs::write(&output, json)?;
            
            println!("Exported {} records to {:?}", payload.records.len(), output);
        }
        
        Commands::Stats => {
            let storage = voicex_edge::storage::LocalStorage::new(&cli.database)?;
            let stats = storage.get_stats()?;
            
            println!("\n=== Storage Statistics ===");
            println!("Total records: {}", stats.total_records);
            println!("Unsynced records: {}", stats.unsynced_records);
            println!("High risk detections: {}", stats.high_risk_count);
        }
        
        Commands::Monitor { device, interval } => {
            info!("Starting monitor mode (device: {}, interval: {}s)", device, interval);
            println!("Monitor mode not yet implemented");
        }
    }
    
    Ok(())
}
