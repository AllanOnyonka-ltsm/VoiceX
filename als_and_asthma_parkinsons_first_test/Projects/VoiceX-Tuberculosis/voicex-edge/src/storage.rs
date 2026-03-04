//! Local storage using SQLite

use crate::types::*;
use rusqlite::{Connection, params, OptionalExtension};
use std::path::Path;
use tracing::{debug, info, warn};

/// Local SQLite storage for detections
pub struct LocalStorage {
    conn: Connection,
}

impl LocalStorage {
    /// Initialize database with schema
    pub fn new(db_path: &Path) -> Result<Self, VoiceXError> {
        info!("Opening database at {:?}", db_path);
        
        let conn = Connection::open(db_path)
            .map_err(|e| VoiceXError::StorageError(format!("Failed to open DB: {}", e)))?;
        
        let storage = Self { conn };
        storage.init_schema()?;
        
        info!("Database initialized successfully");
        Ok(storage)
    }
    
    fn init_schema(&self) -> Result<(), VoiceXError> {
        // Main analysis results table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS analysis_results (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                accuracy_meters REAL,
                tb_risk_score REAL NOT NULL,
                tb_confidence REAL NOT NULL,
                urgency_tier TEXT NOT NULL,
                cough_quality TEXT,
                pathology_detected INTEGER NOT NULL,
                jitter_percent REAL,
                shimmer_percent REAL,
                hnr_db REAL,
                cpp_db REAL,
                f0_mean REAL,
                f0_std REAL,
                audio_duration_secs REAL NOT NULL,
                sample_rate INTEGER NOT NULL,
                synced INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        // Sound events table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sound_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                start_time_ms INTEGER,
                end_time_ms INTEGER,
                FOREIGN KEY (record_id) REFERENCES analysis_results(record_id)
            )",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        // Key features table (for explainability)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS key_features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                FOREIGN KEY (record_id) REFERENCES analysis_results(record_id)
            )",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        // Pathology types table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pathology_types (
                pathology_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                pathology_type TEXT NOT NULL,
                FOREIGN KEY (record_id) REFERENCES analysis_results(record_id)
            )",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        // Indexes for efficient queries
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_results(timestamp)",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_location ON analysis_results(latitude, longitude)",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synced ON analysis_results(synced)",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_urgency ON analysis_results(urgency_tier)",
            [],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Store analysis result
    pub fn store_analysis(&self, result: &AnalysisResult) -> Result<(), VoiceXError> {
        let record_id = format!("rec_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));
        
        // Insert main record
        self.conn.execute(
            "INSERT INTO analysis_results (
                record_id, timestamp, latitude, longitude, accuracy_meters,
                tb_risk_score, tb_confidence, urgency_tier, cough_quality,
                pathology_detected, jitter_percent, shimmer_percent, hnr_db, cpp_db,
                f0_mean, f0_std, audio_duration_secs, sample_rate
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                record_id,
                result.timestamp.to_rfc3339(),
                result.location.as_ref().map(|l| l.latitude),
                result.location.as_ref().map(|l| l.longitude),
                result.location.as_ref().and_then(|l| l.accuracy_meters),
                result.tb_risk.risk_score,
                result.tb_risk.confidence,
                format!("{:?}", result.tb_risk.urgency_tier),
                format!("{:?}", result.tb_risk.cough_quality),
                result.voice_pathology.pathology_detected as i32,
                result.voice_pathology.jitter_percent,
                result.voice_pathology.shimmer_percent,
                result.voice_pathology.hnr_db,
                result.voice_pathology.cpp_db,
                result.voice_pathology.f0_mean,
                result.voice_pathology.f0_std,
                result.audio_metadata.duration_secs,
                result.audio_metadata.sample_rate,
            ],
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        // Insert sound events
        for event in &result.sound_events {
            self.conn.execute(
                "INSERT INTO sound_events (record_id, event_type, confidence, start_time_ms, end_time_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    record_id,
                    format!("{:?}", event.event_type),
                    event.confidence,
                    event.start_time_ms,
                    event.end_time_ms,
                ],
            ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        }
        
        // Insert key features
        for feature in &result.tb_risk.key_features {
            self.conn.execute(
                "INSERT INTO key_features (record_id, feature_name) VALUES (?1, ?2)",
                params![record_id, feature],
            ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        }
        
        // Insert pathology types
        for pathology in &result.voice_pathology.pathology_types {
            self.conn.execute(
                "INSERT INTO pathology_types (record_id, pathology_type) VALUES (?1, ?2)",
                params![record_id, format!("{:?}", pathology)],
            ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        }
        
        debug!("Stored analysis result: {}", record_id);
        Ok(())
    }
    
    /// Get analysis by ID
    pub fn get_analysis(&self, record_id: &str) -> Result<Option<AnalysisResult>, VoiceXError> {
        // This would reconstruct the full AnalysisResult from joined tables
        // Simplified for now
        todo!("Implement full record retrieval")
    }
    
    /// Get records by time range
    pub fn get_by_time_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<DetectionPoint>, VoiceXError> {
        let mut stmt = self.conn.prepare(
            "SELECT record_id, timestamp, latitude, longitude, tb_risk_score, urgency_tier
             FROM analysis_results
             WHERE timestamp >= ?1 AND timestamp <= ?2
             ORDER BY timestamp"
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        let rows = stmt.query_map(
            params![start.to_rfc3339(), end.to_rfc3339()],
            |row| {
                let timestamp_str: String = row.get(1)?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());
                
                Ok(DetectionPoint {
                    id: row.get(0)?,
                    location: GeoLocation {
                        latitude: row.get(2)?,
                        longitude: row.get(3)?,
                        accuracy_meters: None,
                        altitude: None,
                    },
                    timestamp,
                    urgency_tier: parse_urgency_tier(&row.get::<_, String>(5)?),
                    sound_types: Vec::new(), // Would need separate query
                    tb_risk_score: row.get(4)?,
                })
            },
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        let mut points = Vec::new();
        for row in rows {
            points.push(row.map_err(|e| VoiceXError::StorageError(e.to_string()))?);
        }
        
        Ok(points)
    }
    
    /// Get geographic clusters
    pub fn get_clusters_by_timeframe(&self, days: u32) -> Result<Vec<GeoCluster>, VoiceXError> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days as i64);
        
        let points = self.get_by_time_range(cutoff, chrono::Utc::now())?;
        
        // Run clustering
        let clustering = crate::geo::GeoClustering::new(5.0, 3); // 5km radius, min 3 points
        Ok(clustering.cluster_detections(&points))
    }
    
    /// Export unsynced records
    pub fn export_unsynced_records(&self) -> Result<SyncPayload, VoiceXError> {
        let mut stmt = self.conn.prepare(
            "SELECT record_id, timestamp, latitude, longitude, tb_risk_score, urgency_tier
             FROM analysis_results
             WHERE synced = 0
             LIMIT 1000"
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        let rows = stmt.query_map([], |row| {
            let timestamp_str: String = row.get(1)?;
            let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            
            Ok(AnonymizedRecord {
                record_id: row.get(0)?,
                timestamp,
                location: GeoLocation {
                    latitude: row.get(2)?,
                    longitude: row.get(3)?,
                    accuracy_meters: None,
                    altitude: None,
                },
                tb_risk_score: row.get(4)?,
                urgency_tier: parse_urgency_tier(&row.get::<_, String>(5)?),
                sound_event_types: Vec::new(),
                audio_features_hash: String::new(), // Would compute from features
            })
        }).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        let mut records = Vec::new();
        for row in rows {
            records.push(row.map_err(|e| VoiceXError::StorageError(e.to_string()))?);
        }
        
        Ok(SyncPayload {
            device_id: get_device_id(),
            export_timestamp: chrono::Utc::now(),
            records,
        })
    }
    
    /// Mark records as synced
    pub fn mark_records_synced(&self, record_ids: &[String]) -> Result<(), VoiceXError> {
        for id in record_ids {
            self.conn.execute(
                "UPDATE analysis_results SET synced = 1 WHERE record_id = ?1",
                params![id],
            ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        }
        
        debug!("Marked {} records as synced", record_ids.len());
        Ok(())
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> Result<StorageStats, VoiceXError> {
        let total_records: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM analysis_results",
            [],
            |row| row.get(0),
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        let unsynced_records: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM analysis_results WHERE synced = 0",
            [],
            |row| row.get(0),
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        let high_risk_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM analysis_results WHERE urgency_tier IN ('High', 'Critical')",
            [],
            |row| row.get(0),
        ).map_err(|e| VoiceXError::StorageError(e.to_string()))?;
        
        Ok(StorageStats {
            total_records: total_records as u64,
            unsynced_records: unsynced_records as u64,
            high_risk_count: high_risk_count as u64,
        })
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_records: u64,
    pub unsynced_records: u64,
    pub high_risk_count: u64,
}

fn parse_urgency_tier(s: &str) -> UrgencyTier {
    match s {
        "Low" => UrgencyTier::Low,
        "Moderate" => UrgencyTier::Moderate,
        "High" => UrgencyTier::High,
        "Critical" => UrgencyTier::Critical,
        _ => UrgencyTier::Low,
    }
}

fn get_device_id() -> String {
    // In production, this would be a persistent device identifier
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("device_{}", SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs())
}

// Re-export DetectionPoint from geo module
use crate::geo::DetectionPoint;
