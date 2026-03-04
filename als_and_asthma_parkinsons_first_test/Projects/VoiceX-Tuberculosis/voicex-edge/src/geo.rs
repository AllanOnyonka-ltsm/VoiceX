//! Geographic tracking and clustering

use crate::types::*;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Geographic tracker for detection locations
pub struct GeoTracker {
    last_known_location: Option<GeoLocation>,
    location_history: Vec<(GeoLocation, chrono::DateTime<chrono::Utc>)>,
}

impl GeoTracker {
    pub fn new() -> Result<Self, VoiceXError> {
        Ok(Self {
            last_known_location: None,
            location_history: Vec::new(),
        })
    }
    
    /// Get current location (from GPS or cached)
    pub fn get_current_location(&self) -> Result<GeoLocation, VoiceXError> {
        // In production, this would interface with device GPS
        // For now, return last known or error
        self.last_known_location
            .ok_or_else(|| VoiceXError::GeoError("No location available".to_string()))
    }
    
    /// Update location from GPS
    pub fn update_location(&mut self, location: GeoLocation) {
        self.last_known_location = Some(location);
        self.location_history.push((location, chrono::Utc::now()));
        
        // Keep only last 1000 locations
        if self.location_history.len() > 1000 {
            self.location_history.remove(0);
        }
    }
    
    /// Update from GPS coordinates
    pub fn update_from_coords(&mut self, lat: f64, lon: f64, accuracy: Option<f64>) {
        self.update_location(GeoLocation {
            latitude: lat,
            longitude: lon,
            accuracy_meters: accuracy,
            altitude: None,
        });
    }
    
    /// Get location history
    pub fn get_history(&self) -> &[(GeoLocation, chrono::DateTime<chrono::Utc>)] {
        &self.location_history
    }
}

/// Geographic clustering for detection hotspots
pub struct GeoClustering {
    eps_km: f64,      // Maximum distance for points in same cluster
    min_points: u32,  // Minimum points to form cluster
}

impl GeoClustering {
    pub fn new(eps_km: f64, min_points: u32) -> Self {
        Self { eps_km, min_points }
    }
    
    /// Perform DBSCAN clustering on detection locations
    pub fn cluster_detections(
        &self,
        detections: &[DetectionPoint],
    ) -> Vec<GeoCluster> {
        let mut visited = vec![false; detections.len()];
        let mut clusters: Vec<GeoCluster> = Vec::new();
        let mut noise = Vec::new();
        
        for i in 0..detections.len() {
            if visited[i] {
                continue;
            }
            
            visited[i] = true;
            let neighbors = self.get_neighbors(detections, i);
            
            if neighbors.len() < self.min_points as usize {
                noise.push(i);
            } else {
                let cluster = self.expand_cluster(detections, i, neighbors, &mut visited);
                clusters.push(cluster);
            }
        }
        
        clusters
    }
    
    fn get_neighbors(&self, detections: &[DetectionPoint], point_idx: usize) -> Vec<usize> {
        let point = &detections[point_idx];
        
        detections
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != point_idx)
            .filter(|(_, d)| point.location.distance_to(&d.location) <= self.eps_km)
            .map(|(i, _)| i)
            .collect()
    }
    
    fn expand_cluster(
        &self,
        detections: &[DetectionPoint],
        core_idx: usize,
        mut neighbors: Vec<usize>,
        visited: &mut [bool],
    ) -> GeoCluster {
        let mut cluster_points = vec![core_idx];
        
        let mut i = 0;
        while i < neighbors.len() {
            let neighbor_idx = neighbors[i];
            
            if !visited[neighbor_idx] {
                visited[neighbor_idx] = true;
                let neighbor_neighbors = self.get_neighbors(detections, neighbor_idx);
                
                if neighbor_neighbors.len() >= self.min_points as usize {
                    neighbors.extend(neighbor_neighbors);
                }
            }
            
            if !cluster_points.contains(&neighbor_idx) {
                cluster_points.push(neighbor_idx);
            }
            
            i += 1;
        }
        
        self.create_cluster(detections, &cluster_points)
    }
    
    fn create_cluster(
        &self,
        detections: &[DetectionPoint],
        point_indices: &[usize],
    ) -> GeoCluster {
        let points: Vec<&DetectionPoint> = point_indices
            .iter()
            .map(|&i| &detections[i])
            .collect();
        
        // Calculate centroid
        let center = self.calculate_centroid(&points);
        
        // Calculate radius (max distance from center)
        let radius_km = points
            .iter()
            .map(|p| center.distance_to(&p.location))
            .fold(0.0f64, f64::max);
        
        // Count high-risk detections
        let high_risk_count = points
            .iter()
            .filter(|p| matches!(p.urgency_tier, UrgencyTier::High | UrgencyTier::Critical))
            .count() as u32;
        
        // Build risk distribution
        let mut risk_distribution: HashMap<UrgencyTier, u32> = HashMap::new();
        for point in &points {
            *risk_distribution.entry(point.urgency_tier).or_insert(0) += 1;
        }
        
        // Build sound type distribution
        let mut sound_type_distribution: HashMap<SoundEventType, u32> = HashMap::new();
        for point in &points {
            for sound_type in &point.sound_types {
                *sound_type_distribution.entry(*sound_type).or_insert(0) += 1;
            }
        }
        
        // Get time range
        let timestamps: Vec<_> = points.iter().map(|p| p.timestamp).collect();
        let first_detection = timestamps.iter().min().copied().unwrap_or_else(chrono::Utc::now);
        let last_detection = timestamps.iter().max().copied().unwrap_or_else(chrono::Utc::now);
        
        GeoCluster {
            cluster_id: format!("cluster_{}", uuid::Uuid::new_v4().to_string()[..8].to_string()),
            center,
            radius_km,
            detection_count: points.len() as u32,
            high_risk_count,
            first_detection,
            last_detection,
            risk_distribution,
            sound_type_distribution,
        }
    }
    
    fn calculate_centroid(&self, points: &[&DetectionPoint]) -> GeoLocation {
        let n = points.len() as f64;
        
        let avg_lat = points.iter().map(|p| p.location.latitude).sum::<f64>() / n;
        let avg_lon = points.iter().map(|p| p.location.longitude).sum::<f64>() / n;
        
        GeoLocation {
            latitude: avg_lat,
            longitude: avg_lon,
            accuracy_meters: None,
            altitude: None,
        }
    }
}

/// Detection point for clustering
#[derive(Debug, Clone)]
pub struct DetectionPoint {
    pub id: String,
    pub location: GeoLocation,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub urgency_tier: UrgencyTier,
    pub sound_types: Vec<SoundEventType>,
    pub tb_risk_score: f32,
}

/// Geographic grid for efficient spatial queries
pub struct GeoGrid {
    cell_size_km: f64,
    cells: HashMap<(i64, i64), Vec<DetectionPoint>>,
}

impl GeoGrid {
    pub fn new(cell_size_km: f64) -> Self {
        Self {
            cell_size_km,
            cells: HashMap::new(),
        }
    }
    
    /// Insert detection into grid
    pub fn insert(&mut self, detection: DetectionPoint) {
        let cell = self.location_to_cell(&detection.location);
        self.cells.entry(cell).or_default().push(detection);
    }
    
    /// Query detections within radius
    pub fn query_radius(
        &self,
        center: &GeoLocation,
        radius_km: f64,
    ) -> Vec<&DetectionPoint> {
        let center_cell = self.location_to_cell(center);
        let cell_radius = (radius_km / self.cell_size_km).ceil() as i64;
        
        let mut results = Vec::new();
        
        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                let cell = (center_cell.0 + dx, center_cell.1 + dy);
                
                if let Some(detections) = self.cells.get(&cell) {
                    for detection in detections {
                        if center.distance_to(&detection.location) <= radius_km {
                            results.push(detection);
                        }
                    }
                }
            }
        }
        
        results
    }
    
    /// Get density heatmap data
    pub fn get_density_map(&self) -> Vec<(GeoLocation, u32)> {
        self.cells
            .iter()
            .map(|(cell, detections)| {
                let center = self.cell_to_location(*cell);
                (center, detections.len() as u32)
            })
            .collect()
    }
    
    fn location_to_cell(&self, loc: &GeoLocation) -> (i64, i64) {
        // Approximate conversion: 1 degree lat ~ 111 km
        let x = (loc.longitude / (self.cell_size_km / 111.0)).floor() as i64;
        let y = (loc.latitude / (self.cell_size_km / 111.0)).floor() as i64;
        (x, y)
    }
    
    fn cell_to_location(&self, cell: (i64, i64)) -> GeoLocation {
        GeoLocation {
            latitude: cell.1 as f64 * (self.cell_size_km / 111.0),
            longitude: cell.0 as f64 * (self.cell_size_km / 111.0),
            accuracy_meters: None,
            altitude: None,
        }
    }
}

/// Heatmap generator for visualization
pub struct HeatmapGenerator {
    resolution_km: f64,
}

impl HeatmapGenerator {
    pub fn new(resolution_km: f64) -> Self {
        Self { resolution_km }
    }
    
    /// Generate heatmap points with intensity
    pub fn generate_heatmap(
        &self,
        detections: &[DetectionPoint],
    ) -> Vec<HeatmapPoint> {
        let mut grid: HashMap<(i64, i64), Vec<&DetectionPoint>> = HashMap::new();
        
        // Bin detections into grid cells
        for detection in detections {
            let cell = self.location_to_cell(&detection.location);
            grid.entry(cell).or_default().push(detection);
        }
        
        // Calculate intensity for each cell
        grid.iter()
            .map(|(cell, points)| {
                let center = self.cell_to_location(*cell);
                let count = points.len() as f32;
                
                // Weight by risk
                let risk_weight: f32 = points
                    .iter()
                    .map(|p| match p.urgency_tier {
                        UrgencyTier::Low => 0.25,
                        UrgencyTier::Moderate => 0.5,
                        UrgencyTier::High => 0.75,
                        UrgencyTier::Critical => 1.0,
                    })
                    .sum();
                
                let intensity = (count * 0.3 + risk_weight * 0.7).min(1.0);
                
                HeatmapPoint {
                    location: center,
                    intensity,
                    count: points.len() as u32,
                    risk_score: points.iter().map(|p| p.tb_risk_score).sum::<f32>() / points.len() as f32,
                }
            })
            .collect()
    }
    
    fn location_to_cell(&self, loc: &GeoLocation) -> (i64, i64) {
        let x = (loc.longitude / (self.resolution_km / 111.0)).floor() as i64;
        let y = (loc.latitude / (self.resolution_km / 111.0)).floor() as i64;
        (x, y)
    }
    
    fn cell_to_location(&self, cell: (i64, i64)) -> GeoLocation {
        GeoLocation {
            latitude: cell.1 as f64 * (self.resolution_km / 111.0),
            longitude: cell.0 as f64 * (self.resolution_km / 111.0),
            accuracy_meters: None,
            altitude: None,
        }
    }
}

/// Heatmap point for visualization
#[derive(Debug, Clone)]
pub struct HeatmapPoint {
    pub location: GeoLocation,
    pub intensity: f32,  // 0.0 - 1.0
    pub count: u32,
    pub risk_score: f32,
}

// UUID placeholder for cluster IDs
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> String {
            use std::time::{SystemTime, UNIX_EPOCH};
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{:x}", timestamp)
        }
    }
}
