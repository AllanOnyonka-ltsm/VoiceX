import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Activity, AlertTriangle, MapPin, TrendingUp, Filter } from 'lucide-react';

// Mock data for geographic clusters
const mockClusters = [
  { id: 'cluster_1', lat: 19.0760, lng: 72.8777, radius: 2.5, detections: 156, highRisk: 23, intensity: 0.85 },
  { id: 'cluster_2', lat: 28.6139, lng: 77.2090, radius: 4.2, detections: 89, highRisk: 12, intensity: 0.65 },
  { id: 'cluster_3', lat: 12.9716, lng: 77.5946, radius: 3.1, detections: 234, highRisk: 45, intensity: 0.92 },
  { id: 'cluster_4', lat: 17.3850, lng: 78.4867, radius: 1.8, detections: 67, highRisk: 8, intensity: 0.45 },
  { id: 'cluster_5', lat: 22.5726, lng: 88.3639, radius: 5.5, detections: 312, highRisk: 67, intensity: 0.95 },
  { id: 'cluster_6', lat: 13.0827, lng: 80.2707, radius: 2.9, detections: 178, highRisk: 31, intensity: 0.78 },
  { id: 'cluster_7', lat: 18.5204, lng: 73.8567, radius: 3.4, detections: 145, highRisk: 19, intensity: 0.72 },
  { id: 'cluster_8', lat: 23.0225, lng: 72.5714, radius: 2.1, detections: 98, highRisk: 14, intensity: 0.58 },
];

// Heatmap points
const heatmapPoints = [
  { lat: 19.0760, lng: 72.8777, intensity: 0.9 },
  { lat: 19.0860, lng: 72.8877, intensity: 0.7 },
  { lat: 19.0660, lng: 72.8677, intensity: 0.5 },
  { lat: 28.6139, lng: 77.2090, intensity: 0.8 },
  { lat: 28.6239, lng: 77.2190, intensity: 0.6 },
  { lat: 12.9716, lng: 77.5946, intensity: 0.95 },
  { lat: 12.9816, lng: 77.6046, intensity: 0.75 },
  { lat: 12.9616, lng: 77.5846, intensity: 0.55 },
  { lat: 17.3850, lng: 78.4867, intensity: 0.5 },
  { lat: 22.5726, lng: 88.3639, intensity: 1.0 },
  { lat: 22.5826, lng: 88.3739, intensity: 0.8 },
  { lat: 22.5626, lng: 88.3539, intensity: 0.6 },
  { lat: 13.0827, lng: 80.2707, intensity: 0.85 },
  { lat: 13.0927, lng: 80.2807, intensity: 0.65 },
  { lat: 18.5204, lng: 73.8567, intensity: 0.75 },
  { lat: 23.0225, lng: 72.5714, intensity: 0.6 },
];

// Time series data
const timeSeriesData = [
  { date: 'Jan', detections: 120, highRisk: 15 },
  { date: 'Feb', detections: 145, highRisk: 22 },
  { date: 'Mar', detections: 189, highRisk: 31 },
  { date: 'Apr', detections: 234, highRisk: 45 },
  { date: 'May', detections: 278, highRisk: 52 },
  { date: 'Jun', detections: 312, highRisk: 67 },
];

interface Cluster {
  id: string;
  lat: number;
  lng: number;
  radius: number;
  detections: number;
  highRisk: number;
  intensity: number;
}

function getIntensityColor(intensity: number): string {
  if (intensity >= 0.8) return '#dc2626'; // red-600
  if (intensity >= 0.6) return '#ea580c'; // orange-600
  if (intensity >= 0.4) return '#ca8a04'; // yellow-600
  return '#16a34a'; // green-600
}

function MapController({ clusters }: { clusters: Cluster[] }) {
  const map = useMap();
  
  useEffect(() => {
    if (clusters.length > 0 && map) {
      // Fit bounds to show all clusters
      const bounds = clusters.map(c => [c.lat, c.lng] as [number, number]);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [clusters, map]);
  
  return null;
}

export default function GeoDashboard() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('30');
  const [selectedView, setSelectedView] = useState<'clusters' | 'heatmap'>('clusters');
  const [minIntensity, setMinIntensity] = useState(0);

  const filteredClusters = mockClusters.filter(c => c.intensity >= minIntensity);
  const filteredHeatmap = heatmapPoints.filter(p => p.intensity >= minIntensity);

  const totalDetections = mockClusters.reduce((sum, c) => sum + c.detections, 0);
  const totalHighRisk = mockClusters.reduce((sum, c) => sum + c.highRisk, 0);
  const activeClusters = mockClusters.length;

  return (
    <div className="w-full min-h-screen bg-[#F6F7F9] p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <MapPin className="w-6 h-6 text-[#11A300]" />
          <h1 className="text-3xl font-bold text-[#0B0D10]">Geographic Surveillance Dashboard</h1>
        </div>
        <p className="text-[#6B7280]">
          Real-time mapping of TB risk clusters and voice pathology detections across regions
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-2xl p-5 card-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-[#6B7280]">Total Detections</p>
              <p className="text-2xl font-bold text-[#0B0D10]">{totalDetections.toLocaleString()}</p>
            </div>
            <Activity className="w-8 h-8 text-[#11A300]" />
          </div>
          <p className="text-xs text-[#6B7280] mt-2">+23% from last month</p>
        </div>

        <div className="bg-white rounded-2xl p-5 card-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-[#6B7280]">High Risk Cases</p>
              <p className="text-2xl font-bold text-[#dc2626]">{totalHighRisk.toLocaleString()}</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-[#dc2626]" />
          </div>
          <p className="text-xs text-[#6B7280] mt-2">{((totalHighRisk / totalDetections) * 100).toFixed(1)}% of total</p>
        </div>

        <div className="bg-white rounded-2xl p-5 card-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-[#6B7280]">Active Clusters</p>
              <p className="text-2xl font-bold text-[#0B0D10]">{activeClusters}</p>
            </div>
            <MapPin className="w-8 h-8 text-[#11A300]" />
          </div>
          <p className="text-xs text-[#6B7280] mt-2">Across 8 regions</p>
        </div>

        <div className="bg-white rounded-2xl p-5 card-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-[#6B7280]">Coverage Area</p>
              <p className="text-2xl font-bold text-[#0B0D10]">25.4 km²</p>
            </div>
            <TrendingUp className="w-8 h-8 text-[#11A300]" />
          </div>
          <p className="text-xs text-[#6B7280] mt-2">+12% expansion</p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-2xl p-4 mb-6 card-shadow flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-[#6B7280]" />
          <span className="text-sm font-medium text-[#0B0D10]">Filters:</span>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm text-[#6B7280]">Time Range:</label>
          <select 
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-1.5 border border-gray-200 rounded-lg text-sm"
          >
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
            <option value="365">Last year</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm text-[#6B7280]">View:</label>
          <div className="flex rounded-lg border border-gray-200 overflow-hidden">
            <button
              onClick={() => setSelectedView('clusters')}
              className={`px-3 py-1.5 text-sm ${selectedView === 'clusters' ? 'bg-[#11A300] text-white' : 'bg-white text-[#0B0D10]'}`}
            >
              Clusters
            </button>
            <button
              onClick={() => setSelectedView('heatmap')}
              className={`px-3 py-1.5 text-sm ${selectedView === 'heatmap' ? 'bg-[#11A300] text-white' : 'bg-white text-[#0B0D10]'}`}
            >
              Heatmap
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm text-[#6B7280]">Min Intensity:</label>
          <input
            type="range"
            min="0"
            max="100"
            value={minIntensity * 100}
            onChange={(e) => setMinIntensity(Number(e.target.value) / 100)}
            className="w-24"
          />
          <span className="text-sm text-[#0B0D10]">{(minIntensity * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Map and Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map */}
        <div className="lg:col-span-2 bg-white rounded-2xl overflow-hidden card-shadow">
          <div className="h-[500px] w-full">
            <MapContainer
              center={[20.5937, 78.9629]}
              zoom={5}
              scrollWheelZoom={true}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <MapController clusters={filteredClusters} />
              
              {selectedView === 'clusters' && filteredClusters.map((cluster) => (
                <CircleMarker
                  key={cluster.id}
                  center={[cluster.lat, cluster.lng]}
                  radius={10 + cluster.radius * 3}
                  fillColor={getIntensityColor(cluster.intensity)}
                  color={getIntensityColor(cluster.intensity)}
                  fillOpacity={0.6}
                  weight={2}
                >
                  <Popup>
                    <div className="p-2">
                      <h3 className="font-bold text-[#0B0D10]">Cluster {cluster.id}</h3>
                      <p className="text-sm text-[#6B7280]">Detections: {cluster.detections}</p>
                      <p className="text-sm text-[#dc2626]">High Risk: {cluster.highRisk}</p>
                      <p className="text-sm text-[#6B7280]">Radius: {cluster.radius} km</p>
                      <p className="text-sm text-[#6B7280]">Intensity: {(cluster.intensity * 100).toFixed(0)}%</p>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}

              {selectedView === 'heatmap' && filteredHeatmap.map((point, idx) => (
                <CircleMarker
                  key={idx}
                  center={[point.lat, point.lng]}
                  radius={5 + point.intensity * 15}
                  fillColor={getIntensityColor(point.intensity)}
                  color={getIntensityColor(point.intensity)}
                  fillOpacity={0.4}
                  weight={1}
                />
              ))}
            </MapContainer>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Cluster List */}
          <div className="bg-white rounded-2xl p-5 card-shadow">
            <h3 className="font-bold text-[#0B0D10] mb-4 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-[#11A300]" />
              Top Clusters
            </h3>
            <div className="space-y-3 max-h-[300px] overflow-y-auto">
              {filteredClusters
                .sort((a, b) => b.intensity - a.intensity)
                .slice(0, 5)
                .map((cluster) => (
                <div key={cluster.id} className="p-3 bg-[#F6F7F9] rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm text-[#0B0D10]">{cluster.id}</span>
                    <span 
                      className="px-2 py-0.5 rounded-full text-xs font-medium"
                      style={{ 
                        backgroundColor: `${getIntensityColor(cluster.intensity)}20`,
                        color: getIntensityColor(cluster.intensity)
                      }}
                    >
                      {(cluster.intensity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-[#6B7280]">
                    <span>{cluster.detections} detections</span>
                    <span className="text-[#dc2626]">{cluster.highRisk} high risk</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Time Series */}
          <div className="bg-white rounded-2xl p-5 card-shadow">
            <h3 className="font-bold text-[#0B0D10] mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-[#11A300]" />
              Detection Trends
            </h3>
            <div className="space-y-2">
              {timeSeriesData.map((data, idx) => (
                <div key={idx} className="flex items-center gap-3">
                  <span className="text-xs text-[#6B7280] w-10">{data.date}</span>
                  <div className="flex-1 h-6 bg-[#F6F7F9] rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-[#11A300] rounded-full"
                      style={{ width: `${(data.detections / 350) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-[#0B0D10] w-10 text-right">{data.detections}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Sound Type Distribution */}
          <div className="bg-white rounded-2xl p-5 card-shadow">
            <h3 className="font-bold text-[#0B0D10] mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#11A300]" />
              Sound Types
            </h3>
            <div className="space-y-2">
              {[
                { type: 'Cough', count: 1245, color: '#11A300' },
                { type: 'Wheeze', count: 432, color: '#ca8a04' },
                { type: 'Crackles', count: 298, color: '#ea580c' },
                { type: 'Speech', count: 189, color: '#6B7280' },
              ].map((item) => (
                <div key={item.type} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-sm text-[#0B0D10]">{item.type}</span>
                  </div>
                  <span className="text-sm text-[#6B7280]">{item.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 bg-white rounded-2xl p-4 card-shadow">
        <div className="flex flex-wrap items-center gap-6">
          <span className="text-sm font-medium text-[#0B0D10]">Intensity Legend:</span>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#dc2626]" />
            <span className="text-sm text-[#6B7280]">Critical (80-100%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#ea580c]" />
            <span className="text-sm text-[#6B7280]">High (60-80%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#ca8a04]" />
            <span className="text-sm text-[#6B7280]">Moderate (40-60%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#16a34a]" />
            <span className="text-sm text-[#6B7280]">Low (0-40%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
