#!/usr/bin/env python3
"""
Station Metadata Enhancement System
Enhances existing stations.json with spatial relationships, spectral capabilities,
and prioritization for wave forecasting
"""

import json
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Set
from math import radians, degrees, sin, cos, atan2, sqrt
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StationEnhancer:
    """Enhance station metadata with spatial relationships and capabilities"""
    
    def __init__(self):
        # Priority regions for wave forecasting
        self.priority_regions = {
            'north_pacific_storm_track': {
                'bounds': {'lat': (40.0, 60.0), 'lon': (-155.0, -128.0)},
                'priority': 1,
                'description': 'North Pacific Storm Genesis & Swell Tracking'
            },
            'west_coast_north': {
                'bounds': {'lat': (40.0, 50.0), 'lon': (-130.0, -120.0)},
                'priority': 1,
                'description': 'Northern California to Oregon'
            },
            'west_coast_central': {
                'bounds': {'lat': (34.0, 40.0), 'lon': (-125.0, -115.0)},
                'priority': 1,
                'description': 'Central California'
            },
            'west_coast_south': {
                'bounds': {'lat': (30.0, 36.0), 'lon': (-125.0, -115.0)},
                'priority': 1,
                'description': 'Southern California'
            },
            'hawaii': {
                'bounds': {'lat': (18.0, 25.0), 'lon': (-165.0, -154.0)},
                'priority': 1,
                'description': 'Hawaiian Islands'
            },
            'east_coast': {
                'bounds': {'lat': (30.0, 45.0), 'lon': (-85.0, -65.0)},
                'priority': 2,
                'description': 'US East Coast'
            },
            'gulf_mexico': {
                'bounds': {'lat': (25.0, 31.0), 'lon': (-100.0, -80.0)},
                'priority': 3,
                'description': 'Gulf of Mexico'
            },
            'great_lakes': {
                'bounds': {'lat': (41.0, 49.0), 'lon': (-95.0, -75.0)},
                'priority': 4,
                'description': 'Great Lakes'
            }
        }
        
        # Known spectral-capable stations (high-priority for collection)
        self.spectral_stations = {
            # West Coast - Northern California
            '46022', '46026', '46027', '46054', '46013', '46014', '46012',
            
            # West Coast - Central California  
            '46059', '46063', '46042', '46232', '46236', '46237',
            
            # West Coast - Southern California
            '46086', '46221', '46224', '46225', '46231', '46235',
            
            # Hawaii
            '51001', '51002', '51003', '51004', '51101', '51201', '51202', '51203',
            
            # East Coast (selected)
            '44013', '44014', '44017', '44025', '44027', '44065', '41001', '41002',
            
            # Gulf of Mexico
            '42001', '42002', '42019', '42020', '42035', '42036', '42040',
            
            # Alaska (for completeness)
            '46001', '46002', '46003', '46005', '46006', '46060', '46061', '46066'
        }
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in degrees"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = atan2(y, x)
        return (degrees(bearing) + 360) % 360
    
    def find_neighbors(self, stations: Dict, max_distance: float = 500.0, max_neighbors: int = 8) -> Dict[str, Dict]:
        """Find spatial neighbors for each station"""
        enhanced_stations = {}
        
        for station_id, station in stations.items():
            if 'lat' not in station or 'lon' not in station:
                enhanced_stations[station_id] = station.copy()
                continue
                
            neighbors = []
            distances = []
            bearings = []
            
            for other_id, other_station in stations.items():
                if (other_id == station_id or 
                    'lat' not in other_station or 
                    'lon' not in other_station):
                    continue
                
                distance = self.haversine_distance(
                    station['lat'], station['lon'],
                    other_station['lat'], other_station['lon']
                )
                
                if distance <= max_distance:
                    bearing = self.calculate_bearing(
                        station['lat'], station['lon'],
                        other_station['lat'], other_station['lon']
                    )
                    
                    neighbors.append((other_id, distance, bearing))
            
            # Sort by distance and take closest neighbors
            neighbors.sort(key=lambda x: x[1])
            neighbors = neighbors[:max_neighbors]
            
            # Separate into lists
            neighbor_ids = [n[0] for n in neighbors]
            neighbor_distances = [n[1] for n in neighbors]
            neighbor_bearings = [n[2] for n in neighbors]
            
            # Create enhanced station record
            enhanced_station = station.copy()
            enhanced_station.update({
                'neighbors': neighbor_ids,
                'distances': neighbor_distances,
                'bearings': neighbor_bearings,
                'neighbor_count': len(neighbor_ids)
            })
            
            enhanced_stations[station_id] = enhanced_station
        
        return enhanced_stations
    
    def classify_regions(self, stations: Dict) -> Dict[str, Dict]:
        """Classify stations by geographic region and priority"""
        for station_id, station in stations.items():
            if 'lat' not in station or 'lon' not in station:
                continue
                
            lat, lon = station['lat'], station['lon']
            
            # Find matching region
            station['region'] = 'other'
            station['priority'] = 5
            
            for region_name, region_info in self.priority_regions.items():
                bounds = region_info['bounds']
                if (bounds['lat'][0] <= lat <= bounds['lat'][1] and
                    bounds['lon'][0] <= lon <= bounds['lon'][1]):
                    station['region'] = region_name
                    station['priority'] = region_info['priority']
                    station['region_description'] = region_info['description']
                    break
        
        return stations
    
    def add_capabilities(self, stations: Dict) -> Dict[str, Dict]:
        """Add capability flags and metadata"""
        for station_id, station in stations.items():
            # Spectral capability
            station['has_spectral'] = station_id in self.spectral_stations
            
            # Data URLs
            station['data_urls'] = {
                'realtime_txt': f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt",
                'realtime_spec': f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec",
                'station_page': f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"
            }
            
            # Enhanced type classification
            if station.get('type') == 'buoy':
                if station_id in self.spectral_stations:
                    station['enhanced_type'] = 'spectral_buoy'
                else:
                    station['enhanced_type'] = 'standard_buoy'
            else:
                station['enhanced_type'] = station.get('type', 'unknown')
            
            # Collection priority based on region and capabilities
            base_priority = station.get('priority', 5)
            if station['has_spectral']:
                base_priority = max(1, base_priority - 1)  # Boost spectral stations
            
            station['collection_priority'] = base_priority
            
        return stations
    
    def test_station_availability(self, station_id: str) -> Dict[str, bool]:
        """Test if station has active data feeds"""
        availability = {
            'txt_available': False,
            'spec_available': False,
            'last_checked': datetime.now().isoformat()
        }
        
        # Test basic data
        try:
            response = requests.get(
                f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt", 
                timeout=10
            )
            availability['txt_available'] = response.status_code == 200
        except:
            pass
        
        # Test spectral data
        try:
            response = requests.get(
                f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec", 
                timeout=10
            )
            availability['spec_available'] = response.status_code == 200
        except:
            pass
        
        return availability
    
    def enhance_stations(self, stations: Dict, test_availability: bool = False) -> Dict[str, Dict]:
        """Main enhancement pipeline"""
        logger.info(f"Enhancing {len(stations)} stations...")
        
        # Step 1: Find spatial neighbors
        logger.info("Computing spatial relationships...")
        enhanced = self.find_neighbors(stations)
        
        # Step 2: Classify by region
        logger.info("Classifying regions...")
        enhanced = self.classify_regions(enhanced)
        
        # Step 3: Add capabilities
        logger.info("Adding capability metadata...")
        enhanced = self.add_capabilities(enhanced)
        
        # Step 4: Test availability (optional)
        if test_availability:
            logger.info("Testing station availability...")
            for station_id in list(enhanced.keys())[:20]:  # Test subset
                if enhanced[station_id].get('priority', 5) <= 2:
                    availability = self.test_station_availability(station_id)
                    enhanced[station_id]['availability'] = availability
        
        return enhanced
    
    def get_priority_stations(self, enhanced_stations: Dict, max_stations: int = 50) -> List[str]:
        """Get prioritized list of stations for data collection"""
        
        # Filter and sort by priority
        candidates = []
        for station_id, station in enhanced_stations.items():
            if (station.get('has_spectral', False) and 
                station.get('collection_priority', 5) <= 3):
                candidates.append((
                    station_id,
                    station.get('collection_priority', 5),
                    station.get('neighbor_count', 0)
                ))
        
        # Sort by priority, then by neighbor count (better connected stations first)
        candidates.sort(key=lambda x: (x[1], -x[2]))
        
        return [station_id for station_id, _, _ in candidates[:max_stations]]

def main():
    parser = argparse.ArgumentParser(description="Enhance station metadata")
    parser.add_argument("--input", default="stations.json", help="Input stations file")
    parser.add_argument("--output", default="enhanced_stations.json", help="Output enhanced stations file")
    parser.add_argument("--test-availability", action="store_true", help="Test station data availability")
    parser.add_argument("--priority-list", action="store_true", help="Generate priority station list")
    
    args = parser.parse_args()
    
    # Load existing stations
    with open(args.input, 'r') as f:
        stations = json.load(f)
    
    # Enhance stations
    enhancer = StationEnhancer()
    enhanced_stations = enhancer.enhance_stations(stations, args.test_availability)
    
    # Save enhanced stations
    with open(args.output, 'w') as f:
        json.dump(enhanced_stations, f, indent=2)
    
    logger.info(f"Enhanced stations saved to {args.output}")
    
    # Generate priority list if requested
    if args.priority_list:
        priority_stations = enhancer.get_priority_stations(enhanced_stations)
        
        with open("priority_stations.json", 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_stations': len(priority_stations),
                'station_ids': priority_stations,
                'station_details': {
                    sid: enhanced_stations[sid] for sid in priority_stations
                }
            }, f, indent=2)
        
        logger.info(f"Priority list of {len(priority_stations)} stations saved to priority_stations.json")
        
        # Print summary
        print(f"\nTop 20 Priority Stations:")
        for i, station_id in enumerate(priority_stations[:20], 1):
            station = enhanced_stations[station_id]
            print(f"{i:2d}. {station_id} - {station.get('name', 'Unknown')} "
                  f"(Priority: {station.get('collection_priority', 5)}, "
                  f"Region: {station.get('region', 'unknown')})")

if __name__ == "__main__":
    main()