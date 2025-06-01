#!/usr/bin/env python3
"""
Priority Station Selector
Analyzes enhanced_stations.json to create optimized collection lists
"""

import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

class PriorityStationSelector:
    """Select optimal stations for wave forecasting data collection"""
    
    def __init__(self, enhanced_stations_file: str = "enhanced_stations.json"):
        with open(enhanced_stations_file, 'r') as f:
            self.stations = json.load(f)
        
        # Regions ranked by wave forecasting importance
        self.region_priorities = {
            'west_coast_north': 1,
            'west_coast_central': 1, 
            'west_coast_south': 1,
            'hawaii': 1,
            'east_coast': 2,
            'gulf_mexico': 3,
            'great_lakes': 4,
            'other': 5
        }
    
    def score_station(self, station_id: str, station_data: Dict) -> float:
        """Calculate priority score for a station (lower = higher priority)"""
        score = 0.0
        
        # Base regional priority
        region = station_data.get('region', 'other')
        score += self.region_priorities.get(region, 5) * 10
        
        # Spectral capability (critical)
        if station_data.get('has_spectral', False):
            score -= 20  # Major boost for spectral data
        
        # Station type preferences
        station_type = station_data.get('enhanced_type', 'unknown')
        if station_type == 'spectral_buoy':
            score -= 15
        elif station_type == 'standard_buoy':
            score -= 5
        elif station_type in ['oilrig', 'fixed']:
            score += 5  # Less reliable for wave forecasting
        
        # Neighbor connectivity (better for network modeling)
        neighbor_count = station_data.get('neighbor_count', 0)
        score -= min(neighbor_count * 0.5, 5)  # Cap at 5 point bonus
        
        # Existing collection priority
        collection_priority = station_data.get('collection_priority', 5)
        score += collection_priority * 2
        
        # Owner preference (NDBC generally more reliable)
        owner = station_data.get('owner', '').lower()
        if 'ndbc' in owner:
            score -= 3
        elif 'nos' in owner:
            score -= 1
        elif 'faa' in owner or 'aviation' in owner:
            score += 3  # Oil rigs less reliable
        
        return score
    
    def get_priority_stations(self, max_stations: int = 30) -> List[Tuple[str, float, Dict]]:
        """Get prioritized list of stations"""
        scored_stations = []
        
        for station_id, station_data in self.stations.items():
            # Skip stations without coordinates
            if 'lat' not in station_data or 'lon' not in station_data:
                continue
            
            score = self.score_station(station_id, station_data)
            scored_stations.append((station_id, score, station_data))
        
        # Sort by score (lower = higher priority)
        scored_stations.sort(key=lambda x: x[1])
        
        return scored_stations[:max_stations]
    
    def get_spectral_only_stations(self, max_stations: int = 20) -> List[str]:
        """Get only stations with spectral capability"""
        spectral_stations = []
        
        for station_id, station_data in self.stations.items():
            if station_data.get('has_spectral', False):
                score = self.score_station(station_id, station_data)
                spectral_stations.append((station_id, score, station_data))
        
        spectral_stations.sort(key=lambda x: x[1])
        return [station_id for station_id, _, _ in spectral_stations[:max_stations]]
    
    def analyze_regional_coverage(self, station_list: List[str]) -> Dict:
        """Analyze regional coverage of station list"""
        regional_counts = defaultdict(int)
        regional_stations = defaultdict(list)
        
        for station_id in station_list:
            if station_id in self.stations:
                region = self.stations[station_id].get('region', 'other')
                regional_counts[region] += 1
                regional_stations[region].append(station_id)
        
        return {
            'counts': dict(regional_counts),
            'stations': dict(regional_stations),
            'total': len(station_list)
        }
    
    def generate_deployment_commands(self, priority_stations: List[str]) -> str:
        """Generate deployment commands for priority stations"""
        station_string = ','.join(priority_stations)
        
        commands = f"""
# Deploy High-Priority Data Collection
# Generated station list: {len(priority_stations)} stations

# 1. Test collection (single run)
python production_buoy_collector.py \\
  --stations "{station_string}" \\
  --data-dir data/buoy_data \\
  --log-dir logs

# 2. Production deployment (continuous hourly collection)
nohup python production_buoy_collector.py \\
  --continuous \\
  --interval 60 \\
  --max-workers 10 \\
  --stations "{station_string}" \\
  --data-dir data/buoy_data \\
  --log-dir logs \\
  > collector_output.log 2>&1 &

# 3. Save process ID for management
echo $! > collector.pid

# 4. Monitor collection
tail -f logs/collector.log

# 5. Stop collection (graceful)
kill -TERM $(cat collector.pid)
"""
        return commands
    
    def print_station_analysis(self, max_stations: int = 30):
        """Print detailed analysis of priority stations"""
        priority_stations = self.get_priority_stations(max_stations)
        spectral_only = self.get_spectral_only_stations(20)
        
        print(f"=== PRIORITY STATION ANALYSIS ===\n")
        
        print(f"Top {max_stations} Priority Stations:")
        print(f"{'Rank':<4} {'Station':<8} {'Score':<6} {'Region':<18} {'Type':<15} {'Spectral':<8} {'Name'}")
        print("-" * 90)
        
        for i, (station_id, score, data) in enumerate(priority_stations, 1):
            region = data.get('region', 'unknown')[:17]
            station_type = data.get('enhanced_type', 'unknown')[:14]
            has_spectral = "✓" if data.get('has_spectral', False) else "✗"
            name = data.get('name', 'Unknown')[:30]
            
            print(f"{i:<4} {station_id:<8} {score:<6.1f} {region:<18} {station_type:<15} {has_spectral:<8} {name}")
        
        # Regional analysis
        station_ids = [station_id for station_id, _, _ in priority_stations]
        coverage = self.analyze_regional_coverage(station_ids)
        
        print(f"\n=== REGIONAL COVERAGE ===")
        for region, count in coverage['counts'].items():
            stations = ', '.join(coverage['stations'][region][:5])
            if len(coverage['stations'][region]) > 5:
                stations += f" (+{len(coverage['stations'][region])-5} more)"
            print(f"{region:<20}: {count:<3} stations ({stations})")
        
        # Spectral-only analysis
        print(f"\n=== SPECTRAL-CAPABLE STATIONS (Top 20) ===")
        spectral_coverage = self.analyze_regional_coverage(spectral_only)
        
        for region, count in spectral_coverage['counts'].items():
            stations = ', '.join(spectral_coverage['stations'][region])
            print(f"{region:<20}: {count:<3} stations ({stations})")
        
        print(f"\n=== DEPLOYMENT RECOMMENDATIONS ===")
        print(f"• Start with SPECTRAL-ONLY stations: {len(spectral_only)} stations")
        print(f"• Full priority deployment: {len(station_ids)} stations")
        print(f"• Expected data volume: ~{len(spectral_only) * 0.15:.1f} GB/year (spectral only)")
        print(f"• Expected data volume: ~{len(station_ids) * 0.1:.1f} GB/year (full deployment)")
        
        return {
            'priority_all': station_ids,
            'spectral_only': spectral_only,
            'coverage': coverage
        }

def main():
    parser = argparse.ArgumentParser(description="Select priority stations for data collection")
    parser.add_argument("--enhanced-stations", default="enhanced_stations.json",
                       help="Enhanced stations JSON file")
    parser.add_argument("--max-stations", type=int, default=30,
                       help="Maximum number of priority stations")
    parser.add_argument("--spectral-only", action="store_true",
                       help="Show only spectral-capable stations")
    parser.add_argument("--generate-commands", action="store_true",
                       help="Generate deployment commands")
    parser.add_argument("--output-list", help="Save station list to file")
    
    args = parser.parse_args()
    
    try:
        selector = PriorityStationSelector(args.enhanced_stations)
        
        if args.spectral_only:
            stations = selector.get_spectral_only_stations(args.max_stations)
            print("Spectral-Only Priority Stations:")
            print(','.join(stations))
            
            if args.output_list:
                with open(args.output_list, 'w') as f:
                    f.write(','.join(stations))
                print(f"\nStation list saved to {args.output_list}")
        
        else:
            results = selector.print_station_analysis(args.max_stations)
            
            if args.generate_commands:
                commands = selector.generate_deployment_commands(results['spectral_only'])
                print(f"\n{commands}")
            
            if args.output_list:
                with open(args.output_list, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nDetailed results saved to {args.output_list}")
    
    except FileNotFoundError:
        print(f"Error: {args.enhanced_stations} not found")
        print("Run: python station_enhancer.py --input stations.json --output enhanced_stations.json")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()