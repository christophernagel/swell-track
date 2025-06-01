#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
import json
import re
import sys
from urllib.request import urlopen

def parse_station_xml(xml_source):
    """
    Parses an XML file (or URL content) and returns a dict mapping station ids to metadata.
    """
    # Determine if xml_source is a URL or a local file
    try:
        if xml_source.startswith("http://") or xml_source.startswith("https://"):
            with urlopen(xml_source) as response:
                tree = ET.parse(response)
        else:
            tree = ET.parse(xml_source)
    except Exception as e:
        sys.exit(f"Error reading XML: {e}")

    root = tree.getroot()
    stations = {}
    for station in root.findall("station"):
        station_id = station.attrib.get("id")
        if station_id:
            stations[station_id] = {
                "lat": float(station.attrib.get("lat")),
                "lon": float(station.attrib.get("lon")),
                "elev": float(station.attrib.get("elev", 0)),
                "name": station.attrib.get("name"),
                "owner": station.attrib.get("owner"),
                "type": station.attrib.get("type"),
                # You can add more fields if needed.
            }
    return stations

def extract_station_id_from_filename(filename):
    """
    Extracts the station id from a filename like '46232h2024.txt'.
    Returns the station id as a string.
    """
    match = re.match(r"(\d+)[hH]\d+", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Filename {filename} does not match expected pattern.")

def main():
    parser = argparse.ArgumentParser(
        description="Parse NDBC station XML file and output station dictionary."
    )
    parser.add_argument("xml_file", help="Path or URL to the NDBC station XML file.")
    parser.add_argument("-o", "--output", help="Output file to store the dictionary as JSON.")
    args = parser.parse_args()

    stations = parse_station_xml(args.xml_file)
    
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(stations, f, indent=2)
            print(f"Dictionary saved to {args.output}")
        except Exception as e:
            sys.exit(f"Error writing output: {e}")
    else:
        # Print the dictionary in a pretty JSON format to stdout.
        print(json.dumps(stations, indent=2))

if __name__ == "__main__":
    main()
