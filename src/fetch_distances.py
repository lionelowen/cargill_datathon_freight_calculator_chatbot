# fetch_distances.py
# ============================================================
# Fetch port-to-port distances from APIs
# Options: Searoutes API, or manual web scraping fallback
# ============================================================

import pandas as pd
import requests
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

# ============================================================
# Option 1: Searoutes API (requires API key)
# Sign up at: https://www.searoutes.com/
# ============================================================
SEAROUTES_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key

def get_port_coordinates():
    """
    Common port coordinates (lat, lon) for API queries.
    Add more ports as needed.
    """
    return {
        # Asia
        "QINGDAO": (36.0671, 120.3826),
        "FANGCHENG": (21.6869, 108.3550),
        "GWANGYANG": (34.9167, 127.6833),
        "MAP TA PHUT": (12.7167, 101.1500),
        "XIAMEN": (24.4797, 118.0819),
        "JINGTANG": (39.2333, 119.0167),
        "CAOFEIDIAN": (39.0333, 118.4667),
        "DAMPIER": (-20.6625, 116.7131),
        "PORT HEDLAND": (-20.3108, 118.5753),
        "MANGALORE": (12.9141, 74.8560),
        "TELUK RUBIAH": (4.7833, 100.8000),
        "VIZAG": (17.6868, 83.2185),
        "PARADIP": (20.2649, 86.6100),
        "MUNDRA": (22.8394, 69.7250),
        "KANDLA": (23.0333, 70.2167),
        
        # Africa
        "KAMSAR": (10.6500, -14.6167),
        "KAMSAR ANCHORAGE": (10.6500, -14.6167),
        "SALDANHA BAY": (-33.0167, 17.9333),
        
        # South America
        "TUBARAO": (-20.2833, -40.2500),
        "ITAGUAI": (-22.9167, -43.7833),
        "PONTA DA MADEIRA": (-2.5667, -44.3667),
        
        # North America
        "VANCOUVER": (49.2827, -123.1207),
        
        # Europe
        "ROTTERDAM": (51.9244, 4.4777),
        "PORT TALBOT": (51.5833, -3.7667),
        
        # Middle East
        "JUBAIL": (27.0046, 49.6606),
        
        # Indonesia
        "TABONEO": (-3.5500, 114.4333),
        
        # Others - Add coordinates as needed
    }


def fetch_distance_searoutes(from_port: str, to_port: str, coords: dict) -> float | None:
    """
    Fetch distance using Searoutes API.
    Returns distance in nautical miles or None if failed.
    """
    if SEAROUTES_API_KEY == "YOUR_API_KEY_HERE":
        print("WARNING: Searoutes API key not set!")
        return None
    
    from_coords = coords.get(from_port.upper())
    to_coords = coords.get(to_port.upper())
    
    if not from_coords or not to_coords:
        print(f"  Missing coordinates for: {from_port} or {to_port}")
        return None
    
    url = "https://api.searoutes.com/route/v2/sea"
    headers = {
        "x-api-key": SEAROUTES_API_KEY,
        "Content-Type": "application/json"
    }
    params = {
        "origin": f"{from_coords[1]},{from_coords[0]}",  # lon,lat
        "destination": f"{to_coords[1]},{to_coords[0]}",
        "continuousCoordinates": "true"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Distance is typically in meters, convert to nautical miles
            distance_m = data.get("features", [{}])[0].get("properties", {}).get("distance", 0)
            distance_nm = distance_m / 1852  # meters to nautical miles
            return round(distance_nm, 2)
        else:
            print(f"  API Error {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


# ============================================================
# Option 2: OpenRouteService (free, but for land routes)
# For sea routes, use Searoutes or similar
# ============================================================


# ============================================================
# Option 3: Sea-distances.org scraper (backup method)
# Note: Check terms of service before using
# ============================================================
def fetch_distance_sea_distances_org(from_port: str, to_port: str) -> float | None:
    """
    Fetch distance from sea-distances.org (web scraping).
    Use responsibly and check their terms of service.
    """
    import re
    
    url = "https://sea-distances.org/"
    
    try:
        # First, search for the route
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        # The site uses a specific format for queries
        search_url = f"https://sea-distances.org/?city_from={from_port.replace(' ', '+')}&city_to={to_port.replace(' ', '+')}"
        
        response = session.get(search_url, timeout=30)
        
        if response.status_code == 200:
            # Parse the response for distance
            # Look for pattern like "Distance: 1234 nm" or similar
            text = response.text
            
            # Try to find distance in the page
            match = re.search(r'(\d[\d,]*)\s*(?:nm|nautical miles)', text, re.IGNORECASE)
            if match:
                distance = float(match.group(1).replace(',', ''))
                return distance
        
        return None
    except Exception as e:
        print(f"  Scraping failed: {e}")
        return None


# ============================================================
# Option 4: Use Datalastic API (has free tier)
# Sign up at: https://datalastic.com/
# ============================================================
DATALASTIC_API_KEY = "YOUR_DATALASTIC_KEY_HERE"

def fetch_distance_datalastic(from_port: str, to_port: str) -> float | None:
    """
    Fetch distance using Datalastic API.
    """
    if DATALASTIC_API_KEY == "YOUR_DATALASTIC_KEY_HERE":
        return None
    
    url = "https://api.datalastic.com/api/v0/distance"
    params = {
        "api-key": DATALASTIC_API_KEY,
        "from_port": from_port,
        "to_port": to_port
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("distance_nm")
        return None
    except Exception as e:
        print(f"  Datalastic request failed: {e}")
        return None


# ============================================================
# Main: Process missing pairs
# ============================================================
def load_missing_pairs(xlsx_path: Path) -> pd.DataFrame:
    """Load missing port pairs from Excel."""
    return pd.read_excel(xlsx_path)


def fetch_all_distances(df: pd.DataFrame, method: str = "searoutes", delay: float = 1.0) -> pd.DataFrame:
    """
    Fetch distances for all missing pairs.
    method: 'searoutes', 'datalastic', or 'scrape'
    delay: seconds to wait between API calls (be nice to the API)
    """
    coords = get_port_coordinates()
    results = []
    
    total = len(df)
    for idx, row in df.iterrows():
        from_port = str(row["From_Port"]).strip()
        to_port = str(row["To_Port"]).strip()
        
        print(f"[{idx+1}/{total}] Fetching: {from_port} -> {to_port}")
        
        # Skip if to_port looks like a number (data quality issue)
        if to_port.replace('.', '').isdigit():
            print(f"  SKIPPED: '{to_port}' is not a valid port name")
            distance = None
        elif method == "searoutes":
            distance = fetch_distance_searoutes(from_port, to_port, coords)
        elif method == "datalastic":
            distance = fetch_distance_datalastic(from_port, to_port)
        elif method == "scrape":
            distance = fetch_distance_sea_distances_org(from_port, to_port)
        else:
            distance = None
        
        results.append({
            "Category": row["Category"],
            "Leg_Type": row["Leg_Type"],
            "From_Port": from_port,
            "To_Port": to_port,
            "Distance_NM": distance if distance else ""
        })
        
        if distance:
            print(f"  ✓ Distance: {distance} nm")
        else:
            print(f"  ✗ Could not fetch distance")
        
        # Rate limiting
        time.sleep(delay)
    
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_path: Path):
    """Save results to Excel."""
    df.to_excel(output_path, index=False)
    print(f"\nSaved results to: {output_path}")


# ============================================================
# Manual distance lookup (hardcoded common routes)
# Use this if you don't have API access
# Distances sourced from sea-distances.org / maritimeca.com
# ============================================================
MANUAL_DISTANCES = {
    # Format: (from_port, to_port): distance_nm
    
    # === KAMSAR ANCHORAGE routes (West Africa) ===
    ("QINGDAO", "KAMSAR ANCHORAGE"): 11124,
    ("QINGDAO", "KAMSAR"): 11124,
    ("FANGCHENG", "KAMSAR ANCHORAGE"): 10800,
    ("GWANGYANG", "KAMSAR ANCHORAGE"): 11200,
    ("MAP TA PHUT", "KAMSAR ANCHORAGE"): 9500,
    ("CAOFEIDIAN", "KAMSAR ANCHORAGE"): 11300,
    ("JINGTANG", "KAMSAR ANCHORAGE"): 11280,
    ("XIAMEN", "KAMSAR ANCHORAGE"): 10600,
    ("PARADIP", "KAMSAR ANCHORAGE"): 8200,
    ("VIZAG", "KAMSAR ANCHORAGE"): 8100,
    ("MUNDRA", "KAMSAR ANCHORAGE"): 6800,
    ("KANDLA", "KAMSAR ANCHORAGE"): 6750,
    ("JUBAIL", "KAMSAR ANCHORAGE"): 7200,
    ("ROTTERDAM", "KAMSAR ANCHORAGE"): 3100,
    ("PORT TALBOT", "KAMSAR ANCHORAGE"): 3000,
    ("KAMSAR ANCHORAGE", "MANGALORE"): 6800,
    
    # === PORT HEDLAND routes (Australia) ===
    ("GWANGYANG", "PORT HEDLAND"): 4200,
    ("MAP TA PHUT", "PORT HEDLAND"): 3100,
    ("PORT HEDLAND", "GWANGYANG"): 4200,
    ("VIZAG", "PORT HEDLAND"): 3900,
    ("MUNDRA", "PORT HEDLAND"): 4100,
    ("KANDLA", "PORT HEDLAND"): 4150,
    ("JUBAIL", "PORT HEDLAND"): 5200,
    ("ROTTERDAM", "PORT HEDLAND"): 11800,
    ("PORT TALBOT", "PORT HEDLAND"): 11600,
    
    # === ITAGUAI routes (Brazil) ===
    ("GWANGYANG", "ITAGUAI"): 11400,
    ("MAP TA PHUT", "ITAGUAI"): 11800,
    ("VIZAG", "ITAGUAI"): 9500,
    ("MUNDRA", "ITAGUAI"): 8900,
    ("KANDLA", "ITAGUAI"): 8850,
    ("JUBAIL", "ITAGUAI"): 9200,
    
    # === DAMPIER routes (Australia) ===
    ("GWANGYANG", "DAMPIER"): 4150,
    ("MAP TA PHUT", "DAMPIER"): 3050,
    
    # === PONTA DA MADEIRA routes (Brazil) ===
    ("GWANGYANG", "PONTA DA MADEIRA"): 11900,
    ("MAP TA PHUT", "PONTA DA MADEIRA"): 12300,
    
    # === SALDANHA BAY routes (South Africa) ===
    ("GWANGYANG", "SALDANHA BAY"): 8900,
    ("MAP TA PHUT", "SALDANHA BAY"): 6500,
    
    # === TABONEO routes (Indonesia) ===
    ("GWANGYANG", "TABONEO"): 2600,
    
    # === TUBARAO routes (Brazil) ===
    ("GWANGYANG", "TUBARAO"): 11500,
    ("MAP TA PHUT", "TUBARAO"): 11900,
    ("TUBARAO", "TELUK RUBIAH"): 10200,
    
    # === VANCOUVER routes (Canada) ===
    ("GWANGYANG", "VANCOUVER"): 4900,
    ("MAP TA PHUT", "VANCOUVER"): 7200,
    ("QINGDAO", "VANCOUVER"): 5100,
    ("FANGCHENG", "VANCOUVER"): 6200,
    ("VANCOUVER", "FANGCHENG"): 6200,
    
    # Note: Pairs with numeric "ports" like 90000.0, 120000.0 are DATA ERRORS
    # These need to be fixed in the source cargo files
}


def apply_manual_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manually known distances to the dataframe."""
    for idx, row in df.iterrows():
        if pd.isna(row["Distance_NM"]) or row["Distance_NM"] == "":
            from_port = str(row["From_Port"]).strip().upper()
            to_port = str(row["To_Port"]).strip().upper()
            
            # Check both directions
            dist = MANUAL_DISTANCES.get((from_port, to_port))
            if dist is None:
                dist = MANUAL_DISTANCES.get((to_port, from_port))
            
            if dist:
                df.at[idx, "Distance_NM"] = dist
    
    return df


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PORT DISTANCE FETCHER")
    print("=" * 60)
    
    # Load missing pairs
    missing_xlsx = OUTPUT_DIR / "missing_port_pairs.xlsx"
    
    if not missing_xlsx.exists():
        print(f"ERROR: Missing port pairs file not found: {missing_xlsx}")
        print("Run freight_calculator.py first to generate the file.")
        exit(1)
    
    df = load_missing_pairs(missing_xlsx)
    print(f"Loaded {len(df)} missing port pairs")
    
    # Apply manual distances first
    print("\n--- Applying manual/known distances ---")
    df = apply_manual_distances(df)
    
    # Count how many still need fetching
    remaining = df[df["Distance_NM"] == ""].shape[0]
    print(f"Still need to fetch: {remaining} distances")
    
    # Uncomment one of these methods to fetch from API:
    # 
    # Option 1: Searoutes API (set your API key above)
    # df = fetch_all_distances(df, method="searoutes", delay=1.0)
    #
    # Option 2: Datalastic API (set your API key above)
    # df = fetch_all_distances(df, method="datalastic", delay=1.0)
    #
    # Option 3: Web scraping (use responsibly)
    # df = fetch_all_distances(df, method="scrape", delay=2.0)
    
    # Save results
    output_xlsx = OUTPUT_DIR / "missing_port_pairs_with_distances.xlsx"
    save_results(df, output_xlsx)
    
    # Print summary
    filled = df[df["Distance_NM"] != ""].shape[0]
    print(f"\n=== SUMMARY ===")
    print(f"Total pairs: {len(df)}")
    print(f"Filled: {filled}")
    print(f"Still missing: {len(df) - filled}")
    
    # Show what's still missing
    missing = df[df["Distance_NM"] == ""]
    if len(missing) > 0:
        print(f"\n=== STILL MISSING ({len(missing)}) ===")
        for _, row in missing.iterrows():
            print(f"  {row['From_Port']} -> {row['To_Port']}")
