import pandas as pd
import numpy as np
import random
import os

# ==============================================================================
# 1. PORT PHYSICS ENGINE (CONFIGURATION)
# ==============================================================================
# SENSITIVITY LEGEND:
# - Rain: High (2.0) for Liquefaction Risk (Loading) or Monsoon (Discharge).
# - Wind: High (2.0) for Exposed Cranes (Australia/SA).
# - Fog:  High (2.0) for Navigation Safety (River/North China).
# - Mkt:  High (2.0) for Major Hubs (Queueing Theory/Congestion).

PORT_CONFIG = {
    # --- ORIGIN: BRAZIL (The "Liquefaction" Cluster) ---
    'Ponta da Madeira': {'type': 'Load', 'rain_sens': 2.0, 'wind_sens': 0.5, 'fog_sens': 0.1, 'mkt_sens': 1.2},
    'Tubarao':          {'type': 'Load', 'rain_sens': 1.8, 'wind_sens': 0.5, 'fog_sens': 0.2, 'mkt_sens': 1.0},
    'Itaguai':          {'type': 'Load', 'rain_sens': 1.8, 'wind_sens': 0.5, 'fog_sens': 0.2, 'mkt_sens': 1.0},

    # --- ORIGIN: AUSTRALIA (The "Cyclone" Cluster) ---
    'Port Hedland':     {'type': 'Load', 'rain_sens': 0.2, 'wind_sens': 2.0, 'fog_sens': 0.0, 'mkt_sens': 1.5},
    'Dampier':          {'type': 'Load', 'rain_sens': 0.2, 'wind_sens': 2.0, 'fog_sens': 0.0, 'mkt_sens': 1.2},

    # --- ORIGIN: GLOBAL OTHERS ---
    'Saldanha Bay':     {'type': 'Load', 'rain_sens': 0.3, 'wind_sens': 1.8, 'fog_sens': 0.5, 'mkt_sens': 1.0}, # SA
    'Taboneo':          {'type': 'Load', 'rain_sens': 2.5, 'wind_sens': 1.5, 'fog_sens': 0.5, 'mkt_sens': 0.8}, # Indo
    'Kamsar':           {'type': 'Load', 'rain_sens': 2.0, 'wind_sens': 1.0, 'fog_sens': 0.5, 'mkt_sens': 0.8}, # Guinea
    'Vancouver':        {'type': 'Load', 'rain_sens': 1.2, 'wind_sens': 0.8, 'fog_sens': 0.5, 'mkt_sens': 1.0}, # Canada

    # --- DESTINATION: YANGTZE RIVER (The "Fog & Draft" Cluster) ---
    'Zhangjiagang':     {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 2.0, 'wind_sens': 0.5, 'mkt_sens': 1.8},
    'Nantong':          {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 2.0, 'wind_sens': 0.5, 'mkt_sens': 1.6},
    'Nanjing':          {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 2.0, 'wind_sens': 0.5, 'mkt_sens': 1.5},
    'Jiangyin':         {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 2.0, 'wind_sens': 0.5, 'mkt_sens': 1.6},

    # --- DESTINATION: NORTH CHINA (The "Bohai Ice/Fog" Cluster) ---
    'Caofeidian':       {'type': 'Disch', 'rain_sens': 0.1, 'fog_sens': 2.0, 'wind_sens': 1.0, 'mkt_sens': 1.8},
    'Tianjin':          {'type': 'Disch', 'rain_sens': 0.1, 'fog_sens': 2.0, 'wind_sens': 1.0, 'mkt_sens': 1.6},
    'Jingtang':         {'type': 'Disch', 'rain_sens': 0.1, 'fog_sens': 2.0, 'wind_sens': 1.0, 'mkt_sens': 1.8},
    'Qinhuangdao':      {'type': 'Disch', 'rain_sens': 0.1, 'fog_sens': 1.8, 'wind_sens': 1.0, 'mkt_sens': 1.5},
    'Dalian':           {'type': 'Disch', 'rain_sens': 0.1, 'fog_sens': 1.8, 'wind_sens': 1.2, 'mkt_sens': 1.5},
    'Yingkou':          {'type': 'Disch', 'rain_sens': 0.1, 'fog_sens': 2.0, 'wind_sens': 1.2, 'mkt_sens': 1.5},

    # --- DESTINATION: EAST CHINA / HUBS (The "Congestion" Cluster) ---
    'Qingdao':          {'type': 'Disch', 'rain_sens': 0.2, 'fog_sens': 1.8, 'wind_sens': 0.8, 'mkt_sens': 2.0},
    'Rizhao':           {'type': 'Disch', 'rain_sens': 0.2, 'fog_sens': 1.5, 'wind_sens': 0.8, 'mkt_sens': 1.6},
    'Lianyungang':      {'type': 'Disch', 'rain_sens': 0.2, 'fog_sens': 1.5, 'wind_sens': 0.8, 'mkt_sens': 1.5},
    'Ningbo':           {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 1.5, 'wind_sens': 1.5, 'mkt_sens': 2.0},
    'Zhoushan':         {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 1.5, 'wind_sens': 1.5, 'mkt_sens': 1.8},

    # --- DESTINATION: SOUTH CHINA / INDIA (The "Monsoon/Typhoon" Cluster) ---
    'Fangcheng':        {'type': 'Disch', 'rain_sens': 1.0, 'fog_sens': 0.5, 'wind_sens': 2.0, 'mkt_sens': 1.2},
    'Zhanjiang':        {'type': 'Disch', 'rain_sens': 1.0, 'fog_sens': 0.5, 'wind_sens': 2.0, 'mkt_sens': 1.5},
    'Guangzhou':        {'type': 'Disch', 'rain_sens': 1.0, 'fog_sens': 0.5, 'wind_sens': 1.5, 'mkt_sens': 1.5},
    'Krishnapatnam':    {'type': 'Disch', 'rain_sens': 2.0, 'fog_sens': 0.2, 'wind_sens': 1.5, 'mkt_sens': 1.0},
    'Mangalore':        {'type': 'Disch', 'rain_sens': 2.5, 'fog_sens': 0.2, 'wind_sens': 1.5, 'mkt_sens': 0.8},
    
    # --- OTHERS ---
    'Gwangyang':        {'type': 'Disch', 'rain_sens': 0.2, 'fog_sens': 0.5, 'wind_sens': 0.5, 'mkt_sens': 0.9},
    'Teluk Rubiah':     {'type': 'Disch', 'rain_sens': 0.5, 'fog_sens': 0.5, 'wind_sens': 0.5, 'mkt_sens': 0.8},
}

# ==============================================================================
# 2. CLIMATOLOGY DATABASE (SPARSE LOOKUP)
# ==============================================================================
# Format: Month: [Rain_mm, Wind_kmh, Fog_Prob_0to1]
# We define Anchor Months (Jan & Jul) and interpolate the rest.

CLIMATE_DB = {
    # BRAZIL
    'Ponta da Madeira': {1: [300, 15, 0.0], 7: [50, 20, 0.0]}, 
    'Tubarao':          {1: [200, 15, 0.0], 7: [40, 18, 0.0]},
    'Itaguai':          {1: [220, 12, 0.0], 7: [50, 12, 0.0]},

    # AUSTRALIA
    'Port Hedland':     {1: [60, 28, 0.0], 7: [5, 15, 0.0]}, 
    'Dampier':          {1: [55, 26, 0.0], 7: [5, 14, 0.0]},

    # CHINA (NORTH - FOGGY)
    'Qingdao':          {1: [10, 22, 0.35], 7: [150, 18, 0.1]},
    'Caofeidian':       {1: [8, 24, 0.40],  7: [140, 20, 0.1]},
    'Tianjin':          {1: [5, 20, 0.35],  7: [160, 15, 0.1]},
    'Dalian':           {1: [10, 25, 0.30], 7: [150, 20, 0.2]},
    'Yingkou':          {1: [5, 22, 0.35],  7: [140, 18, 0.1]},

    # CHINA (RIVER - FOGGY WINTER)
    'Zhangjiagang':     {1: [50, 15, 0.45], 7: [180, 20, 0.1]}, 
    'Nantong':          {1: [50, 18, 0.40], 7: [170, 22, 0.1]},
    'Nanjing':          {1: [45, 12, 0.50], 7: [160, 15, 0.1]},

    # CHINA (SOUTH - TYPHOON)
    'Fangcheng':        {1: [30, 15, 0.0],  8: [400, 30, 0.0]}, 
    'Zhanjiang':        {1: [20, 15, 0.1],  8: [350, 30, 0.05]},

    # INDIA (MONSOON)
    'Mangalore':        {1: [5, 10, 0.0],   7: [900, 35, 0.0]}, 
    'Krishnapatnam':    {1: [10, 12, 0.0], 10: [300, 25, 0.0]},

    # OTHERS
    'Taboneo':          {1: [350, 15, 0.0], 8: [100, 15, 0.0]}, 
    'Kamsar':           {1: [0, 10, 0.0],   8: [500, 20, 0.0]}, 
    'Saldanha Bay':     {1: [5, 30, 0.1],   7: [80, 25, 0.2]}, 
    'Vancouver':        {1: [200, 15, 0.1], 7: [40, 10, 0.0]}, 
}

def get_climate(port, month):
    # 1. Fallback to generic if port not in DB (e.g., Jingtang uses Caofeidian logic)
    if port not in CLIMATE_DB:
        # Defaults based on region heuristics
        if port in ['Jingtang', 'Qinhuangdao', 'Huludao', 'Jinzhou']: 
            data = CLIMATE_DB['Caofeidian']
        elif port in ['Jiangyin', 'Yangzhou', 'Changzhou', 'Xinminzhou']: 
            data = CLIMATE_DB['Zhangjiagang']
        elif port in ['Rizhao', 'Lianyungang', 'Dongjiakou']: 
            data = CLIMATE_DB['Qingdao']
        elif port in ['Guangzhou', 'Qinzhou']:
            data = CLIMATE_DB['Fangcheng']
        elif port in ['Ningbo', 'Zhoushan']:
            data = {1: [50, 18, 0.2], 7: [200, 25, 0.05]} # Mix of rain/wind
        else:
            data = {1: [50, 15, 0.0], 7: [50, 15, 0.0]} # Safe default
    else:
        data = CLIMATE_DB[port]

    # 2. Interpolate
    jan = data.get(1, [50, 15, 0.0])
    # Some ports peak in Aug/Oct, handle generic 'Peak Season' logic
    peak_month = max(data.keys()) 
    peak = data[peak_month]
    
    # Distance to peak
    if month == 1: weight = 1.0
    elif month == peak_month: weight = 0.0
    else:
        # Simple linear distance
        dist = abs(month - 1)
        total_dist = abs(peak_month - 1)
        weight = 1.0 - (dist / total_dist)

    rain = jan[0] * weight + peak[0] * (1-weight)
    wind = jan[1] * weight + peak[1] * (1-weight)
    fog_prob = jan[2] * weight + peak[2] * (1-weight)
    
    return max(0, rain), max(0, wind), fog_prob

# ==============================================================================
# 3. THE SIMULATION ENGINE
# ==============================================================================

def calculate_delay(row, port, weather):
    rain, wind, is_fog = weather
    config = PORT_CONFIG[port]
    
    # --- A. MARKET INERTIA ---
    # (Unchanged)
    mkt_stress = (row['BDI_Inertia'] / 1500.0) ** 2.0
    # Cap Market Delay: Even in worst crisis, 30 days is extreme.
    delay_mkt = min(30.0, mkt_stress * config['mkt_sens'] * 0.5)
    
    # --- B. WEATHER PHYSICS (THE FIX) ---
    
    # 1. RAIN Saturation
    # Raw calculation
    raw_rain_delay = (rain / 20.0) * config['rain_sens']
    # FIX: Operational Saturation. 
    # You cannot lose more than ~15 days to rain in a month (backlog clearing takes time, but not 70 days).
    delay_rain = min(15.0, raw_rain_delay)
    
    # 2. WIND Saturation
    # Raw calculation
    if wind > 35: raw_wind = 1.0 * config['wind_sens']
    elif wind > 25: raw_wind = 0.3 * config['wind_sens']
    else: raw_wind = 0.0
    # FIX: Cap wind delay (e.g., max 7 days for prolonged storms)
    delay_wind = min(7.0, raw_wind)
        
    # 3. FOG Saturation
    # Fog is usually transient (hours), not weeks. Cap at 5 days.
    raw_fog = 1.5 * config['fog_sens'] if is_fog else 0.0
    delay_fog = min(5.0, raw_fog)
    
    # --- C. CHAOS ---
    # Black Swan is allowed to break the caps (Strike, Accident)
    black_swan = random.uniform(5.0, 10.0) if random.random() < 0.005 else 0.0
    noise = np.random.normal(0, 0.2)
    
    total = delay_mkt + delay_rain + delay_wind + delay_fog + black_swan + noise + 0.5
    return max(0.0, total)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Loading BDI Data...")
    try:
        df_bdi = pd.read_csv('././data/bdi_clean.csv')
        df_bdi['Date'] = pd.to_datetime(df_bdi['Date'])
    except FileNotFoundError:
        print("❌ Error: 'bdi_clean.csv' not found. Run clean_data.py first.")
        exit()

    # Calculate Inertia
    df_bdi['BDI_Inertia'] = df_bdi['BDI'].rolling(window=45, min_periods=1).mean()

    training_data = []
    print(f"🌊 Simulating {len(PORT_CONFIG)} Ports over {len(df_bdi)} days...")

    for index, row in df_bdi.iterrows():
        if index < 30: continue # Skip initial volatility
        
        curr_date = row['Date']
        month = curr_date.month
        
        for port in PORT_CONFIG.keys():
            # 1. Generate Environment
            avg_rain, avg_wind, fog_prob = get_climate(port, month)
            
            # Add Day-to-Day Weather Volatility
            actual_rain = avg_rain * random.uniform(0.0, 2.5) # Rain is spiky
            actual_wind = avg_wind * random.uniform(0.5, 1.5)
            is_fog = True if (random.random() < fog_prob) else False
            
            # 2. Calculate Truth
            target_delay = calculate_delay(row, port, (actual_rain, actual_wind, is_fog))
            
            # 3. Create Inputs (Forecasts with Error)
            # The model sees "Forecast", not "Actual"
            input_rain = actual_rain * random.uniform(0.8, 1.2)
            input_wind = actual_wind * random.uniform(0.8, 1.2)
            
            training_data.append({
                'Date': curr_date,
                'Port': port,
                'Month': month,
                'BDI': row['BDI'],
                'BDI_Inertia': round(row['BDI_Inertia'], 1),
                'Rain_Forecast': round(input_rain, 1),
                'Wind_Forecast': round(input_wind, 1),
                'Fog_Risk': round(fog_prob, 2),
                'Target_Delay': round(target_delay, 2)
            })

    # Save
    df_out = pd.DataFrame(training_data)
    # Save to data directory relative to project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'hybrid_training_data.csv')
    df_out.to_csv(DATA_PATH, index=False)
    print(f"✅ SUCCESS! Generated {len(df_out)} training rows.")
    print(f"Saved to: {DATA_PATH}")