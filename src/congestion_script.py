import pandas as pd
import numpy as np
import joblib
import os

# Climate database
CLIMATE_DB = {
    # BRAZIL
    'PONTA DA MADEIRA': {1: [300, 15, 0.0], 7: [50, 20, 0.0]}, 
    'TUBARAO':          {1: [200, 15, 0.0], 7: [40, 18, 0.0]},
    'ITAGUAI':          {1: [220, 12, 0.0], 7: [50, 12, 0.0]},

    # AUSTRALIA
    'PORT HEDLAND':     {1: [60, 28, 0.0], 7: [5, 15, 0.0]}, 
    'DAMPIER':          {1: [55, 26, 0.0], 7: [5, 14, 0.0]},

    # CHINA (NORTH - FOGGY)
    'QINGDAO':          {1: [10, 22, 0.35], 7: [150, 18, 0.1]},
    'CAOFEIDIAN':       {1: [8, 24, 0.40],  7: [140, 20, 0.1]},
    'TIANJIN':          {1: [5, 20, 0.35],  7: [160, 15, 0.1]},
    'DALIAN':           {1: [10, 25, 0.30], 7: [150, 20, 0.2]},
    'YINGKOU':          {1: [5, 22, 0.35],  7: [140, 18, 0.1]},

    # CHINA (RIVER - FOGGY WINTER)
    'ZHANGJIAGANG':     {1: [50, 15, 0.45], 7: [180, 20, 0.1]}, 
    'NANTONG':          {1: [50, 18, 0.40], 7: [170, 22, 0.1]},
    'NANJING':          {1: [45, 12, 0.50], 7: [160, 15, 0.1]},

    # CHINA (SOUTH - TYPHOON)
    'FANGCHENG':        {1: [30, 15, 0.0],  8: [400, 30, 0.0]}, 
    'ZHANJIANG':        {1: [20, 15, 0.1],  8: [350, 30, 0.05]},

    # INDIA (MONSOON)
    'MANGALORE':        {1: [5, 10, 0.0],   7: [900, 35, 0.0]}, 
    'KRISHNAPATNAM':    {1: [10, 12, 0.0], 10: [300, 25, 0.0]},

    # OTHERS
    'TABONEO':          {1: [350, 15, 0.0], 8: [100, 15, 0.0]}, 
    'KAMSAR':           {1: [0, 10, 0.0],   8: [500, 20, 0.0]}, 
    'SALDANHA BAY':     {1: [5, 30, 0.1],   7: [80, 25, 0.2]}, 
    'VANCOUVER':        {1: [200, 15, 0.1], 7: [40, 10, 0.0]}, 
}

def predict_congestion(target_date, target_bdi, port_name, scenario_type, weather_profile=None):
    """
    Standalone function to predict congestion.
    Requires: 'models/congestion_model.pkl' and 'data/hybrid_training_data.csv'
    """
    # --- A. SETUP PATHS ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    # Assuming standard folder structure
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'ml_port_congestion','models', 'congestion_model.pkl')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'hybrid_training_data.csv')

    # Load Resources
    try:
        model = joblib.load(MODEL_PATH)
        df_train = pd.read_csv(DATA_PATH)
        df_train['Date'] = pd.to_datetime(df_train['Date'])
    except FileNotFoundError as e:
        return {"Error": f"Missing Resource: {e}"}

    # --- B. HANDLE WEATHER ---
    target_date_obj = pd.to_datetime(target_date)
    
    if weather_profile is None:
        # Fetch default from our helper
        rain, wind, fog = get_climate(port_name, target_date_obj.month)
        weather_profile = {'rain': rain, 'wind': wind, 'fog': fog}
        weather_source = "Historical Avg"
    elif isinstance(weather_profile, (list, tuple)) and len(weather_profile) == 3:
      # If user passed a tuple/list, convert to dict
      weather_profile = {'rain': weather_profile[0], 'wind': weather_profile[1], 'fog': weather_profile[2]}
      weather_source = "User Scenario"
    else:
        weather_source = "User Scenario"

    # --- C. BRIDGE HISTORY TO FUTURE ---
    # Get latest historical point
    df_history = df_train[['Date', 'BDI']].sort_values('Date').drop_duplicates('Date')
    last_known_date = df_history['Date'].max()
    last_known_bdi = df_history.iloc[-1]['BDI']
    
    days_gap = (target_date_obj - last_known_date).days
    
    # Constant Scenario Override
    if scenario_type == 'constant': 
        target_bdi = last_known_bdi

    # Generate Bridge
    if days_gap > 0:
        bridge_vals = generate_bridge_curve(last_known_bdi, target_bdi, days_gap, scenario_type)
        # Create bridge dataframe
        bridge_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), end=target_date_obj)
        df_bridge = pd.DataFrame({'Date': bridge_dates, 'BDI': bridge_vals})
        df_full = pd.concat([df_history, df_bridge])
    else:
        df_full = df_history

    # --- D. CALCULATE INERTIA ---
    df_full['BDI_Inertia'] = df_full['BDI'].rolling(window=45, min_periods=1).mean()
    final_inertia = df_full.iloc[-1]['BDI_Inertia']

    # --- E. PREPARE MODEL INPUT ---
    # 1. Create Base Row
    input_data = pd.DataFrame([{
        'BDI': target_bdi,
        'BDI_Inertia': final_inertia,
        'Rain_Forecast': weather_profile['rain'],
        'Wind_Forecast': weather_profile['wind'],
        'Fog_Risk': weather_profile['fog'],
        'Month': target_date_obj.month
    }])

    # 2. Handle One-Hot Encoding (Crucial Step!)
    # The model expects columns like "Port_QINGDAO", "Port_TUBARAO", etc.
    # We must ensure all columns used in training exist here, set to 0 or 1.
    
    # Get feature names from the trained model
    model_features = model.get_booster().feature_names
    
    for feature in model_features:
        if feature not in input_data.columns:
            # Check if this feature matches our target port
            if feature == f"Port_{port_name.upper()}":
                input_data[feature] = 1
            else:
                input_data[feature] = 0
                
    # Reorder columns to match training EXACTLY
    input_data = input_data[model_features]

    # --- F. PREDICT ---
    pred_delay = model.predict(input_data)[0]

    return {
        'Scenario': scenario_type.upper(),
        'Weather_Mode': weather_source,
        'Weather_Details': weather_profile,
        'Target_Date': target_date,
        'End_BDI': round(float(target_bdi), 1),
        'Calculated_Inertia': round(float(final_inertia), 1),
        'Predicted_Delay': round(float(pred_delay), 2)
    }

# Function to get climate given port and month
def get_climate(port, month):
    port = port.upper()
    # 1. Fallback to generic if port not in DB (e.g., Jingtang uses Caofeidian logic)
    if port not in CLIMATE_DB:
        # Defaults based on region heuristics (all uppercase)
        if port in ['JINGTANG', 'QINHUANGDAO', 'HULUDAO', 'JINZHOU']: 
            data = CLIMATE_DB['CAOFEIDIAN']
        elif port in ['JIANGYIN', 'YANGZHOU', 'CHANGZHOU', 'XINMINZHOU']: 
            data = CLIMATE_DB['ZHANGJIAGANG']
        elif port in ['RIZHAO', 'LIANYUNGANG', 'DONGJIAKOU']: 
            data = CLIMATE_DB['QINGDAO']
        elif port in ['GUANGZHOU', 'QINZHOU']:
            data = CLIMATE_DB['FANGCHENG']
        elif port in ['NINGBO', 'ZHOUSHAN']:
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

# Generating future BDI curve based on scenario
def generate_bridge_curve(start_val, end_val, num_days, scenario_type='linear', last_known_bdi=None):
    """Generates synthetic market trajectory."""
    if num_days <= 0: return np.array([end_val])
    
    if scenario_type == 'linear':
        return np.linspace(start_val, end_val, num_days)
    
    elif scenario_type == 'spike': # Late Surge (Low Inertia)
        split_idx = int(num_days * 0.8)
        part1 = np.full(split_idx, start_val) 
        part2 = np.linspace(start_val, end_val, num_days - split_idx)
        return np.concatenate([part1, part2])
    
    elif scenario_type == 'crash': # Early Shift (High Inertia)
        split_idx = int(num_days * 0.2)
        part1 = np.linspace(start_val, end_val, split_idx)
        part2 = np.full(num_days - split_idx, end_val)
        return np.concatenate([part1, part2])
    
    elif scenario_type == 'stable': # Boring Market
        base = np.linspace(start_val, end_val, num_days)
        noise = np.random.normal(0, 20, num_days)
        return base + noise
    
    elif scenario_type == 'constant': # Constant BDI (use last_known_bdi)
        if last_known_bdi is not None:
            return np.full(num_days, last_known_bdi)
        else:
            return np.full(num_days, start_val)
    
    return np.linspace(start_val, end_val, num_days)
