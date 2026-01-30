import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# 8 routes with distance and multipliers
routes_info = {
    'australia_china': {'distance_nm': 5400, 'multiplier': 0.8},
    'australia_south_korea': {'distance_nm': 5700, 'multiplier': 0.8},
    'brazil_china': {'distance_nm': 9200, 'multiplier': 1.2},
    'brazil_malaysia': {'distance_nm': 8400, 'multiplier': 1.15},
    'canada_china': {'distance_nm': 6800, 'multiplier': 0.95},
    'indonesia_india': {'distance_nm': 4200, 'multiplier': 0.7},
    'south_africa_china': {'distance_nm': 7800, 'multiplier': 1.0},
    'west_africa_india': {'distance_nm': 6500, 'multiplier': 0.9},
}

# Seasonal weather patterns with wider variance
seasonal_patterns = {
    'winter': {'base_weather': 28, 'variance': 12, 'base_wave': 2.5, 'base_wind': 16},
    'spring': {'base_weather': 22, 'variance': 10, 'base_wave': 1.8, 'base_wind': 12},
    'summer': {'base_weather': 18, 'variance': 8, 'base_wave': 1.2, 'base_wind': 9},
    'autumn': {'base_weather': 24, 'variance': 11, 'base_wave': 2.2, 'base_wind': 14},
}

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8, 9]:
        return 'summer'
    else:
        return 'autumn'

def generate_expanded_data(start_date, end_date, obs_per_route=200):
    """Generate expanded synthetic data with wider variance"""
    
    data = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Sample dates evenly across the range
    sample_dates = np.random.choice(date_range, size=obs_per_route, replace=False)
    sample_dates = sorted(sample_dates)
    
    for route_name, route_info in routes_info.items():
        for date in sample_dates:
            # Convert numpy datetime64 to pandas Timestamp
            date = pd.Timestamp(date)
            month = date.month
            season = get_season(month)
            pattern = seasonal_patterns[season]
            
            # Generate weather with WIDER VARIANCE (10-40%)
            # Add seasonal and random components
            seasonal_factor = np.sin(2 * np.pi * date.dayofyear / 365.0)
            random_weather_variation = np.random.normal(0, pattern['variance'])
            
            # Base weather + seasonal swing + random variation
            speed_reduction_pct = pattern['base_weather'] + (seasonal_factor * 8) + random_weather_variation
            speed_reduction_pct = np.clip(speed_reduction_pct, 10, 40)  # Constrain to 10-40%
            
            # Generate wave height (more variance)
            wave_variation = np.random.normal(0, 0.8)
            significant_wave_height_m = pattern['base_wave'] + (seasonal_factor * 0.5) + wave_variation
            significant_wave_height_m = np.clip(significant_wave_height_m, 0.5, 5.0)
            
            # Generate wind speed (more variance)
            wind_variation = np.random.normal(0, 3)
            wind_speed_knots = pattern['base_wind'] + (seasonal_factor * 3) + wind_variation
            wind_speed_knots = np.clip(wind_speed_knots, 5, 25)
            
            # Sea state code (1-7)
            sea_state_code = np.clip(int(2 + significant_wave_height_m), 1, 7)
            
            # Fuel increase correlates with speed reduction
            fuel_increase_pct = speed_reduction_pct * 0.5 + np.random.normal(0, 2)
            fuel_increase_pct = np.clip(fuel_increase_pct, 5, 25)
            
            # Voyage delay calculation
            # Distance effect + weather impact + random variation
            distance_factor = route_info['distance_nm'] / 5000  # Normalize by typical distance
            weather_delay_factor = speed_reduction_pct / 20  # Higher weather = more delay
            random_delay = np.random.normal(0, 0.15)
            
            voyage_delay_days = (distance_factor * route_info['multiplier'] * weather_delay_factor) + random_delay
            voyage_delay_days = np.clip(voyage_delay_days, 0.1, 3.5)
            
            data.append({
                'date': date,
                'route': route_name,
                'season': season,
                'month': month,
                'significant_wave_height_m': round(significant_wave_height_m, 2),
                'wind_speed_knots': round(wind_speed_knots, 2),
                'sea_state_code': sea_state_code,
                'speed_reduction_pct': round(speed_reduction_pct, 2),
                'fuel_increase_pct': round(fuel_increase_pct, 2),
                'voyage_delay_days': round(voyage_delay_days, 3)
            })
    
    return pd.DataFrame(data)

# Generate data from 2025-01-01 to 2026-12-31 (2 years)
start = '2024-12-01'
end = '2026-12-31'
observations_per_route = 250  # Increased from 100 to 250

print(f"\nDate Range: {start} to {end}")
print(f"Observations per route: {observations_per_route}")
print(f"Total routes: {len(routes_info)}")
print(f"Expected total observations: {len(routes_info) * observations_per_route}")

df = generate_expanded_data(start, end, obs_per_route=observations_per_route)

print(f"\nActual observations generated: {len(df)}")
print(f"\nData Summary:")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Routes: {df['route'].nunique()}")
print(f"  Seasons: {df['season'].unique().tolist()}")

print(f"\nWeather Metrics (EXPANDED RANGE):")
print(f"  Speed Reduction %:")
print(f"    Min:  {df['speed_reduction_pct'].min():.2f}%")
print(f"    Max:  {df['speed_reduction_pct'].max():.2f}%")
print(f"    Mean: {df['speed_reduction_pct'].mean():.2f}%")
print(f"    Std:  {df['speed_reduction_pct'].std():.2f}%")

print(f"\n  Voyage Delay:")
print(f"    Min:  {df['voyage_delay_days'].min():.3f} days")
print(f"    Max:  {df['voyage_delay_days'].max():.3f} days")
print(f"    Mean: {df['voyage_delay_days'].mean():.3f} days")
print(f"    Std:  {df['voyage_delay_days'].std():.3f} days")

print(f"\n  Wave Height:")
print(f"    Min:  {df['significant_wave_height_m'].min():.2f} m")
print(f"    Max:  {df['significant_wave_height_m'].max():.2f} m")
print(f"    Mean: {df['significant_wave_height_m'].mean():.2f} m")

print(f"\n  Wind Speed:")
print(f"    Min:  {df['wind_speed_knots'].min():.2f} knots")
print(f"    Max:  {df['wind_speed_knots'].max():.2f} knots")
print(f"    Mean: {df['wind_speed_knots'].mean():.2f} knots")

# Save the expanded data
output_path = 'data/synthetic_weather_data_expanded.csv'
df.to_csv(output_path, index=False)
print(f"\nExpanded data saved to: {output_path}")

# Show sample rows
print(f"\nSample Data (first 10 rows):")
print(df.head(10).to_string(index=False))
