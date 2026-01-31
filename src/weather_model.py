"""
Weather Prediction Model for Freight Calculator
Predicts how weather conditions affect:
- Vessel speed
- Bunker consumption
- Voyage duration
"""

from dataclasses import dataclass
from typing import Tuple
import math

@dataclass
class WeatherCondition:
    """Represents weather conditions on a route"""
    significant_wave_height_m: float  # meters (0-15+)
    wind_speed_knots: float           # knots (0-60+)
    sea_state_code: int               # Beaufort scale (0-12)
    route: str = "default"
    season: str = "summer"            # summer, winter, monsoon


@dataclass
class WeatherImpact:
    """Output: Weather impact on vessel performance"""
    speed_reduction_pct: float        # % reduction in speed (0-50%)
    consumption_increase_pct: float   # % increase in fuel (0-100%)
    voyage_delay_days: float          # additional days added
    weather_severity: str             # light, moderate, heavy, severe


class WeatherPredictor:
    """
    Predicts weather impact on vessel operations.
    
    References:
    - Beaufort scale (wave height & wind)
    - Vessel response to sea state
    - Route-specific seasonal patterns
    - IMO Energy Efficiency Design Index (EEDI) guidelines
    """
    
    # Beaufort scale thresholds (significant wave height in meters)
    BEAUFORT_THRESHOLDS = {
        0: (0, 0.1),           # Calm
        1: (0.1, 0.5),         # Light air
        2: (0.5, 1.25),        # Light breeze
        3: (1.25, 2.5),        # Gentle breeze
        4: (2.5, 4),           # Moderate breeze
        5: (4, 6),             # Fresh breeze
        6: (6, 9),             # Strong breeze
        7: (9, 12.5),          # Near gale
        8: (12.5, 16),         # Gale
        9: (16, 20),           # Strong gale
        10: (20, 24),          # Storm
        11: (24, 32),          # Violent storm
        12: (32, 999)          # Hurricane
    }
    
    # Speed reduction factors by sea state
    SPEED_REDUCTION_BY_STATE = {
        0: 0.0,      # Calm
        1: 0.02,     # Light
        2: 0.03,
        3: 0.05,     # Gentle
        4: 0.08,     # Moderate
        5: 0.12,     # Fresh
        6: 0.18,     # Strong
        7: 0.25,     # Near gale
        8: 0.35,     # Gale
        9: 0.45,     # Strong gale
        10: 0.55,    # Storm
        11: 0.70,    # Violent
        12: 0.85     # Hurricane
    }
    
    # Fuel consumption increase factors by sea state
    FUEL_INCREASE_BY_STATE = {
        0: 0.0,      # Calm
        1: 0.02,
        2: 0.04,
        3: 0.08,     # Gentle
        4: 0.15,     # Moderate
        5: 0.25,     # Fresh
        6: 0.40,     # Strong
        7: 0.55,     # Near gale
        8: 0.75,     # Gale
        9: 1.00,     # Strong gale
        10: 1.40,    # Storm
        11: 1.90,    # Violent
        12: 2.50     # Hurricane
    }
    
    # Route-specific seasonal patterns (multipliers)
    ROUTE_SEASONAL_FACTORS = {
        "south_china_sea": {
            "summer": {"speed_reduction": 0.08, "fuel_increase": 0.15, "delay": 0.2},
            "winter": {"speed_reduction": 0.05, "fuel_increase": 0.10, "delay": 0.1},
            "monsoon": {"speed_reduction": 0.20, "fuel_increase": 0.35, "delay": 0.5}
        },
        "north_atlantic": {
            "summer": {"speed_reduction": 0.05, "fuel_increase": 0.08, "delay": 0.1},
            "winter": {"speed_reduction": 0.15, "fuel_increase": 0.25, "delay": 0.4},
        },
        "indian_ocean": {
            "summer": {"speed_reduction": 0.06, "fuel_increase": 0.12, "delay": 0.15},
            "winter": {"speed_reduction": 0.04, "fuel_increase": 0.08, "delay": 0.1},
            "monsoon": {"speed_reduction": 0.18, "fuel_increase": 0.30, "delay": 0.45}
        },
        "default": {
            "summer": {"speed_reduction": 0.03, "fuel_increase": 0.05, "delay": 0.05},
            "winter": {"speed_reduction": 0.08, "fuel_increase": 0.15, "delay": 0.2},
        }
    }
    
    @staticmethod
    def get_beaufort_scale(wave_height_m: float) -> int:
        """Convert wave height (meters) to Beaufort scale (0-12)"""
        for scale, (min_h, max_h) in WeatherPredictor.BEAUFORT_THRESHOLDS.items():
            if min_h <= wave_height_m < max_h:
                return scale
        return 12  # Hurricane
    
    @staticmethod
    def predict_impact(weather: WeatherCondition, 
                      distance_nm: float = 1000,
                      vessel_speed_knots: float = 14) -> WeatherImpact:
        """
        Predict weather impact on vessel performance.
        
        Args:
            weather: Weather condition object
            distance_nm: Distance to travel (nautical miles)
            vessel_speed_knots: Vessel's normal speed (knots)
        
        Returns:
            WeatherImpact object with speed reduction, fuel increase, delay
        """
        
        # Step 1: Determine sea state from wave height
        sea_state = weather.sea_state_code or WeatherPredictor.get_beaufort_scale(
            weather.significant_wave_height_m
        )
        
        # Step 2: Get base speed & fuel impact from sea state
        speed_reduction = WeatherPredictor.SPEED_REDUCTION_BY_STATE.get(sea_state, 0.5)
        fuel_increase = WeatherPredictor.FUEL_INCREASE_BY_STATE.get(sea_state, 1.0)
        
        # Step 3: Apply route-specific seasonal adjustments
        route = weather.route.lower() or "default"
        season = weather.season.lower() or "summer"
        
        route_factors = WeatherPredictor.ROUTE_SEASONAL_FACTORS.get(route, {})
        seasonal_factor = route_factors.get(season, {})
        
        speed_adjustment = seasonal_factor.get("speed_reduction", 0)
        fuel_adjustment = seasonal_factor.get("fuel_increase", 0)
        
        # Combine effects (additive approach)
        final_speed_reduction_pct = (speed_reduction + speed_adjustment) * 100
        final_fuel_increase_pct = (fuel_increase + fuel_adjustment) * 100
        
        # Step 4: Calculate delay in days
        # Reduced speed increases transit time
        actual_speed = vessel_speed_knots * (1 - speed_reduction - speed_adjustment)
        normal_transit_days = distance_nm / (vessel_speed_knots * 24)
        actual_transit_days = distance_nm / (actual_speed * 24) if actual_speed > 0 else normal_transit_days * 2
        voyage_delay_days = actual_transit_days - normal_transit_days
        
        # Determine severity
        if sea_state <= 3:
            severity = "light"
        elif sea_state <= 6:
            severity = "moderate"
        elif sea_state <= 9:
            severity = "heavy"
        else:
            severity = "severe"
        
        return WeatherImpact(
            speed_reduction_pct=min(final_speed_reduction_pct, 85),  # Cap at 85%
            consumption_increase_pct=final_fuel_increase_pct,
            voyage_delay_days=max(voyage_delay_days, 0),
            weather_severity=severity
        )
    
    @staticmethod
    def predict_multiple_routes(weather_conditions: list,
                               distance_nm: float,
                               vessel_speed_knots: float) -> dict:
        """
        Predict impact for multiple weather scenarios.
        
        Returns dict with impacts for each weather condition.
        """
        impacts = {}
        for i, weather in enumerate(weather_conditions):
            key = f"route_{i}" if not weather.route else weather.route
            impacts[key] = WeatherPredictor.predict_impact(
                weather, distance_nm, vessel_speed_knots
            )
        return impacts


# Example usage scenarios
def example_weather_scenarios():
    """Demonstrate weather prediction for different conditions"""
    
    print("=" * 70)
    print("WEATHER IMPACT PREDICTION - FREIGHT CALCULATOR")
    print("=" * 70)
    
    # Scenario 1: Calm weather
    calm = WeatherCondition(
        significant_wave_height_m=0.5,
        wind_speed_knots=5,
        sea_state_code=2,
        route="north_atlantic",
        season="summer"
    )
    
    impact1 = WeatherPredictor.predict_impact(calm, distance_nm=3000, vessel_speed_knots=14)
    print("\n1. CALM CONDITIONS (North Atlantic, Summer)")
    print(f"   Wave height: 0.5m, Wind: 5 knots")
    print(f"   Speed reduction: {impact1.speed_reduction_pct:.1f}%")
    print(f"   Fuel increase: {impact1.consumption_increase_pct:.1f}%")
    print(f"   Delay: {impact1.voyage_delay_days:.2f} days")
    print(f"   Severity: {impact1.weather_severity}")
    
    # Scenario 2: Moderate weather
    moderate = WeatherCondition(
        significant_wave_height_m=4.0,
        wind_speed_knots=20,
        sea_state_code=4,
        route="indian_ocean",
        season="monsoon"
    )
    
    impact2 = WeatherPredictor.predict_impact(moderate, distance_nm=3000, vessel_speed_knots=14)
    print("\n2. MODERATE CONDITIONS (Indian Ocean, Monsoon)")
    print(f"   Wave height: 4.0m, Wind: 20 knots")
    print(f"   Speed reduction: {impact2.speed_reduction_pct:.1f}%")
    print(f"   Fuel increase: {impact2.consumption_increase_pct:.1f}%")
    print(f"   Delay: {impact2.voyage_delay_days:.2f} days")
    print(f"   Severity: {impact2.weather_severity}")
    
    # Scenario 3: Heavy weather
    heavy = WeatherCondition(
        significant_wave_height_m=10.0,
        wind_speed_knots=40,
        sea_state_code=8,
        route="south_china_sea",
        season="winter"
    )
    
    impact3 = WeatherPredictor.predict_impact(heavy, distance_nm=3000, vessel_speed_knots=14)
    print("\n3. HEAVY CONDITIONS (South China Sea, Winter)")
    print(f"   Wave height: 10.0m, Wind: 40 knots")
    print(f"   Speed reduction: {impact3.speed_reduction_pct:.1f}%")
    print(f"   Fuel increase: {impact3.consumption_increase_pct:.1f}%")
    print(f"   Delay: {impact3.voyage_delay_days:.2f} days")
    print(f"   Severity: {impact3.weather_severity}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_weather_scenarios()
