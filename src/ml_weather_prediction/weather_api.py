import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import calendar

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_weather_predictor import WeatherPredictorML


class WeatherPredictionAPI:
    """
    Production-ready API for weather predictions in freight calculator.
    
    Usage:
        api = WeatherPredictionAPI()
        prediction = api.predict_voyage("2026-02-15", "south_china_sea")
        print(f"Expected delay: {prediction['voyage_delay_days']} days")
        print(f"Bad weather: {prediction['bad_weather_pct']}%")
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize the API with pre-trained models"""
        if model_dir is None:
            model_dir = Path(__file__).parent / "models"
        
        self.predictor = WeatherPredictorML(model_dir=str(model_dir))
        
        if not self.predictor.load_models():
            raise RuntimeError("No trained models found. Please run train_weather_model.py first.")
    
    def predict_voyage(self, date: str, route: str) -> Dict[str, float]:
        """
        Predict weather impact for a planned voyage using trained ML models.
        
        Args:
            date: ISO format date (YYYY-MM-DD) of voyage start
            route: One of 'australia_china', 'australia_south_korea', 'brazil_china',
                   'brazil_malaysia', 'canada_china', 'indonesia_india',
                   'south_africa_china', 'west_africa_india'
        
        Returns:
            {
                'bad_weather_pct': 0-100 (percentage),
                'voyage_delay_days': 0+ (expected additional days)
            }
        """
        return self.predictor.predict(date, route)
    
    def predict_batch(self, dates: list, route: str) -> pd.DataFrame:
        """
        Predict weather for multiple dates on a single route.
        
        Args:
            dates: List of ISO format dates (YYYY-MM-DD)
            route: One of 8 shipping routes
        
        Returns:
            DataFrame with columns: date, route, bad_weather_pct, voyage_delay_days
        """
        return self.predictor.predict_batch(dates, route)
    
    def predict_route_month(self, year: int, month: int, route: str) -> Dict:
        """
        Predict weather for all days in a month on a route.
        Useful for identifying risky periods.
        """
        # Generate dates for month
        days_in_month = calendar.monthrange(year, month)[1]
        dates = [f"{year:04d}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]
        
        # Get predictions
        predictions = []
        for date in dates:
            try:
                pred = self.predict_voyage(date, route)
                predictions.append(pred)
            except:
                predictions.append({'bad_weather_pct': None, 'voyage_delay_days': None})
        
        # Calculate summary
        valid_delays = [p['voyage_delay_days'] for p in predictions if p['voyage_delay_days'] is not None]
        valid_weather = [p['bad_weather_pct'] for p in predictions if p['bad_weather_pct'] is not None]
        
        risky_dates = [dates[i] for i, p in enumerate(predictions) 
                      if p['bad_weather_pct'] is not None and p['bad_weather_pct'] > 15]
        
        summary = {
            'avg_delay_days': sum(valid_delays) / len(valid_delays) if valid_delays else 0,
            'avg_bad_weather_pct': sum(valid_weather) / len(valid_weather) if valid_weather else 0,
            'max_delay_days': max(valid_delays) if valid_delays else 0,
            'min_delay_days': min(valid_delays) if valid_delays else 0,
            'risky_days': risky_dates,
            'risky_days_count': len(risky_dates)
        }
        
        return {
            'dates': dates,
            'predictions': predictions,
            'summary': summary
        }
    
    def compare_routes(self, date: str, routes: List[str] = None) -> Dict:
        """
        Compare weather predictions for different routes on the same date.
        Useful for voyage planning and route optimization.
        """
        if routes is None:
            routes = ['australia_china', 'brazil_china', 'south_africa_china', 'west_africa_india']
        
        comparisons = {}
        for route in routes:
            try:
                pred = self.predict_voyage(date, route)
                comparisons[route] = {
                    'bad_weather_pct': pred['bad_weather_pct'],
                    'voyage_delay_days': pred['voyage_delay_days']
                }
            except:
                comparisons[route] = {'bad_weather_pct': None, 'voyage_delay_days': None}
        
        # Find best and worst
        valid_comparisons = {k: v for k, v in comparisons.items() if v['bad_weather_pct'] is not None}
        best_route = min(valid_comparisons, key=lambda k: valid_comparisons[k]['bad_weather_pct']) if valid_comparisons else None
        worst_route = max(valid_comparisons, key=lambda k: valid_comparisons[k]['bad_weather_pct']) if valid_comparisons else None
        
        return {
            'date': date,
            'comparisons': comparisons,
            'best_route': best_route,
            'worst_route': worst_route
        }
    
    def calculate_impact_on_voyage(self, date: str, route: str, 
                                   base_voyage_days: float, 
                                   fuel_consumption_mt_per_day: float,
                                   fuel_price_per_mt: float) -> Dict:
        """
        Calculate the financial impact of weather on a voyage.
        """
        pred = self.predict_voyage(date, route)
        
        # Weather impact on voyage duration
        weather_delay = pred['voyage_delay_days']
        total_voyage_days = base_voyage_days + weather_delay
        
        # Weather impact on fuel consumption
        fuel_increase_pct = pred['bad_weather_pct']
        base_fuel = fuel_consumption_mt_per_day * base_voyage_days
        additional_fuel_mt = base_fuel * (fuel_increase_pct / 100)
        additional_fuel_cost = additional_fuel_mt * fuel_price_per_mt
        
        return {
            'base_voyage_days': base_voyage_days,
            'weather_delay_days': weather_delay,
            'total_voyage_days': total_voyage_days,
            'base_fuel_consumption_mt': base_fuel,
            'weather_fuel_increase_pct': fuel_increase_pct,
            'additional_fuel_mt': additional_fuel_mt,
            'additional_fuel_cost': additional_fuel_cost,
            'extra_hire_cost': 0,
            'total_weather_impact_cost': additional_fuel_cost
        }


def example_api_usage():
    """Demonstrate the API usage"""
    
    print("\n" + "=" * 70)
    print("WEATHER PREDICTION API - USAGE EXAMPLES")
    print("=" * 70)
    
    try:
        api = WeatherPredictionAPI()
        
        # Example 1: Single voyage prediction
        print("\n1. SINGLE VOYAGE PREDICTION")
        print("-" * 70)
        pred = api.predict_voyage("2026-02-15", "brazil_china")
        print(f"Date: 2026-02-15 | Route: Brazil-China")
        print(f"  Bad Weather: {pred['bad_weather_pct']:.1f}%")
        print(f"  Expected Delay: {pred['voyage_delay_days']:.2f} days")
        
        # Example 2: Route comparison
        print("\n2. ROUTE COMPARISON (2026-03-01)")
        print("-" * 70)
        comparison = api.compare_routes("2026-03-01")
        for route, data in comparison['comparisons'].items():
            print(f"{route:20s}: Weather={data['bad_weather_pct']:5.1f}% | Delay={data['voyage_delay_days']:4.2f} days")
        print(f"\nBest route: {comparison['best_route']}")
        
        # Example 3: Monthly forecast
        print("\n3. FEBRUARY 2026 - WEST AFRICA-INDIA")
        print("-" * 70)
        monthly = api.predict_route_month(2026, 2, "west_africa_india")
        summary = monthly['summary']
        print(f"Average bad weather: {summary['avg_bad_weather_pct']:.1f}%")
        print(f"Average delay: {summary['avg_delay_days']:.2f} days")
        print(f"Risky days (>15%): {summary['risky_days_count']}")
        
        # Example 4: Voyage cost impact
        print("\n4. WEATHER IMPACT ON VOYAGE ECONOMICS")
        print("-" * 70)
        impact = api.calculate_impact_on_voyage(
            date="2026-02-15",
            route="australia_china",
            base_voyage_days=20,
            fuel_consumption_mt_per_day=20,
            fuel_price_per_mt=450
        )
        print(f"Base duration: {impact['base_voyage_days']} days")
        print(f"Weather delay: {impact['weather_delay_days']:.2f} days")
        print(f"Additional fuel cost: ${impact['additional_fuel_cost']:,.0f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_api_usage()
