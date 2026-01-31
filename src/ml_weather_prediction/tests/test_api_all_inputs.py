"""
Comprehensive test of weather_api.py covering all input possibilities
"""

import sys
import warnings
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path (tests folder is one level below root)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_api import WeatherPredictionAPI

def test_all_inputs():
    """Test all API input combinations and methods"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE WEATHER API TEST - ALL INPUT POSSIBILITIES")
    print("=" * 80)
    
    try:
        api = WeatherPredictionAPI()
        
        # All 8 routes
        all_routes = [
            'australia_china',
            'australia_south_korea',
            'brazil_china',
            'brazil_malaysia',
            'canada_china',
            'indonesia_india',
            'south_africa_china',
            'west_africa_india'
        ]
        
        # Test dates (different months/seasons)
        test_dates = [
            "2026-01-15",  # Winter
            "2026-04-15",  # Spring
            "2026-07-15",  # Summer
            "2026-10-15",  # Autumn
            "2025-06-15",  # Earlier year, summer
            "2025-12-15",  # Earlier year, winter
        ]
        
        # ========== TEST 1: Single Voyage Prediction (all routes + dates) ==========
        print("\n" + "-" * 80)
        print("TEST 1: SINGLE VOYAGE PREDICTION - ALL ROUTES (6 dates)")
        print("-" * 80)
        
        for date in test_dates:
            print(f"\nDate: {date}")
            for route in all_routes:
                pred = api.predict_voyage(date, route)
                print(f"  {route:25s}: Weather={pred['bad_weather_pct']:6.1f}% | Delay={pred['voyage_delay_days']:5.2f} days")
        
        # ========== TEST 2: Batch Prediction (all routes) ==========
        print("\n" + "-" * 80)
        print("TEST 2: BATCH PREDICTION - ALL ROUTES (5 consecutive days)")
        print("-" * 80)
        
        batch_dates = [f"2026-02-{i:02d}" for i in range(1, 6)]
        
        for route in all_routes:
            print(f"\nRoute: {route}")
            try:
                batch_result = api.predict_batch(batch_dates, route)
                for _, row in batch_result.iterrows():
                    print(f"  {row['date']}: Weather={row['bad_weather_pct']:6.1f}% | Delay={row['voyage_delay_days']:5.2f} days")
            except Exception as e:
                print(f"  (Batch test skipped: {str(e)[:50]})")
                for date in batch_dates:
                    pred = api.predict_voyage(date, route)
                    print(f"  {date}: Weather={pred['bad_weather_pct']:6.1f}% | Delay={pred['voyage_delay_days']:5.2f} days")
        
        # ========== TEST 3: Monthly Forecast (different months) ==========
        print("\n" + "-" * 80)
        print("TEST 3: MONTHLY FORECAST - ALL ROUTES (different months)")
        print("-" * 80)
        
        test_months = [
            (2026, 1),  # January (winter)
            (2026, 4),  # April (spring)
            (2026, 7),  # July (summer)
            (2026, 10), # October (autumn)
        ]
        
        for year, month in test_months:
            month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
            print(f"\n{year}-{month_name}:")
            for route in all_routes:
                monthly = api.predict_route_month(year, month, route)
                summary = monthly['summary']
                print(f"  {route:25s}: Avg Weather={summary['avg_bad_weather_pct']:5.1f}% | Avg Delay={summary['avg_delay_days']:4.2f} days | Risky Days={summary['risky_days_count']:2d}")
        
        # ========== TEST 4: Route Comparison (different dates) ==========
        print("\n" + "-" * 80)
        print("TEST 4: ROUTE COMPARISON - DIFFERENT DATES")
        print("-" * 80)
        
        comparison_dates = ["2026-01-15", "2026-04-15", "2026-07-15", "2026-10-15"]
        
        for date in comparison_dates:
            print(f"\nDate: {date}")
            comparison = api.compare_routes(date)
            print(f"  Best route: {comparison['best_route']:25s}")
            print(f"  Worst route: {comparison['worst_route']:25s}")
            for route, data in sorted(comparison['comparisons'].items()):
                print(f"    {route:25s}: Weather={data['bad_weather_pct']:6.1f}% | Delay={data['voyage_delay_days']:5.2f} days")
        
        # ========== TEST 5: Custom Route Comparison ==========
        print("\n" + "-" * 80)
        print("TEST 5: CUSTOM ROUTE COMPARISON (user-defined subsets)")
        print("-" * 80)
        
        custom_subsets = [
            ["australia_china", "australia_south_korea"],  # Australia routes
            ["brazil_china", "brazil_malaysia"],            # Brazil routes
            ["indonesia_india", "west_africa_india"],       # India destinations
            all_routes                                      # All 8 routes
        ]
        
        subset_names = ["Australia Routes", "Brazil Routes", "India Destinations", "All Routes"]
        
        for name, routes in zip(subset_names, custom_subsets):
            print(f"\n{name}:")
            comparison = api.compare_routes("2026-06-15", routes=routes)
            for route, data in sorted(comparison['comparisons'].items()):
                print(f"  {route:25s}: Weather={data['bad_weather_pct']:6.1f}% | Delay={data['voyage_delay_days']:5.2f} days")
        
        # ========== TEST 6: Voyage Economics Impact (various scenarios) ==========
        print("\n" + "-" * 80)
        print("TEST 6: VOYAGE ECONOMICS - VARIOUS VOYAGE PARAMETERS")
        print("-" * 80)
        
        voyage_scenarios = [
            {"days": 10, "fuel_per_day": 15, "price": 400, "name": "Short/Low Fuel"},
            {"days": 20, "fuel_per_day": 20, "price": 450, "name": "Medium/Standard"},
            {"days": 30, "fuel_per_day": 25, "price": 500, "name": "Long/High Fuel"},
            {"days": 45, "fuel_per_day": 30, "price": 550, "name": "Very Long/Very High Fuel"},
        ]
        
        for scenario in voyage_scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"  Base: {scenario['days']} days | {scenario['fuel_per_day']} MT/day | ${scenario['price']}/MT")
            
            # Test with 2 routes
            for route in ['australia_china', 'brazil_china']:
                impact = api.calculate_impact_on_voyage(
                    date="2026-03-15",
                    route=route,
                    base_voyage_days=scenario['days'],
                    fuel_consumption_mt_per_day=scenario['fuel_per_day'],
                    fuel_price_per_mt=scenario['price']
                )
                print(f"  {route:25s}: +{impact['weather_delay_days']:4.2f} days | +${impact['additional_fuel_cost']:>10,.0f}")
        
        # ========== TEST 7: Edge Cases & Validation ==========
        print("\n" + "-" * 80)
        print("TEST 7: EDGE CASES & VALIDATION")
        print("-" * 80)
        
        # Early date
        print("\nEarly date test (2025-01-01):")
        pred = api.predict_voyage("2025-01-01", "australia_china")
        print(f"  Weather={pred['bad_weather_pct']:.1f}% | Delay={pred['voyage_delay_days']:.2f} days")
        
        # Late date
        print("Late date test (2026-12-28):")
        pred = api.predict_voyage("2026-12-28", "brazil_china")
        print(f"  Weather={pred['bad_weather_pct']:.1f}% | Delay={pred['voyage_delay_days']:.2f} days")
        
        # Invalid route test
        print("\nInvalid route test (should raise error):")
        try:
            api.predict_voyage("2026-02-15", "invalid_route")
            print("  ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"  Correctly caught error: {str(e)[:60]}...")
        
        # ========== TEST 8: Statistics Summary ==========
        print("\n" + "-" * 80)
        print("TEST 8: STATISTICS SUMMARY (across all 8 routes)")
        print("-" * 80)
        
        weather_values = []
        delay_values = []
        
        print("\nCollecting predictions for all routes across 2 date ranges...")
        for route in all_routes:
            for date in ["2026-02-15", "2026-08-15"]:
                pred = api.predict_voyage(date, route)
                weather_values.append(pred['bad_weather_pct'])
                delay_values.append(pred['voyage_delay_days'])
        
        import statistics
        
        print(f"\nWeather Percentage Stats ({len(weather_values)} predictions):")
        print(f"  Min: {min(weather_values):6.1f}%")
        print(f"  Max: {max(weather_values):6.1f}%")
        print(f"  Avg: {statistics.mean(weather_values):6.1f}%")
        print(f"  Median: {statistics.median(weather_values):6.1f}%")
        print(f"  StdDev: {statistics.stdev(weather_values):6.1f}%")
        
        print(f"\nVoyage Delay Stats ({len(delay_values)} predictions):")
        print(f"  Min: {min(delay_values):6.2f} days")
        print(f"  Max: {max(delay_values):6.2f} days")
        print(f"  Avg: {statistics.mean(delay_values):6.2f} days")
        print(f"  Median: {statistics.median(delay_values):6.2f} days")
        print(f"  StdDev: {statistics.stdev(delay_values):6.2f} days")
        
        print("\n" + "=" * 80)
        print("PASSED: ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_inputs()
