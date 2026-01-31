"""
ML Weather Prediction Model for Freight Calculator
Predicts how weather conditions affect vessel speed, fuel consumption, and voyage delays.
Input: date and route. Output: bad_weather_pct and voyage_delay_days.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
import pickle
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class WeatherPredictorML:
    """
    ML-based Weather Predictor for freight calculation.
    
    Predicts bad weather % and voyage delays from date and route.
    Uses Gradient Boosting Regressor with proper feature engineering.
    
    Features:
    - Route: south_china_sea, north_atlantic, indian_ocean
    - Time: Cyclical encoding of day/month for seasonal patterns
    - Output: bad_weather_pct (0-100), voyage_delay_days (0+)
    """
    
    def __init__(self, model_dir: str = None):
        """Initialize the model"""
        self.model_dir = model_dir or Path(__file__).parent / "models"
        Path(self.model_dir).mkdir(exist_ok=True)
        
        # Models
        self.gb_delay_model = None
        self.gb_weather_model = None
        self.gb_combined_delay_model = None  # NEW: Combined model using weather as input
        self.nn_delay_model = None
        self.nn_weather_model = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.route_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        
        # Metadata
        self.feature_names_delay = None
        self.feature_names_weather = None
        self.routes = ['west_africa_china', 'australia_china', 'brazil_china']
        self.seasons = ['winter', 'spring', 'summer', 'autumn', 'monsoon']
        
    def _load_weather_data(self, csv_path: str) -> pd.DataFrame:
        """Load synthetic weather data"""
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal and categorical features"""
        df = df.copy()
        
        # Temporal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding (sin/cos)
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        # Encoding
        df['route_encoded'] = self.route_encoder.fit_transform(df['route'])
        df['season_encoded'] = self.season_encoder.fit_transform(df['season'])
        
        # Quartiles
        df['wave_quartile'] = pd.qcut(df['significant_wave_height_m'], 
                                       q=4, labels=False, duplicates='drop')
        df['wind_quartile'] = pd.qcut(df['wind_speed_knots'], 
                                       q=4, labels=False, duplicates='drop')
        
        return df
    
    def _create_feature_sets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create feature vectors for models"""
        features = [
            'day_of_year', 'month', 'week_of_year',
            'sin_day', 'cos_day', 'sin_month', 'cos_month',
            'route_encoded', 'season_encoded',
            'wave_quartile', 'wind_quartile'
        ]
        
        self.feature_names_delay = features
        self.feature_names_weather = features
        
        return df[features], df[features]
    
    def train(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
        """Train weather prediction models"""
        print("=" * 70)
        print("TRAINING ML WEATHER PREDICTION MODELS")
        print("=" * 70)
        
        # Load and prepare data
        df = self._load_weather_data(csv_path)
        print(f"\n[LOAD] {len(df)} weather observations loaded")
        
        df = self._engineer_features(df)
        print("[FEAT] Feature engineering complete")
        
        X_delay, X_weather = self._create_feature_sets(df)
        
        y_delay = df['voyage_delay_days'].values
        y_weather = df['speed_reduction_pct'].values
        
        print(f"[DATA] Delay: mean={y_delay.mean():.2f} days, std={y_delay.std():.2f}")
        print(f"[DATA] Weather: mean={y_weather.mean():.2f}%, std={y_weather.std():.2f}%")
        
        # Scale features
        X_delay_scaled = self.scaler.fit_transform(X_delay)
        X_weather_scaled = self.scaler.fit_transform(X_weather)
        
        # Split data
        (X_delay_train, X_delay_test, y_delay_train, y_delay_test) = train_test_split(
            X_delay_scaled, y_delay, test_size=test_size, random_state=random_state
        )
        
        (X_weather_train, X_weather_test, y_weather_train, y_weather_test) = train_test_split(
            X_weather_scaled, y_weather, test_size=test_size, random_state=random_state
        )
        
        print(f"[SPLIT] {len(X_delay_train)} train / {len(X_delay_test)} test")
        
        # Train delay model
        print("\n" + "-" * 70)
        print("TRAINING DELAY PREDICTION")
        print("-" * 70)
        
        print("\n[TRAIN] Gradient Boosting (Delay)...")
        self.gb_delay_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=random_state, verbose=0
        )
        self.gb_delay_model.fit(X_delay_train, y_delay_train)
        
        y_delay_pred = self.gb_delay_model.predict(X_delay_test)
        r2 = r2_score(y_delay_test, y_delay_pred)
        rmse = np.sqrt(mean_squared_error(y_delay_test, y_delay_pred))
        mae = mean_absolute_error(y_delay_test, y_delay_pred)
        
        print(f"   R2: {r2:.4f}, RMSE: {rmse:.4f} days, MAE: {mae:.4f} days")
        
        # Train weather model
        print("\n" + "-" * 70)
        print("TRAINING BAD WEATHER PREDICTION")
        print("-" * 70)
        
        print("\n[TRAIN] Gradient Boosting (Bad Weather %)...")
        self.gb_weather_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=random_state, verbose=0
        )
        self.gb_weather_model.fit(X_weather_train, y_weather_train)
        
        y_weather_pred = self.gb_weather_model.predict(X_weather_test)
        r2_w = r2_score(y_weather_test, y_weather_pred)
        rmse_w = np.sqrt(mean_squared_error(y_weather_test, y_weather_pred))
        mae_w = mean_absolute_error(y_weather_test, y_weather_pred)
        
        print(f"   R2: {r2_w:.4f}, RMSE: {rmse_w:.4f}%, MAE: {mae_w:.4f}%")
        
        # Train COMBINED delay model (using bad weather % as input)
        print("\n" + "-" * 70)
        print("TRAINING COMBINED DELAY MODEL (uses weather as input)")
        print("-" * 70)
        
        # First predict bad weather % on training set
        y_weather_train_pred = self.gb_weather_model.predict(X_weather_train)
        
        # Create extended feature set: original features + predicted bad_weather_pct
        X_combined_train = np.hstack([X_delay_train, y_weather_train_pred.reshape(-1, 1)])
        X_combined_test = np.hstack([X_delay_test, self.gb_weather_model.predict(X_weather_test).reshape(-1, 1)])
        
        print("\n[TRAIN] Gradient Boosting (Combined Delay using Weather)...")
        self.gb_combined_delay_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=random_state, verbose=0
        )
        self.gb_combined_delay_model.fit(X_combined_train, y_delay_train)
        
        y_combined_delay_pred = self.gb_combined_delay_model.predict(X_combined_test)
        r2_combined = r2_score(y_delay_test, y_combined_delay_pred)
        rmse_combined = np.sqrt(mean_squared_error(y_delay_test, y_combined_delay_pred))
        mae_combined = mean_absolute_error(y_delay_test, y_combined_delay_pred)
        
        print(f"   R2: {r2_combined:.4f}, RMSE: {rmse_combined:.4f} days, MAE: {mae_combined:.4f} days")
        
        # Compare models
        print(f"\n[COMPARE] Separate vs Combined Delay Model:")
        print(f"   Separate R2:  {r2:.4f}")
        print(f"   Combined R2:  {r2_combined:.4f}")
        improvement = ((r2_combined - r2) / abs(r2)) * 100 if r2 != 0 else 0
        print(f"   Improvement:  {improvement:+.1f}%")
        
        print("\n" + "=" * 70)
        
        self._save_models()
    
    def predict(self, date: str, route: str, use_combined: bool = True) -> Dict[str, float]:
        """
        Predict bad weather % and delay for a date and route.
        
        Args:
            date: ISO format (YYYY-MM-DD)
            route: One of 'west_africa_china', 'australia_china', 'brazil_china'
            use_combined: If True, use combined model (weather→delay). If False, use separate model.
        
        Returns:
            {bad_weather_pct, voyage_delay_days, confidence}
        """
        if self.gb_weather_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        if route not in self.routes:
            raise ValueError(f"Route must be one of {self.routes}")
        
        # Parse date
        date_obj = pd.to_datetime(date)
        month = date_obj.month
        season = self._get_season(month)
        
        # Engineer features
        features_dict = {
            'date': date_obj,
            'route': route,
            'season': season,
            'day_of_year': date_obj.dayofyear,
            'month': month,
            'week_of_year': date_obj.isocalendar().week,
            'sin_day': np.sin(2 * np.pi * date_obj.dayofyear / 365.0),
            'cos_day': np.cos(2 * np.pi * date_obj.dayofyear / 365.0),
            'sin_month': np.sin(2 * np.pi * month / 12.0),
            'cos_month': np.cos(2 * np.pi * month / 12.0),
            'route_encoded': self.route_encoder.transform([route])[0],
            'season_encoded': self.season_encoder.transform([season])[0],
            'wave_quartile': 2,
            'wind_quartile': 2,
        }
        
        # Create feature vector
        X = np.array([[features_dict[name] for name in self.feature_names_delay]])
        X_scaled = self.scaler.transform(X)
        
        # Predict bad weather percentage
        weather = float(self.gb_weather_model.predict(X_scaled)[0])
        
        # Predict delay using combined model (weather → delay) or separate model
        if use_combined and self.gb_combined_delay_model is not None:
            # Combined model: uses features + weather prediction
            X_combined = np.hstack([X_scaled, np.array([[weather]])])
            delay = float(self.gb_combined_delay_model.predict(X_combined)[0])
            model_type = "combined"
        else:
            # Separate model: uses only original features
            delay = float(self.gb_delay_model.predict(X_scaled)[0])
            model_type = "separate"
        
        return {
            'bad_weather_pct': max(0, min(100, weather)),
            'voyage_delay_days': max(0, delay),
            'confidence': 0.70,
            'model': model_type
        }
    
    def predict_batch(self, dates: list, route: str) -> pd.DataFrame:
        """Predict for multiple dates on a route"""
        results = []
        for date in dates:
            pred = self.predict(date, route)
            pred['date'] = date
            pred['route'] = route
            results.append(pred)
        return pd.DataFrame(results)
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8, 9]:
            return 'summer'
        else:
            return 'autumn'
    
    def _save_models(self):
        """Save models to disk"""
        model_path = Path(self.model_dir)
        model_path.mkdir(exist_ok=True)
        
        with open(model_path / 'gb_delay_model.pkl', 'wb') as f:
            pickle.dump(self.gb_delay_model, f)
        
        with open(model_path / 'gb_weather_model.pkl', 'wb') as f:
            pickle.dump(self.gb_weather_model, f)
        
        with open(model_path / 'gb_combined_delay_model.pkl', 'wb') as f:
            pickle.dump(self.gb_combined_delay_model, f)
        
        with open(model_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_path / 'route_encoder.pkl', 'wb') as f:
            pickle.dump(self.route_encoder, f)
        
        with open(model_path / 'season_encoder.pkl', 'wb') as f:
            pickle.dump(self.season_encoder, f)
        
        with open(model_path / 'feature_names.pkl', 'wb') as f:
            pickle.dump({
                'delay': self.feature_names_delay,
                'weather': self.feature_names_weather
            }, f)
        
        print(f"\n[SAVE] Models saved to {model_path}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        model_path = Path(self.model_dir)
        
        try:
            with open(model_path / 'gb_delay_model.pkl', 'rb') as f:
                self.gb_delay_model = pickle.load(f)
            
            with open(model_path / 'gb_weather_model.pkl', 'rb') as f:
                self.gb_weather_model = pickle.load(f)
            
            # Load combined model if available
            combined_path = model_path / 'gb_combined_delay_model.pkl'
            if combined_path.exists():
                with open(combined_path, 'rb') as f:
                    self.gb_combined_delay_model = pickle.load(f)
            
            with open(model_path / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(model_path / 'route_encoder.pkl', 'rb') as f:
                self.route_encoder = pickle.load(f)
            
            with open(model_path / 'season_encoder.pkl', 'rb') as f:
                self.season_encoder = pickle.load(f)
            
            with open(model_path / 'feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
                self.feature_names_delay = feature_names['delay']
                self.feature_names_weather = feature_names['weather']
            
            print("[LOAD] Models loaded successfully")
            return True
        
        except FileNotFoundError:
            print("[ERROR] No saved models found. Please train first.")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from GB model"""
        if self.gb_delay_model is None:
            raise ValueError("Model not trained yet")
        
        importance_delay = self.gb_delay_model.feature_importances_
        importance_weather = self.gb_weather_model.feature_importances_
        
        avg_importance = (importance_delay + importance_weather) / 2
        
        feature_importance = dict(zip(self.feature_names_delay, avg_importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))


def example_predictions():
    """Demonstrate model usage"""
    
    print("\n" + "=" * 70)
    print("WEATHER PREDICTION EXAMPLES")
    print("=" * 70)
    
    predictor = WeatherPredictorML()
    
    # Try to load existing models
    if not predictor.load_models():
        print("\n[TRAIN] Training new models...")
        data_path = Path(__file__).parent.parent / "data" / "synthetic_weather_data.csv"
        predictor.train(str(data_path))
    
    # Test predictions
    test_cases = [
        ("2026-02-15", "brazil_china"),
        ("2026-03-10", "australia_china"),
        ("2026-04-02", "west_africa_china"),
        ("2025-01-15", "brazil_china"),
        ("2024-07-01", "australia_china"),
    ]
    
    for date, route in test_cases:
        pred = predictor.predict(date, route)
        print(f"\n{date} | {route}")
        print(f"  Bad Weather: {pred['bad_weather_pct']:.1f}%")
        print(f"  Delay:       {pred['voyage_delay_days']:.2f} days")
        print(f"  Confidence:  {pred['confidence']:.1%}")
    
    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (TOP 5)")
    print("-" * 70)
    importance = predictor.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
        print(f"{i}. {feature:15s}: {score:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    example_predictions()
