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


class WeatherPredictorML:
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or Path(__file__).parent / "models"
        Path(self.model_dir).mkdir(exist_ok=True)
        
        self.gb_weather_model = None
        self.gb_combined_delay_model = None
        
        self.scaler = StandardScaler()
        self.route_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        
        self.feature_names = None
        self.routes = [
            'australia_china',
            'brazil_china',
            'south_africa_china',
            'indonesia_india',
            'canada_china',
            'west_africa_india',
            'australia_south_korea',
            'brazil_malaysia'
        ]
        self.seasons = ['winter', 'spring', 'summer', 'autumn', 'monsoon']
        
    def _load_weather_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        df['route_encoded'] = self.route_encoder.fit_transform(df['route'])
        df['season_encoded'] = self.season_encoder.fit_transform(df['season'])
        
        df['wave_quartile'] = pd.qcut(df['significant_wave_height_m'], 
                                       q=4, labels=False, duplicates='drop')
        df['wind_quartile'] = pd.qcut(df['wind_speed_knots'], 
                                       q=4, labels=False, duplicates='drop')
        
        return df
    
    def _create_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        features = [
            'day_of_year', 'month', 'week_of_year',
            'sin_day', 'cos_day', 'sin_month', 'cos_month',
            'route_encoded', 'season_encoded',
            'wave_quartile', 'wind_quartile'
        ]
        
        self.feature_names = features
        return df[features]
    
    def train(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
        df = self._load_weather_data(csv_path)
        
        df = self._engineer_features(df)
        
        X = self._create_feature_set(df)
        y_delay = df['voyage_delay_days'].values
        y_weather = df['speed_reduction_pct'].values
        
        print(f"Delay: mean={y_delay.mean():.2f} days, std={y_delay.std():.2f}")
        print(f"Weather: mean={y_weather.mean():.2f}%, std={y_weather.std():.2f}%")
        
        X_scaled = self.scaler.fit_transform(X)
        
        (X_train, X_test, y_delay_train, y_delay_test, y_weather_train, y_weather_test) = train_test_split(
            X_scaled, y_delay, y_weather, test_size=test_size, random_state=random_state
        )
        
        print("\nGradient Boosting (Weather)")
        self.gb_weather_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=random_state, verbose=0
        )
        self.gb_weather_model.fit(X_train, y_weather_train)
        
        y_weather_pred = self.gb_weather_model.predict(X_test)
        r2_weather = r2_score(y_weather_test, y_weather_pred)
        rmse_weather = np.sqrt(mean_squared_error(y_weather_test, y_weather_pred))
        mae_weather = mean_absolute_error(y_weather_test, y_weather_pred)
        
        print(f"   R2: {r2_weather:.4f}, RMSE: {rmse_weather:.4f}%, MAE: {mae_weather:.4f}%")
        
        y_weather_train_pred = self.gb_weather_model.predict(X_train)
        y_weather_test_pred = self.gb_weather_model.predict(X_test)
        
        X_combined_train = np.hstack([X_train, y_weather_train_pred.reshape(-1, 1)])
        X_combined_test = np.hstack([X_test, y_weather_test_pred.reshape(-1, 1)])
        
        print("\nGradient Boosting Delay")
        self.gb_combined_delay_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=random_state, verbose=0
        )
        self.gb_combined_delay_model.fit(X_combined_train, y_delay_train)
        
        y_delay_pred = self.gb_combined_delay_model.predict(X_combined_test)
        r2_delay = r2_score(y_delay_test, y_delay_pred)
        rmse_delay = np.sqrt(mean_squared_error(y_delay_test, y_delay_pred))
        mae_delay = mean_absolute_error(y_delay_test, y_delay_pred)
        
        print("\n")
        print(f"   R2: {r2_delay:.4f}, RMSE: {rmse_delay:.4f} days, MAE: {mae_delay:.4f} days")
        
        print("\nModels saved to", self.model_dir)
        
        self._save_models()
    
    def predict(self, date: str, route: str) -> Dict[str, float]:
        if self.gb_weather_model is None or self.gb_combined_delay_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        if route not in self.routes:
            raise ValueError(f"Route must be one of {self.routes}")
        
        date_obj = pd.to_datetime(date)
        month = date_obj.month
        season = self._get_season(month)
        
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
        
        X = np.array([[features_dict[name] for name in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        weather = float(self.gb_weather_model.predict(X_scaled)[0])
        X_combined = np.hstack([X_scaled, np.array([[weather]])])
        delay = float(self.gb_combined_delay_model.predict(X_combined)[0])
        
        return {
            'bad_weather_pct': max(0, min(100, weather)),
            'voyage_delay_days': max(0, delay)
        }
    
    def predict_batch(self, dates: list, route: str) -> pd.DataFrame:
        results = []
        for date in dates:
            pred = self.predict(date, route)
            pred['date'] = date
            pred['route'] = route
            results.append(pred)
        
        return pd.DataFrame(results)
    
    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8, 9]:
            return 'summer'
        else:
            return 'autumn'
    
    def _save_models(self):
        model_path = Path(self.model_dir)
        model_path.mkdir(exist_ok=True)
        
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
            pickle.dump(self.feature_names, f)
    
    def load_models(self):
        model_path = Path(self.model_dir)
        
        try:
            with open(model_path / 'gb_weather_model.pkl', 'rb') as f:
                self.gb_weather_model = pickle.load(f)
            
            with open(model_path / 'gb_combined_delay_model.pkl', 'rb') as f:
                self.gb_combined_delay_model = pickle.load(f)
            
            with open(model_path / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(model_path / 'route_encoder.pkl', 'rb') as f:
                self.route_encoder = pickle.load(f)
            
            with open(model_path / 'season_encoder.pkl', 'rb') as f:
                self.season_encoder = pickle.load(f)
            
            with open(model_path / 'feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            print("Models loaded successfully")
            return True
        
        except FileNotFoundError:
            print("No saved models found.")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        if self.gb_combined_delay_model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.gb_combined_delay_model.feature_importances_
        feature_importance = dict(zip(self.feature_names + ['weather_pct'], 
                                     list(importance[:-1]) + [importance[-1]]))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))


def example_predictions():
    print("\n")
    
    predictor = WeatherPredictorML()
    
    if not predictor.load_models():
        print("\nTraining new models")
        data_path = Path(__file__).parent.parent / "data" / "synthetic_weather_data_expanded.csv"
        predictor.train(str(data_path))
    
    test_cases = [
        ("2026-02-15", "brazil_china"),
        ("2026-03-10", "australia_china"),
        ("2026-04-02", "south_africa_china"),
        ("2025-01-15", "canada_china"),
        ("2025-07-01", "indonesia_india"),
    ]
    
    for date, route in test_cases:
        pred = predictor.predict(date, route)
        print(f"\n{date} | {route}")
        print(f"  Bad Weather: {pred['bad_weather_pct']:.1f}%")
        print(f"  Delay:       {pred['voyage_delay_days']:.2f} days")
    
    print("\n")
    print("FEATURE IMPORTANCE (TOP 5)\n")
    importance = predictor.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
        print(f"{i}. {feature:15s}: {score:.4f}")
    


if __name__ == "__main__":
    example_predictions()
