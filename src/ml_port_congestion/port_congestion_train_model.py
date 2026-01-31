import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the Synthetic Data
import os

# Get the project root directory (assumes script is run from anywhere)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'hybrid_training_data.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Feature Engineering (Preparing data for AI)
# We need to turn "Port Name" (Text) into Numbers (One-Hot Encoding or Categorical)
# XGBoost handles categories well, but let's use One-Hot for safety/transparency.
df_encoded = pd.get_dummies(df, columns=['Port'], drop_first=True)

# Define Features (X) and Target (y)
features = [col for col in df_encoded.columns if col not in ['Date', 'Target_Delay', 'BDI_Inertia']]
# NOTE: We REMOVE 'BDI_Inertia' from input! 
# Why? Because in real life, the user only gives us a "Scenario" (BDI). 
# The model must figure out the inertia implicitly or we feed it as a lagged feature.
# actually, let's KEEP it but rename it to 'Scenario_Trend' in the app.
features = ['BDI', 'BDI_Inertia', 'Rain_Forecast', 'Wind_Forecast', 'Fog_Risk', 'Month'] + [c for c in df_encoded.columns if 'Port_' in c]

X = df_encoded[features]
y = df_encoded['Target_Delay']

# 3. Time-Series Splitting (Crucial)
# We cannot pick random rows. We must train on Past (2020-2023) and Test on Future (2024).
split_point = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"Training on {len(X_train)} rows. Testing on {len(X_test)} rows...")

# 4. Train XGBoost
model = xgb.XGBRegressor(
    n_estimators=1000,      # Number of "Trees"
    learning_rate=0.05,     # Slow and steady learning
    max_depth=6,            # Depth of decision trees
    early_stopping_rounds=50, # Stop if not improving
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# 5. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n=== MODEL REPORT ===")
print(f"Mean Absolute Error: {mae:.2f} days")
print(f"Accuracy (R2 Score): {r2:.2f}")
print("Interpretation: On average, the model is off by +/- {:.2f} days.".format(mae))

# 6. Save the Brain
joblib.dump(model, os.path.join(MODEL_DIR, 'congestion_model.pkl'))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, 'model_features.pkl')) # Save column names to ensure match later
print("✅ Model Saved to 'models/congestion_model.pkl'")