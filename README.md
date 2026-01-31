# Cargill Datathon Freight Calculator Chatbot

A comprehensive maritime freight calculation and weather prediction system for dry bulk shipping operations.

## 🌊 Weather Prediction System

### Overview
This project includes an advanced **Machine Learning Weather Prediction System** that forecasts weather impact on maritime shipping operations across 8 major global shipping routes. The system predicts:

- **Speed Reduction %**: Weather-induced vessel speed reduction (10-40%)
- **Voyage Delays**: Additional transit time due to weather (0.1-3.5 days)
- **Fuel Impact**: Weather-related fuel consumption changes
- **Route-Specific Risk**: Geographic and seasonal weather patterns

### Key Features

#### 🚢 **8 Global Shipping Routes**
- **Australia → China/Korea**: Southern Ocean winter storms
- **Brazil → China/Malaysia**: Atlantic-Pacific weather systems  
- **Indonesia → India**: Monsoon zone operations
- **Canada → China**: North Pacific conditions
- **South/West Africa → China/India**: Cape weather and Indian Ocean patterns

#### 🤖 **Two-Stage ML Architecture**
1. **Stage 1**: Weather Impact Prediction (Speed reduction %)
2. **Stage 2**: Voyage Delay Prediction (Days delay)
- **Algorithm**: Gradient Boosting Regressor (200 estimators each stage)
- **Performance**: 30.7% R² for delays, ±0.7 day accuracy
- **Features**: Route, seasonal patterns, temporal encoding

#### 📊 **Synthetic Training Data**
- **Dataset Size**: 2,000 observations across 2 years
- **Geographic Coverage**: 8 shipping corridors (80%+ of global dry bulk traffic)
- **Weather Realism**: Physics-based generation using maritime engineering principles
- **Seasonal Modeling**: Winter/spring/summer/autumn patterns with realistic variance

### Technical Implementation

#### **Model Performance**
```
Weather Model (Stage 1):
- R² Score: 17.3% (appropriate for weather prediction)
- RMSE: 8.96% speed reduction
- MAE: 7.29% speed reduction

Delay Model (Stage 2):  
- R² Score: 30.7% (strong practical performance)
- RMSE: 0.69 days
- MAE: 0.53 days (±13 hours accuracy)
```

#### **Feature Importance**
1. **Weather Prediction** (46.1%): Primary delay driver
2. **Route Classification** (43.9%): Geographic factors  
3. **Temporal Features** (6.0%): Seasonal patterns

#### **Data Generation Strategy**

The synthetic dataset was created using `generate_expanded_data.py` with sophisticated modeling:

**Route-Distance Correlation:**
```python
routes_info = {
    'australia_china': {'distance_nm': 5400, 'multiplier': 0.8},
    'brazil_china': {'distance_nm': 9200, 'multiplier': 1.2},
    # ... 8 routes total
}
```

**Seasonal Weather Patterns:**
```python
seasonal_patterns = {
    'winter': {'base_weather': 28, 'variance': 12},
    'summer': {'base_weather': 18, 'variance': 8},
    # Natural seasonal variations
}
```

**Physics-Based Weather Modeling:**
- **Sinusoidal seasonal cycles** for natural weather patterns
- **Correlated maritime variables** (wave height ↔ sea state ↔ speed reduction)
- **Distance-weather interactions** (longer routes amplify weather delays)
- **Operational constraints** (realistic bounds: 10-40% speed reduction, max 3.5 day delays)

### Business Value

#### 🎯 **Freight Calculator Integration**
- **Enhanced Cost Accuracy**: Weather-adjusted voyage time estimates
- **Risk Assessment**: Confidence intervals for planning uncertainty
- **Route Optimization**: Data-driven shipping corridor selection
- **Fuel Planning**: Weather impact on consumption and costs

#### 📈 **Key Benefits**
- **±13 hour delay accuracy**: Actionable precision for logistics planning  
- **75% confidence estimates**: Risk management for contingency planning
- **Route-specific insights**: Australia-China (3.3 days) vs Indonesia-India (0.5 days)
- **Seasonal intelligence**: Winter weather impact 56% higher than summer

### Usage Examples

#### **Basic Weather Prediction**
```python
import sys
sys.path.append('src')
from ml_weather_predictor import WeatherPredictorML

predictor = WeatherPredictorML()
prediction = predictor.predict("2026-02-15", "brazil_china")

print(f"Bad Weather: {prediction['bad_weather_pct']:.1f}%")
print(f"Delay: {prediction['voyage_delay_days']:.2f} days") 
print(f"Confidence: {prediction['confidence']:.1%}")
```

#### **Training with Visualization**
```python
predictor.train('data/synthetic_weather_data_expanded.csv', show_plots=True)
# Generates training_performance.png and feature_importance.png in outputs/
```

### Files Structure

```
src/ml_weather_prediction/
├── ml_weather_predictor.py          # Main ML model implementation
├── generate_expanded_data.py        # Synthetic data generation
├── weather_prediction_docs.md       # Technical documentation
└── models/                          # Trained model files (.pkl)

data/
└── synthetic_weather_data_expanded.csv  # Training dataset (2,000 obs)

outputs/
├── training_performance.png         # 6-panel training visualization
├── feature_importance.png           # Feature analysis charts
└── ml_weather_prediction_report.md  # Comprehensive technical report
```

---

## Managing Python Libraries and Conda Environment

### 1. Updating Requirements / Libraries in Your Conda Environment

To install a new library (e.g., pandas) in your active conda environment:

```
conda install pandas
```

Or, using pip:

```
pip install pandas
```

After installing new packages, update your `environment.yml` so others can reproduce your environment:

```
conda env export > environment.yml
```

Commit and push the updated `environment.yml` to your repository.

### 2. Creating and Running Your Own Environment Using `environment.yml`

To create a new conda environment from the provided `environment.yml` file:

```
conda env create -f environment.yml
```

To activate the environment:

```
conda activate <env_name>
```
Replace `<env_name>` with the name specified in the `environment.yml` file (usually near the top under `name:`).

To update an existing environment after changes to `environment.yml`:

```
conda env update -f environment.yml --prune
```

---
For more details, see the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).