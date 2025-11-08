from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from prophet import Prophet
import os

# ==================== CONFIGURATION ====================
MODEL_PATH = r"C:\Users\Ajwa\OneDrive\Desktop\fyp_backend\forecast_dfs.pkl"   # path to your saved models file

THRESHOLDS = {
    'current': {'min': 0.3, 'max': 10.0, 'unit': 'Amps', 'severity': 'HIGH'},
    'current1': {'min': 0.3, 'max': 8.0, 'unit': 'Amps', 'severity': 'HIGH'},
    'current2': {'min': 0.3, 'max': 8.0, 'unit': 'Amps', 'severity': 'HIGH'},
    'temperature': {'min': 15, 'max': 45, 'unit': '°C', 'severity': 'HIGH'},
    'humidity': {'min': 30, 'max': 150, 'unit': '% RH', 'severity': 'MEDIUM'},
    'airquality': {'min': 700, 'max': 1500, 'unit': 'PPM', 'severity': 'MEDIUM'},
    'mq6': {'min': 0, 'max': 1000, 'unit': 'PPM', 'severity': 'HIGH'},
    'waterflow_rate': {'min': 0.2, 'max': 15, 'unit': 'L/min', 'severity': 'MEDIUM'},
    'soilmoisture': {'min': 1000, 'max': 3300, 'unit': '%', 'severity': 'LOW'}
}

# ==================== FIREBASE INIT ====================
if not firebase_admin._apps:
    cred = credentials.Certificate(r'C:\Users\Ajwa\OneDrive\Desktop\fyp_backend\serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://maintenance-e2c07-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# ==================== LOAD TRAINED MODELS ====================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

print("🔹 Loading trained models...")
models = joblib.load(MODEL_PATH)
print(f"✅ {len(models)} models loaded successfully.")

# ==================== FASTAPI APP ====================
app = FastAPI(title="AI Forecast API", description="Forecast multiple sensors using pretrained Prophet models")

# ==================== REQUEST MODEL ====================
class ForecastRequest(BaseModel):
    sensor_name: str
    hours_ahead: int = 1

# ==================== FORECAST ENDPOINT ====================
@app.post("/forecast")
def forecast_sensor(request: ForecastRequest):
    sensor_name = request.sensor_name
    hours_ahead = request.hours_ahead

    # Validate sensor
    if sensor_name not in models:
        return {"error": f"Model for '{sensor_name}' not found. Available sensors: {list(models.keys())}"}

    model = models[sensor_name]
    print(f"🔮 Forecasting for sensor: {sensor_name}")

    # Create future dataframe
    future = model.make_future_dataframe(periods=hours_ahead, freq='H')
    forecast = model.predict(future)

    # Extract last few predictions
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(hours_ahead)

    # Check thresholds
    min_t = THRESHOLDS.get(sensor_name, {}).get('min', None)
    max_t = THRESHOLDS.get(sensor_name, {}).get('max', None)
    unit = THRESHOLDS.get(sensor_name, {}).get('unit', '')

    violations = []
    if min_t is not None and max_t is not None:
        for _, row in result.iterrows():
            if row['yhat'] < min_t or row['yhat'] > max_t:
                violations.append({
                    "timestamp": str(row['ds']),
                    "predicted_value": round(row['yhat'], 2),
                    "status": "BELOW MIN" if row['yhat'] < min_t else "ABOVE MAX",
                    "range": f"{min_t}-{max_t} {unit}"
                })

    return {
        "sensor": sensor_name,
        "forecast": result.to_dict('records'),
        "violations": violations,
        "unit": unit,
        "status": "ok"
    }

# ==================== FORECAST ALL SENSORS ====================
# @app.get("/forecast_all")
# def forecast_all():
#     predictions = {}
#     for sensor_name, model in models.items():
#         future = model.make_future_dataframe(periods=1, freq='H')
#         forecast = model.predict(future)
#         value = float(forecast['yhat'].iloc[-1])
#         predictions[sensor_name] = round(value, 2)
#     return {"timestamp": str(datetime.now()), "predictions": predictions}

@app.get("/forecast_all")
def forecast_all():
    try:
        predictions = {}
        
        # Check if models dictionary exists and has items
        if not models or not isinstance(models, dict):
            return {"error": "No models found"}, 500
            
        for sensor_name, model in models.items():
            # Validate model
            if not hasattr(model, 'make_future_dataframe'):
                predictions[sensor_name] = "Invalid model"
                continue
                
            try:
                future = model.make_future_dataframe(periods=1, freq='H')
                
                # Check if future dataframe is created properly
                if future.empty:
                    predictions[sensor_name] = "No future data"
                    continue
                    
                forecast = model.predict(future)
                
                # Check if forecast has yhat column
                if 'yhat' not in forecast.columns or forecast.empty:
                    predictions[sensor_name] = "No prediction data"
                    continue
                    
                value = float(forecast['yhat'].iloc[-1])
                predictions[sensor_name] = round(value, 2)
                
            except Exception as model_error:
                predictions[sensor_name] = f"Model error: {str(model_error)}"
                continue
                
        return {"timestamp": str(datetime.now()), "predictions": predictions}
        
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}, 500
# ==================== ROOT ====================
@app.get("/")
def root():
    return {"message": "Welcome to the AI Forecast API! Use /forecast or /forecast_all"}




