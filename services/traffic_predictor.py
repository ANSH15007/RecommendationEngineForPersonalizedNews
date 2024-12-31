from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

class TrafficPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        
    def predict_traffic(self, location: tuple, time: datetime) -> float:
        features = self._extract_features(location, time)
        prediction = self.model.predict([features])[0]
        return prediction
    
    def _extract_features(self, location: tuple, time: datetime) -> np.array:
        hour = time.hour
        day_of_week = time.weekday()
        month = time.month
        
        return np.array([
            location[0],
            location[1],
            hour,
            day_of_week,
            month
        ])