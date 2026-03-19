import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

model = joblib.load(MODEL_PATH)


def predict(hr, hrv):
    pred = model.predict([[hr, hrv]])
    return int(pred[0])