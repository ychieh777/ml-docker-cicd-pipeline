from fastapi import FastAPI
from predict import predict

app = FastAPI()


@app.get("/")
def root():
    return {"message": "ML Pipeline API running"}


@app.get("/predict")
def get_prediction(hr: float, hrv: float):
    result = predict(hr, hrv)

    return {
        "hr": hr,
        "hrv": hrv,
        "prediction": result
    }