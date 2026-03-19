import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# fake dataset
data = {
    "hr": [60, 70, 90, 110, 55, 120],
    "hrv": [0.05, 0.04, 0.02, 0.01, 0.06, 0.015],
    "label": [0, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["hr", "hrv"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")

print("Model trained and saved!")