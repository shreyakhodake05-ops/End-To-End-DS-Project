from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

# ✅ JSON input schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "API चालू आहे 🚀"}

@app.post("/predict")
def predict(data: InputData):
    features = data.features

    # ✅ fix length
    if len(features) < 188:
        features = features + [0]*(188 - len(features))

    arr = np.array(features).reshape(1, -1)
    prediction = model.predict(arr)

    return {"prediction": float(prediction[0])}