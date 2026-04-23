from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Credit Card Fraud Detection")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = 'model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("Warning: Model file not found.")

class TransactionData(BaseModel):
    v_features: list[float]
    amount: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
async def predict(data: TransactionData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        features = np.array(data.v_features + [data.amount]).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        return {
            "is_fraud": bool(prediction),
            "confidence": round(float(max(probability)) * 100, 2),
            "fraud_probability": round(float(probability[1]) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
async def get_stats():
    return {
        "total_transactions": 284807,
        "fraud_cases": 492,
        "normal_cases": 284315,
        "accuracy": 99.92,
        "precision": 86.0,
        "recall": 78.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
