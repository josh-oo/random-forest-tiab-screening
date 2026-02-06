from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
import os
from .llm_extractor import extract_features

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "rf_smote_model.joblib")

MODEL = load(MODEL_PATH)

class InferenceRequest(BaseModel):
    title: str
    abstract: str

class InferenceResponse(BaseModel):
    prediction: int
    probability: float
    input_tokens: int
    output_tokens: int

@app.post("/prediction", response_model=InferenceResponse)
async def predict(request: InferenceRequest):

    features, input_tokens, output_tokens = await extract_features(request.title, request.abstract)
    print(features)
    X = np.array(features).reshape(1, -1)
    proba = MODEL.predict_proba(X)[0, 1]
    pred = int(proba >= 0.35)
    return InferenceResponse(prediction=pred, probability=proba, input_tokens=input_tokens, output_tokens=output_tokens)
