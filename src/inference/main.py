from fastapi import FastAPI
from pydantic import BaseModel
import os
from .classifier import predict_proba

app = FastAPI()

LLM_MODEL = os.environ.get("MODEL")

class InferenceRequest(BaseModel):
    title: str
    abstract: str

class InferenceResponse(BaseModel):
    payload: dict
    prediction: int
    probability: float
    input_tokens: int
    output_tokens: int
    extracted_features: list
    llm: str


@app.post("/prediction", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    proba, extracted_features, payload = await predict_proba(request.title, request.abstract)
    pred = int(proba >= 0.5)
    input_tokens = -1
    output_tokens = -1
    return InferenceResponse(payload=payload, prediction=pred, probability=proba, input_tokens=input_tokens, output_tokens=output_tokens, extracted_features=extracted_features, llm=LLM_MODEL)
