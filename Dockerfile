# Stage 1: Training
FROM python:3.11-slim AS trainer

ENV DATA="data/"

WORKDIR /app


# Install only training dependencies
COPY requirements-training.txt ./
RUN pip install --no-cache-dir -r requirements-training.txt

# Copy training code and data
COPY src/ ./src/
COPY data/ ./data/

# Run training script to produce the model artifact
RUN python src/training/train.py

# Stage 2: Inference
FROM python:3.11-slim AS inference

ENV DATA="data/"

WORKDIR /app

# Install only necessary dependencies for inference
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and inference code
COPY --from=trainer /app/rf_smote_model.joblib ./rf_smote_model.joblib
COPY src/inference/ ./src/inference/
COPY data/llm_questions.csv ./data/llm_questions.csv

# Set entrypoint for inference (example: replace with your actual inference script)
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
