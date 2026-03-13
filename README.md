# TIAB Screening — Simple User Guide

**What this does**

- **Quick summary:** This tool reads a paper title and abstract and gives a simple yes/no screening recommendation (and a confidence score) for whether the paper matches the project's inclusion criteria.

Getting started (easiest — recommended)

1. Install Docker on your computer if it's not installed. Follow Docker's installer for macOS from https://www.docker.com/get-started.
2. In the project folder create a file named `.env` with one line containing your OpenAI API key (replace with your key):

```bash
OPENAI_API_KEY=sk-...
```

3. From the project folder, start the application with Docker Compose:

```bash
docker compose up --build
```

4. Once it starts, open your browser to:

		http://localhost:8000/docs

	 This opens a simple web page where you can try the prediction endpoint by pasting a title and abstract and clicking "Execute".

Quick example (copy/paste)

Use curl from your terminal to send a single title + abstract to the service (replace text as needed):

```bash
curl -X POST "http://localhost:8000/prediction" \
	-H "Content-Type: application/json" \
	-d '{"title": "Example title", "abstract": "Short abstract text here."}'
```

You will get a JSON response like:

```json
{
	"prediction": 1,
	"probability": 0.72,
	"input_tokens": 123,
	"output_tokens": 45
}
```

- `prediction`: 1 means the model recommends including the paper; 0 means not recommended.
- `probability`: how confident the model is (0.00–1.00 scale).

Using the app without Docker (optional)

If you prefer to run the app directly with Python, you will need Python 3.10+ and to install the requirements. From the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export MODEL_PATH=training/rf_smote_model.joblib
uvicorn src.inference.main:app --host 0.0.0.0 --port 8000 --reload
```

Where to find the model and files

- The trained model used for predictions is `training/rf_smote_model.joblib` inside the project.
- Configuration for Docker and environment variables is in `docker-compose.yml`.

If something goes wrong

- Check Docker's logs in the terminal where you ran `docker-compose up` for error messages.
- If running with Python, error messages appear in the terminal where you started `uvicorn`.

Need help or next steps?

- If you want, I can:
	- Help you test the service with example abstracts.
	- Add a very small web form that non-technical users can open in a browser and paste titles/abstracts into.

Results GPT-5 Mini

full:

Fixed Threshold (Endpoint):
recall:  0.9439252336448598
precision:  0.643312101910828
specificity:  0.9463087248322147

Best threshold: 0.25
Recall: 0.96
Precision: 0.60
Specificity: 0.93

Results Qwen/Qwen3-14B :

full:

Fixed Threshold (Endpoint):
recall:  0.8598130841121495
precision:  0.7666666666666667
specificity:  0.9731543624161074

(Best threshold: 0.50)
Recall: 0.79
Precision: 0.84
Accuracy: 0.97
Specificity: 0.98

Results medgemma-27-text-it:

full:

Fixed Threshold (Endpoint):
recall:  0.8785046728971962
precision:  0.8034188034188035
specificity:  0.9779482262703739

Best threshold: 0.10
Recall: 0.95
Precision: 0.48
Accuracy: 0.90
Specificity: 0.89

Results gpt-oss-120b:

full:

Fixed Threshold (Endpoint):
recall:  0.9065420560747663
precision:  0.7950819672131147
specificity:  0.9760306807286673

Best threshold: 0.05
Recall: 0.93
Precision: 0.42
Accuracy: 0.87
Specificity: 0.87

PARTIAL TREE:

GPT-5 Mini (Reasoning low)

Fixed Threshold (Endpoint):
recall:  0.9252336448598131
precision:  0.678082191780822
specificity:  0.9549376797698945

Best threshold: 0.05
Recall: 0.94
Precision: 0.25
Accuracy: 0.74
Specificity: 0.71

GPT-5 Mini (Reasoning low + only one response)

Fixed Threshold (Endpoint):
recall:  0.9065420560747663
precision:  0.7238805970149254
specificity:  0.9645254074784276

Best threshold: 0.05
Recall: 0.94
Precision: 0.24
Accuracy: 0.72
Specificity: 0.70


GPT-5 Mini (Reasoning low + UNSURE handling)

Fixed Threshold (Endpoint):
recall:  0.9345794392523364
precision:  0.6289308176100629
specificity:  0.9434324065196549

Best threshold: 0.15
Recall: 0.97
Precision: 0.42
Specificity: 0.86


GPT-5

Fixed Threshold (Endpoint):
recall:  0.8867924528301887
precision:  0.4017094017094017
specificity:  0.8659003831417624

Best threshold: 0.05
Recall: 0.93
Precision: 0.20
Specificity: 0.61


GPT-5 including exclude

recall:  0.9439252336448598
precision:  0.7062937062937062
specificity:  0.959731543624161

Best threshold: 0.10
Recall: 0.95
Precision: 0.40
Specificity: 0.86
