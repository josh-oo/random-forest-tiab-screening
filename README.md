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
