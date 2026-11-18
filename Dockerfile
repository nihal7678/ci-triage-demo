# Optional: run the triage API anywhere with Docker.
#   docker build -t ci-triage .        docker run -p 8000:8000 ci-triage
FROM python:3.11-slim
WORKDIR /app
COPY model/ model/
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r model/requirements.lock.txt -r requirements-serve.txt
COPY api.py triage_model.py triage_features.py ./
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
