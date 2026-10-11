# Dockerfile for the FastAPI backend.
# Ollama itself is NOT containerized here -- it's expected to be running on
# the host machine and reachable at host.docker.internal.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app_streamlit.py .

ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
