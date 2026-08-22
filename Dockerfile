# Hugging Face Spaces Docker SDK Dockerfile for Recovery Decision Engine

FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and model artifacts
COPY . .

# Expose port 7860 required by Hugging Face Spaces
EXPOSE 7860

ENV PYTHONPATH=/app
ENV PORT=7860

# Run FastAPI app with Uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "src.orchestration.app:app", "--host", "0.0.0.0", "--port", "7860"]
