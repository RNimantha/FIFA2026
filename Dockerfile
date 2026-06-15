FROM python:3.11-slim

WORKDIR /app

# Install system deps for XGBoost/LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Train models at build time (needs KAGGLE_API_TOKEN build arg)
ARG KAGGLE_API_TOKEN
ENV KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN

RUN python scripts/build.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"]
