FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:8080
ENV REGISTERED_MODEL_NAME=car_price_model
ENV CHAMPION_ALIAS=Champion

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
