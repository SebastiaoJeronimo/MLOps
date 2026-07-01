"""FastAPI service for car price prediction."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import AliasChoices, BaseModel, Field

from app import history
from app.model_loader import load_champion_model
from app.preprocess import preprocess_raw_cars

_model = None


class CarFeatures(BaseModel):
    """Raw car listing fields (snake_case or original CSV names)."""

    car_id: int | None = Field(
        default=None, validation_alias=AliasChoices("car_id", "carID")
    )
    brand: str = Field(validation_alias=AliasChoices("brand", "Brand"))
    model: str
    year: float | int
    transmission: str
    mileage: float
    fuel_type: str = Field(validation_alias=AliasChoices("fuel_type", "fuelType"))
    tax: float
    mpg: float
    engine_size: float = Field(
        validation_alias=AliasChoices("engine_size", "engineSize")
    )
    previous_owners: float | None = Field(
        default=None,
        validation_alias=AliasChoices("previous_owners", "previousOwners"),
    )
    paint_quality_pct: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "paint_quality_pct", "paintQuality%", "paintQuality"
        ),
    )
    has_damage: int | float | None = Field(
        default=0, validation_alias=AliasChoices("has_damage", "hasDamage")
    )

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    cars: list[CarFeatures]


class PredictResponse(BaseModel):
    predictions: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    mlflow.set_tracking_uri(tracking_uri)
    _model = load_champion_model()
    yield
    _model = None


app = FastAPI(
    title="Car Price Prediction API",
    description="Predict used car prices using MLflow Champion model.",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


def _car_to_row(car: CarFeatures) -> dict[str, Any]:
    row = car.model_dump()
    if row.get("car_id") is None:
        row["car_id"] = 0
    return {k: v for k, v in row.items() if v is not None}


@app.get("/predictions/history")
def list_prediction_history(limit: int = Query(default=50, ge=1, le=500)):
    return {"count": limit, "records": history.list_records(limit=limit)}


@app.get("/predictions/history/{record_id}")
def get_prediction_history(record_id: str):
    record = history.get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


def _run_prediction(cars: list[CarFeatures]) -> list[float]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not cars:
        raise HTTPException(status_code=400, detail="At least one car is required")
    raw_df = pd.DataFrame([_car_to_row(car) for car in cars])
    features = preprocess_raw_cars(raw_df)
    preds = _model.predict(features)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    return [float(p) for p in preds]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        predictions = _run_prediction(req.cars)
        history.append_record(
            {"cars": [_car_to_row(car) for car in req.cars]},
            {"predictions": predictions},
        )
        return PredictResponse(predictions=predictions)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/single")
def predict_single(car: CarFeatures):
    try:
        request_payload = _car_to_row(car)
        prediction = _run_prediction([car])[0]
        response_payload = {"prediction": prediction}
        history.append_record(request_payload, response_payload)
        return response_payload
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
