"""LK09 - Containerized model inference with PostgreSQL persistence.

Model loading is intentionally performed at startup from the LK08 DagsHub
MLflow Registry, with a Staging-only guard and no Production promotion.
"""

import math
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, field_validator

from train import FEATURES

MODEL_NAME = os.getenv("MODEL_NAME", "Jambi-Wind-Residual-RandomForest")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging-candidate")


def database_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=5,
    )


class ForecastRequest(BaseModel):
    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, values):
        expected = set(FEATURES)
        received = set(values)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(f"Feature mismatch: missing={missing}, extra={extra}")

        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("All features must be finite")
        if values["lead_hours"] not in (6, 12, 18, 24):
            raise ValueError("Unsupported forecast horizon")
        if values["forecast_wind_speed_10m"] < 0:
            raise ValueError("Wind speed cannot be negative")
        return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    with database_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_history (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                raw_wind_ms DOUBLE PRECISION NOT NULL,
                predicted_residual_ms DOUBLE PRECISION NOT NULL,
                corrected_wind_ms DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL,
                input_features JSONB NOT NULL
            )
        """)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    if version.current_stage != "Staging":
        raise RuntimeError("The selected registered model is not in Staging")

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    app.state.model = mlflow.pyfunc.load_model(model_uri)
    app.state.model_version = str(version.version)
    app.state.model_uri = model_uri
    print(f"[PASS] Loaded registered model {model_uri} version {version.version}", flush=True)
    yield


app = FastAPI(
    title="Jambi Wind Correction - LK09",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    try:
        with database_connection() as conn:
            conn.execute("SELECT 1").fetchone()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {
        "status": "healthy",
        "database": "connected",
        "model": "loaded",
        "model_version": app.state.model_version,
    }


@app.get("/model")
def model_info():
    return {
        "model_uri": app.state.model_uri,
        "model_version": app.state.model_version,
        "model_alias": MODEL_ALIAS,
        "source": "DagsHub MLflow Model Registry",
        "operational_approval": "not_approved",
    }


@app.post("/predict")
def predict(payload: ForecastRequest):
    frame = pd.DataFrame(
        [{name: payload.features[name] for name in FEATURES}]
    ).astype("float64")

    prediction = np.asarray(app.state.model.predict(frame), dtype=float).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction).all():
        raise HTTPException(status_code=500, detail="Invalid model prediction")

    raw = float(payload.features["forecast_wind_speed_10m"])
    residual = float(prediction[0])
    corrected = max(0.0, raw + residual)

    with database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO inference_history (
                    raw_wind_ms, predicted_residual_ms,
                    corrected_wind_ms, model_version, input_features
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                raw, residual, corrected, app.state.model_version,
                Jsonb(payload.features),
            ))
            record_id = cursor.fetchone()[0]

    return {
        "record_id": record_id,
        "raw_wind_ms": raw,
        "predicted_residual_ms": residual,
        "corrected_wind_ms": corrected,
        "model_version": app.state.model_version,
        "stored_in_postgresql": True,
        "academic_simulation": True,
    }


@app.get("/history")
def history():
    with database_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""
                SELECT id, created_at, raw_wind_ms,
                       predicted_residual_ms, corrected_wind_ms, model_version
                FROM inference_history
                ORDER BY id DESC
                LIMIT 20
            """)
            rows = cursor.fetchall()
    return {"count": len(rows), "records": rows}
