"""
api/main.py — Colon Cancer Prediction API
"""

import os, json, pickle, logging
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

app = FastAPI(
    title="Colon Cancer Prediction API",
    description="API de prédiction du cancer du côlon basée sur l'expression génomique.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

scaler = selector = model = metadata = None

@app.on_event("startup")
async def load_artifacts():
    global scaler, selector, model, metadata
    d = Path(MODEL_DIR)
    missing = [f for f in ["scaler.pkl","selector.pkl","model.pkl","metadata.json"] if not (d/f).exists()]
    if missing:
        raise RuntimeError(f"Artefacts manquants : {missing}. Lancez d'abord ml_service.")
    with open(d/"scaler.pkl","rb")   as f: scaler   = pickle.load(f)
    with open(d/"selector.pkl","rb") as f: selector = pickle.load(f)
    with open(d/"model.pkl","rb")    as f: model    = pickle.load(f)
    with open(d/"metadata.json")     as f: metadata = json.load(f)
    logger.info(f"✅ Modèle chargé — Test acc: {metadata['metrics']['test_accuracy']} | AUC: {metadata['metrics']['test_auc_roc']}")

class PredictionRequest(BaseModel):
    gene_expression: list[float] = Field(..., description="2001 valeurs d'expression génomique")

class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: int
    probability_normal: float
    probability_abnormal: float
    confidence: str
    top_genes_used: list[str]

class GeneInfo(BaseModel):
    rank: int
    gene: str
    f_score: float

class ModelInfoResponse(BaseModel):
    strategy: str
    top_n: int
    metrics: dict
    model_params: dict

@app.get("/", tags=["Health"])
async def root():
    return {"service": "Colon Cancer Prediction API", "status": "running",
            "endpoints": ["/predict", "/top-genes", "/model-info", "/health", "/docs"]}

@app.get("/health", tags=["Health"])
async def health():
    ready = all(x is not None for x in [scaler, selector, model, metadata])
    return {"status": "ready" if ready else "not_ready", "model_loaded": ready}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    expected = len(metadata["feature_names_all"])
    if len(request.gene_expression) != expected:
        raise HTTPException(status_code=422, detail=f"Attendu {expected} features, reçu {len(request.gene_expression)}.")
    X = np.array(request.gene_expression).reshape(1, -1)
    X = selector.transform(scaler.transform(X))
    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    conf  = "High" if max(proba) > 0.80 else "Medium" if max(proba) > 0.60 else "Low"
    return PredictionResponse(
        prediction="Abnormal" if pred==1 else "Normal",
        prediction_label=pred,
        probability_normal=round(float(proba[0]),4),
        probability_abnormal=round(float(proba[1]),4),
        confidence=conf,
        top_genes_used=metadata["selected_feature_names"]
    )

@app.get("/top-genes", response_model=list[GeneInfo], tags=["Model"])
async def top_genes(n: int = Query(default=20, ge=1, le=100)):
    if metadata is None:
        raise HTTPException(status_code=503, detail="Métadonnées non chargées.")
    return [GeneInfo(**g) for g in metadata["top_genes"][:n]]

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Métadonnées non chargées.")
    return ModelInfoResponse(
        strategy=metadata["model_params"]["strategy"],
        top_n=metadata["top_n"],
        metrics=metadata["metrics"],
        model_params=metadata["model_params"]
    )
