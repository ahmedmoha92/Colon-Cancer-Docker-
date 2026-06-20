# 🧬 Colon Cancer ML Pipeline

Complete pipeline for colon cancer classification based on genomic expression.

## Architecture

```
colon_cancer_project/
├── docker-compose.yml
├── data/
│   └── colon_cancer_dataset.csv        ← dataset (62 samples, 2001 genes)
├── ml_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py                        ← training + model saving
└── api_service/
    ├── Dockerfile
    ├── requirements.txt
    └── main.py                         ← FastAPI REST API
```

## ML Strategy

| Problem | Chosen Solution | Reason |
|---|---|---|
| Overfitting (train=1.0, test=1.0) | `StandardScaler` + `SelectKBest(k=20)` | Reduction from 2001→20 interpretable features |
| Class Imbalance (40/22) | `class_weight='balanced'` | Penalizes errors on the minority class more heavily |
| High Dimensionality | ANOVA F-test (SelectKBest) | Selects statistically significant genes |

**Results:**
- Train Accuracy : **0.9184**
- Test Accuracy  : **0.7692**
- CV-5 Accuracy  : **0.8846 (±0.0855)**
- AUC-ROC        : **0.8500**

## Quickstart

### Prerequisites
- Docker ≥ 24.0
- Docker Compose ≥ 2.20

### 1. Full Startup (Training + API)

```bash
# Navigate to the project directory
cd colon_cancer_project/

# Launch both services
docker compose up --build

# Or run in the background
docker compose up --build -d
```

`ml_service` runs first, trains the model, saves the artifacts in the shared `colon_models_volume` volume, and then terminates.

`api_service` waits for `ml_service` to complete successfully before starting.

### 2. Retrain Only

```bash
docker compose run --rm ml_service
```

### 3. Stopping the Services

```bash
docker compose down
# To also delete the volume (start fresh next time)
docker compose down -v
```

---

## API Endpoints

The API is available at **http://localhost:8000**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home + list of available endpoints |
| `/health` | GET | Service and model status |
| `/predict` | POST | Predict on a single sample |
| `/top-genes?n=20` | GET | Top N most important genes |
| `/model-info` | GET | Model parameters and metrics |
| `/docs` | GET | Interactive documentation (Swagger UI) |
| `/redoc` | GET | ReDoc documentation |

---

## cURL Examples

### Health check
```bash
curl http://localhost:8000/health
```

### Top 10 important genes
```bash
curl "http://localhost:8000/top-genes?n=10"
```

### Model info
```bash
curl http://localhost:8000/model-info
```

### Prediction (example with random values)
```bash
# Generate a vector of 2001 values (Python example)
python3 -c "
import json, random
vals = [random.uniform(1000, 15000) for _ in range(2001)]
print(json.dumps({'gene_expression': vals}))
" > sample.json

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @sample.json
```

**Example response:**
```json
{
  "prediction": "Abnormal",
  "prediction_label": 1,
  "probability_normal": 0.2341,
  "probability_abnormal": 0.7659,
  "confidence": "Medium",
  "top_genes_used": ["M63391", "M76378__", "R87126", ...]
}
```

---

## Environment Variables

### ml_service
| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `/data/colon_cancer_dataset.csv` | Dataset path |
| `MODEL_DIR` | `/models` | Directory to save artifacts |
| `TOP_N` | `20` | Number of genes to select |

### api_service
| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `/models` | Directory for artifacts (shared volume) |

---

## Generated Artifacts

After training, the `colon_models_volume` volume contains:

| File | Content |
|---|---|
| `scaler.pkl` | `StandardScaler` fitted on training data |
| `selector.pkl` | `SelectKBest(f_classif, k=20)` fitted |
| `model.pkl` | Trained `LogisticRegression` |
| `metadata.json` | Top genes, metrics, and model parameters |

---

## Top 20 Identified Genes

| Rank | Gene | F-score |
|---|---|---|
| 1 | M63391 | 39.81 |
| 2 | M76378__ | 33.15 |
| 3 | R87126 | 32.02 |
| 4 | J02854 | 31.76 |
| 5 | M76378 | 30.95 |
| 6 | M76378_ | 29.64 |
| 7 | Z50753 | 25.34 |
| 8 | T92451 | 24.81 |
| 9 | U25138 | 20.54 |
| 10 | H08393 | 19.44 |
