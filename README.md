# 🧬 Colon Cancer ML Pipeline

Pipeline complet de classification du cancer du côlon basé sur l'expression génomique.

## Architecture

```
colon_cancer_project/
├── docker-compose.yml
├── data/
│   └── colon_cancer_dataset.csv        ← dataset (62 échantillons, 2001 gènes)
├── ml_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train.py                        ← entraînement + sauvegarde modèle
└── api_service/
    ├── Dockerfile
    ├── requirements.txt
    └── main.py                         ← FastAPI REST API
```

## Stratégie ML

| Problème | Solution choisie | Raison |
|---|---|---|
| Overfitting (train=1.0, test=1.0) | `StandardScaler` + `SelectKBest(k=20)` | Réduction 2001→20 features interprétables |
| Déséquilibre de classes (40/22) | `class_weight='balanced'` | Pénalise davantage les erreurs sur la classe minoritaire |
| Haute dimensionnalité | ANOVA F-test (SelectKBest) | Sélectionne les gènes statistiquement significatifs |

**Résultats :**
- Train Accuracy : **0.9184**
- Test Accuracy  : **0.7692**
- CV-5 Accuracy  : **0.8846 (±0.0855)**
- AUC-ROC        : **0.8500**

## Lancement

### Prérequis
- Docker ≥ 24.0
- Docker Compose ≥ 2.20

### 1. Démarrage complet (entraînement + API)

```bash
# Cloner/placer le projet
cd colon_cancer_project/

# Lancer les deux services
docker compose up --build

# En arrière-plan
docker compose up --build -d
```

`ml_service` s'exécute en premier, entraîne le modèle, sauvegarde les artefacts dans le volume partagé `colon_models_volume`, puis termine.

`api_service` attend que `ml_service` ait terminé avec succès avant de démarrer.

### 2. Re-entraîner uniquement

```bash
docker compose run --rm ml_service
```

### 3. Arrêt

```bash
docker compose down
# Supprimer aussi le volume (repart de zéro)
docker compose down -v
```

---

## Endpoints de l'API

L'API est disponible sur **http://localhost:8000**

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Accueil + liste des endpoints |
| `/health` | GET | Statut du service et du modèle |
| `/predict` | POST | Prédiction sur un échantillon |
| `/top-genes?n=20` | GET | Top N gènes importants |
| `/model-info` | GET | Métriques et paramètres du modèle |
| `/docs` | GET | Documentation interactive (Swagger UI) |
| `/redoc` | GET | Documentation ReDoc |

---

## Exemples cURL

### Health check
```bash
curl http://localhost:8000/health
```

### Top 10 gènes importants
```bash
curl "http://localhost:8000/top-genes?n=10"
```

### Infos du modèle
```bash
curl http://localhost:8000/model-info
```

### Prédiction (exemple avec des valeurs aléatoires)
```bash
# Générer un vecteur de 2001 valeurs (exemple Python)
python3 -c "
import json, random
vals = [random.uniform(1000, 15000) for _ in range(2001)]
print(json.dumps({'gene_expression': vals}))
" > sample.json

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @sample.json
```

**Réponse exemple :**
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

## Variables d'environnement

### ml_service
| Variable | Défaut | Description |
|---|---|---|
| `DATA_PATH` | `/data/colon_cancer_dataset.csv` | Chemin du dataset |
| `MODEL_DIR` | `/models` | Dossier de sauvegarde des artefacts |
| `TOP_N` | `20` | Nombre de gènes sélectionnés |

### api_service
| Variable | Défaut | Description |
|---|---|---|
| `MODEL_DIR` | `/models` | Dossier des artefacts (volume partagé) |

---

## Artefacts générés

Après l'entraînement, le volume `colon_models_volume` contient :

| Fichier | Contenu |
|---|---|
| `scaler.pkl` | `StandardScaler` fitté sur les données d'entraînement |
| `selector.pkl` | `SelectKBest(f_classif, k=20)` fitté |
| `model.pkl` | `LogisticRegression` entraîné |
| `metadata.json` | Top gènes, métriques, paramètres du modèle |

---

## Top 20 Gènes Identifiés

| Rang | Gène | F-score |
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
