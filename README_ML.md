# Livestock Hybrid Disease Screening (LSD/FMD/ECF/CBPP)

This ML package implements a defensible hybrid system with strict scope:
- Final labels: `Normal`, `LSD`, `FMD`, `ECF`, `CBPP`
- Image model labels: `Normal`, `LSD`, `FMD` only
- ECF/CBPP support: transparent clinical rules, with explainable trigger scores

## Folder Layout

```
ml/
  configs/config.yaml
  data/
    raw/
    processed/
      images_merged/{Normal,LSD,FMD}/
      symptoms_merged.csv
      splits_manifest.csv
  src/
    utils/seed.py, io.py, viz.py, metrics.py
    data/merge_images.py
    data/clean_merge_symptoms.py
    models/image_mobilenetv2.py
    models/symptom_model.py
    models/gradcam.py
    models/rules_engine.py
    models/fusion.py
    train/train_image.py
    train/train_symptoms.py
    evaluate/eval_image.py
    evaluate/eval_symptoms.py
    evaluate/eval_hybrid.py
    export/export_savedmodel.py
    export/export_tflite.py
    infer/predict.py
  artifacts/
    image_model/
    image_model.tflite
    symptom_model.pkl
    symptom_features.json
    label_map.json
    fusion_config.json
    reports/
      image_report.txt
      symptom_report.txt
      hybrid_report.txt
      confusion_matrix_image.png
      confusion_matrix_hybrid.png
      gradcam_examples/
```

## Data Truthfulness and Scope

- The image classifier is trained and evaluated only on `Normal/LSD/FMD`.
- ECF/CBPP are never produced by the image branch.
- If ECF/CBPP are absent in symptom training rows after strict mapping/filtering, the pipeline prints warnings and defers those diagnoses to clinical rules.
- No hidden synthetic labels are introduced.

## Windows (PowerShell) End-to-End Commands

Run from repo root: `c:\cattle_disease_ml`

```powershell
python -m venv .venv_ml
.\.venv_ml\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r ml\requirements.txt
$env:PYTHONPATH = "$PWD\ml"
```

### 1) Data Engineering

1. Merge images from two Kaggle download folders (configure in `ml/configs/config.yaml` `paths.image_sources`):

```powershell
python -m src.data.merge_images
```

2. Clean and merge symptom CSVs:

```powershell
python -m src.data.clean_merge_symptoms
```

### 2) Training

```powershell
python -m src.train.train_image
python -m src.train.train_symptoms
```

### 3) Evaluation + Ablation

```powershell
python -m src.evaluate.eval_image
python -m src.evaluate.eval_symptoms
python -m src.evaluate.eval_hybrid
```

### 4) Export

```powershell
python -m src.export.export_savedmodel
python -m src.export.export_tflite
```

### 5) Smoke Test (Inference QA)

```powershell
python -m src.evaluate.smoke_test_inference
```

Output:
- `ml/artifacts/reports/smoke_test_report.json`

## Inference API Usage (`ml/src/infer/predict.py`)

Functions exposed:
- `load_image_model()`
- `load_symptom_model()`
- `preprocess_image(image_bytes)`
- `predict_image(image_bytes)`
- `predict_symptoms(symptoms_dict)`
- `predict_full(image_bytes=None, symptoms_dict=None)`

Example symptom payload:

```json
{
  "fever": 1,
  "swollen_lymph_nodes": 1,
  "eye_discharge": 1,
  "nasal_discharge": 1,
  "difficulty_breathing": 0,
  "coughing": 0,
  "chest_pain_signs": 0,
  "loss_of_appetite": 1,
  "painless_lumps": 0
}
```

Example quick test (PowerShell):

```powershell
python -c "from src.infer.predict import predict_full; import json; d={'fever':1,'swollen_lymph_nodes':1,'eye_discharge':1,'nasal_discharge':1}; print(json.dumps(predict_full(symptoms_dict=d), indent=2))"
```

## FastAPI Serving

Start API:

```powershell
$env:PYTHONPATH = "$PWD\ml"
python -m uvicorn src.infer.api:app --host 0.0.0.0 --port 8000
```

Swagger UI:
- `http://127.0.0.1:8000/docs`
- Symptom catalog endpoint: `GET /symptoms/catalog`

PowerShell examples:

1. Symptoms-only:

```powershell
$body = @{ symptoms = @{ fever = 1; swollen_lymph_nodes = 1; eye_discharge = 1; nasal_discharge = 1 } } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/symptoms" -Method Post -ContentType "application/json" -Body $body
```

2. Image-only:

```powershell
$img = "C:\path\to\cow.jpg"
curl.exe -X POST "http://127.0.0.1:8000/predict/image" -F "file=@$img"
```

3. Full hybrid (multipart with optional JSON payload + optional image):

```powershell
$payload = '{"symptoms":{"fever":1,"swollen_lymph_nodes":1,"eye_discharge":1,"nasal_discharge":1}}'
$img = "C:\path\to\cow.jpg"
curl.exe -X POST "http://127.0.0.1:8000/predict/full" --form-string "payload=$payload" -F "file=@$img"
```

## Deploy (Docker)

Build image:

```powershell
docker build -t livestock-health-api:latest .
```

Run container:

```powershell
docker run --rm -p 8000:8000 -e PYTHONPATH=/app/ml -e ML_CONFIG_PATH=/app/ml/configs/config.yaml livestock-health-api:latest
```

Or with Compose:

```powershell
docker compose up --build
```

## Deploy (Render Blueprint)

This repo includes `render.yaml` for one-click deployment on Render.

Render settings are pre-defined for TensorFlow stability:
- single worker
- constrained thread env vars
- `healthCheckPath: /health`

Use Render:
1. New -> Blueprint
2. Select this repo
3. Deploy using `render.yaml`

## Explainability

- Image explainability: Grad-CAM examples exported to `ml/artifacts/reports/gradcam_examples/`.
- Symptom explainability: top active symptom importances from RF feature importances.
- Rule explainability: triggered ECF/CBPP symptoms in `explain.rule_triggers`.
- Clinical advisories are included in `explain.clinical_advisories`.

## Fusion and Provenance

Every final response includes:
- `method`: `image_model | symptom_model | clinical_rules | hybrid`
- `probs` over all 5 final labels
- `confidence`
- `explain` payload
- recommendation flags (`retake_image`, `contact_vet_urgent`)

## Notes

- Pairing image rows and symptom rows for ablation is class-aware pseudo pairing when no shared case ID exists.
- For production, replace with true case-level linkage.
- Set image source folders in `config.yaml` to local Kaggle extraction paths containing:
  - Dataset A folders: `healthy`, `foot-and-mouth`, `lumpy`
  - Dataset B folders: `Normal Skin`, `Foot and Mouth disease`, `Lumpy Skin`
