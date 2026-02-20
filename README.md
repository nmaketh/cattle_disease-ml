# Cattle Symptom Dataset Build Workflow (VS Code, Python 3.10+)

This project includes a CLI pipeline to:
- Download two Kaggle datasets (`kagglehub` first, Kaggle API fallback).
- Auto-discover CSV files.
- Filter to cattle rows (if species column exists).
- Keep only diseases in scope: `Normal`, `LSD`, `FMD`, `CBPP`.
- Normalize disease variants.
- Convert both dataset formats into one shared one-hot symptom feature space.
- Merge, deduplicate, shuffle, and save outputs.

## Project Structure

- `scripts/build_symptom_dataset.py`
- `scripts/utils_symptoms.py`
- `data/raw/`
- `data/processed/`
- `logs/`
- `requirements.txt`

## 1) Open in VS Code

1. Open folder `c:\cattle_disease_ml` in VS Code.
2. Open terminal: `Terminal > New Terminal`.
3. Ensure Python 3.10+ is selected (`Ctrl+Shift+P` > `Python: Select Interpreter`).

## 2) Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Kaggle authentication setup (for fallback API)

The script tries `kagglehub` first. If it fails, it automatically uses Kaggle API.

### Option A: `kaggle.json` file

- Windows: `C:\Users\<USER>\.kaggle\kaggle.json`
- Mac/Linux: `~/.kaggle/kaggle.json`

Recommended permissions on Mac/Linux:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Option B: Environment variables (optional)

PowerShell (Windows):

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
```

Bash (Mac/Linux):

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

## 4) Run the dataset build

```powershell
python scripts/build_symptom_dataset.py --ds1 captaingee/livestock-disease-diagnosis-dataset --ds2 researcher1548/livestock-symptoms-and-diseases
```

## 5) Outputs

After success, check:

- `data/processed/merged_symptoms_onehot.csv`
- `data/processed/merged_symptoms_core.csv`
- `data/processed/label_mapping.json`
- `logs/merge_report.txt`

## 6) Multimodal app model pipeline (Normal/LSD/FMD)

For image-based app integration, use a separate Python 3.11 environment:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-image-model.txt
```

Build merged `Normal` / `LSD` / `FMD` images (using notebook Kaggle sources):

```powershell
python scripts/build_lsd_fmd_image_dataset.py --clean --out-dir data/images_multiclass
```

Train multimodal model (image + symptom one-hot features) and export app artifacts:

```powershell
python scripts/train_multimodal_normal_lsd_fmd.py --image-dir data/images_multiclass --symptom-csv data/processed/merged_symptoms_onehot.csv --output-dir artifacts/multimodal_normal_lsd_fmd --epochs 5
```

Saved artifacts:

- `artifacts/multimodal_normal_lsd_fmd/best_multimodal_model.keras`
- `artifacts/multimodal_normal_lsd_fmd/final_multimodal_model.keras`
- `artifacts/multimodal_normal_lsd_fmd/multimodal_model.tflite`
- `artifacts/multimodal_normal_lsd_fmd/labels.json`
- `artifacts/multimodal_normal_lsd_fmd/symptom_columns.json`
- `artifacts/multimodal_normal_lsd_fmd/training_summary.json`

Optional image-only baseline:

```powershell
python scripts/train_lsd_fmd_image_model.py --data-dir data/images_lsd_fmd --output-dir artifacts/lsd_fmd_image_model --epochs 5
```

## Notes on validation and cleaning

- Fails with a clear error if Disease column is missing.
- Prints unique diseases before/after normalization.
- Handles mixed casing, spaces, punctuation in labels.
- Drops likely non-symptom fields (id, location, date, owner, etc.).
- Asserts datasets are non-empty after filtering.
- Supports:
  - Wide binary symptom columns + disease label
  - Symptom list columns (`Symptom_1..Symptom_n`) + disease label
