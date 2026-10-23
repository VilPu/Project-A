## Unzip the Archive.zip or Download used datasets from ARMD to datasets folder:
- `microbiology_cultures_cohort.csv`
- `microbiology_cultures_ward_info.csv`

## Install Required Libraries

```bash
pip install -r requirements.txt
```

## Run the Data Pipeline

```bash
python data_pipeline.py
```

## Run the Models

**Model A – Regression**

```bash
python model_a_regression.py
```

**Model B – LightGBM**

```bash
python model_b_lightgbm.py
```

## Target Folder

The `target/` folder contains:

- `antibiotic_clean.csv` – Cleaned dataset.
- `ml_candidates.csv` – Preliminary dataset processed for the models.

## Results Folder

The `results/` folder contains:

- `Calibration plots`
- `dataset_profile.txt` – Dataset information, such as the number of rows, etc.

## Other information
Other results such as model accuracy, precision, etc. are printed to console when the model script has terminated.
