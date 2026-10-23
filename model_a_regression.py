import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibrationDisplay
from pathlib import Path

Path("results").mkdir(exist_ok=True)

# load data
df = pd.read_csv("target/antibiotic_clean.csv")

# features and target
X = df[[
    "ordering_mode",
    "culture_description",
    "antibiotic",
    "hosp_ward_IP",
    "hosp_ward_OP",
    "hosp_ward_ER",
    "hosp_ward_ICU"    
    ]]
y = df["target_susceptible"]

# split by patient (80/20)
splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train, test = next(splitter.split(X, y, groups=df["anon_id"]))


encoder = ColumnTransformer(
    [("categories", OneHotEncoder(handle_unknown="ignore"),
      ["ordering_mode", "culture_description", "antibiotic"])],
    remainder="passthrough"
)

# train model
model = make_pipeline(encoder, LogisticRegression(max_iter=1000))
model.fit(X.iloc[train], y.iloc[train])

# evaluation
predictions = model.predict(X.iloc[test])

print("Accuracy:", accuracy_score(y.iloc[test], predictions))
print("\nClassification Report:")
print(classification_report(
    y.iloc[test], predictions,
    labels=[0, 1],
    target_names=["Resistant", "Susceptible"],
    zero_division=0
))

print("\n\nConfusion Matrix:")
print(confusion_matrix(y.iloc[test], predictions, labels=[0, 1]))

probabilities = model.predict_proba(X.iloc[test])[:, 1]

print("Balanced accuracy:",
      balanced_accuracy_score(y.iloc[test], predictions))

print("AUROC:",
      roc_auc_score(y.iloc[test], probabilities))

susceptible_auprc = average_precision_score(
    y.iloc[test], probabilities
)

resistant_auprc = average_precision_score(
    1 - y.iloc[test], 1 - probabilities
)

macro_auprc = (susceptible_auprc + resistant_auprc) / 2

print("Susceptible AUPRC:", susceptible_auprc)
print("Resistant AUPRC:", resistant_auprc)
print("Macro AUPRC:", macro_auprc)

print("Brier score:",
      brier_score_loss(y.iloc[test], probabilities))

CalibrationDisplay.from_predictions(
    y.iloc[test],
    probabilities,
    n_bins=10
)

plt.title("Logistic Regression Calibration")
plt.savefig("results/logistic_regression_calibration.png")
plt.close()
