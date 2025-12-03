import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# paths and setup
os.makedirs("results", exist_ok=True)

data_path = "results/beat_features.csv"
df = pd.read_csv(data_path)

# define: features and labels
# original label: "normal", "ventricular", "other"
# building a binary label: normal vs abnormal (ventricular + other)
df["is_abnormal"] = (df["label"] != "normal").astype(int)

feature_cols = [
    "RR_ms",
    "PR_peak_ms",
    "QT_peak_ms",
    "QTc_ms",
    "P_duration_ms",
    "T_duration_ms",
]

X = df[feature_cols].values
y = df["is_abnormal"].values


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# define models
logreg_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # mitigate class imbalance
            ),
        ),
    ]
)

rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)

models = {
    "logistic_regression": logreg_pipeline,
    "random_forest": rf_clf,
}

metrics_out = {}


# train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    # predictions and probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback if model does not support predict_proba
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    if y_proba is not None and len(np.unique(y_test)) > 1:
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = None
    else:
        auc = None

    metrics_out[name] = {
        "accuracy": float(acc),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    # print key metrics to terminal
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.3f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

# save metrics
with open("results/model_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=4)

# Save feature importances for Random Forest
if "random_forest" in models:
    rf = models["random_forest"]
    importances = rf.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    fi_df.to_csv("results/rf_feature_importance.csv", index=False)

    print("\nRandom Forest feature importances:")
    print(fi_df.to_string(index=False))
    print("\nSaved to results/rf_feature_importance.csv")

print("\nAll model metrics saved to results/model_metrics.json")
