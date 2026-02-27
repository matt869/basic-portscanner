import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Optional: Gradient Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Optional: AutoML (if installed)
# from tpot import TPOTClassifier
# import autosklearn.classification

# -----------------------
# Example Data (replace with your dataset)
# -----------------------
data = {
    "Banner_Length": [22, 14, 10, 18, 15],
    "Banner_has_Apache": [1, 0, 0, 0, 0],
    "Banner_has_nginx": [0, 1, 0, 0, 0],
    "Banner_has_Microsoft": [0, 0, 0, 1, 0],
    "Port_Scaled": [0.1, 0.5, -1.2, 2.0, -0.8],
    "Service_https": [0, 1, 0, 0, 0],
    "Service_ssh": [0, 0, 1, 0, 0],
    "Port_Service_443_https": [0, 1, 0, 0, 0],
    "Port_Service_22_ssh": [0, 0, 1, 0, 0],
    "label": [0, 1, 0, 1, 0]  # Example binary labels
}

df = pd.DataFrame(data)

X = df.drop("label", axis=1)
y = df["label"]

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------
# Models to Compare
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

print("🔹 Model Evaluation with Cross-Validation")
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = np.mean(cv_scores)
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# -----------------------
# Fit Best Models
# -----------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print(f"\n✅ Best Model: {best_model_name}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------
# Ensemble (Stacking)
# -----------------------
estimators = [
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("lgbm", LGBMClassifier(random_state=42)),
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

print("\n Stacking Ensemble Results:")
print(classification_report(y_test, y_pred_stack))

# -----------------------
# (Optional) AutoML Section
# -----------------------
# Uncomment if TPOT installed
# print("\n⚡ Running TPOT AutoML...")
# tpot = TPOTClassifier(generations=5, population_size=20, cv=5, verbosity=2, random_state=42)
# tpot.fit(X_train, y_train)
# print("TPOT Best Pipeline:", tpot.fitted_pipeline_)

# Uncomment if AutoSklearn installed
# print("\n⚡ Running AutoSklearn AutoML...")
# automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=30)
# automl.fit(X_train, y_train)
# print("AutoSklearn Models:", automl.show_models())

