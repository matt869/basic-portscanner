import os
import joblib
import logging
import pandas as pd
import numpy as np
import argparse
import json
import time
import platform
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# --- Configuration Defaults ---
DEFAULT_DATA_PATH = "scan_results.csv"
DEFAULT_MODEL_DIR = "trained_model_advanced"
MODEL_FILENAME = "port_classifier_tuned.pkl"
ENCODER_FILENAME = "label_encoder_Label.pkl"
LOG_FILE = "model_training.log"

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="Train a port classification model.")
parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH, help='Path to CSV dataset')
parser.add_argument('--output', type=str, default=DEFAULT_MODEL_DIR, help='Directory to save model')
args = parser.parse_args()

DATA_PATH = args.data
MODEL_DIR = args.output

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

logging.info("🚀 Starting port classification model training script.")

# --- Environment Info ---
logging.info("🧪 Environment Info:")
logging.info(f"Python: {platform.python_version()} | OS: {platform.system()} {platform.release()}")
logging.info(f"scikit-learn: {sklearn.__version__} | pandas: {pd.__version__} | numpy: {np.__version__}")

# --- Ensure Output Directory ---
os.makedirs(MODEL_DIR, exist_ok=True)
logging.info(f"✅ Ensured model directory: '{MODEL_DIR}'")

# --- Load Dataset ---
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"✅ Loaded data from '{DATA_PATH}' | Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"❌ File not found: {DATA_PATH}")
    exit(1)

# --- Inspect Data ---
logging.info("🔍 Preview of data:\n%s", df.head().to_string())
buf = pd.io.common.StringIO()
df.info(buf=buf)
logging.info("🔍 Data Info:\n%s", buf.getvalue())
logging.info("🔍 Label Distribution:\n%s", df['Label'].value_counts())

# --- Column Config ---
target_col = 'Label'
features = ['Port', 'Service', 'Banner']
categorical_cols = ['Service', 'Banner']
numeric_cols = ['Port']

# --- Null Handling ---
if df[features].isnull().any().any():
    logging.warning("⚠️ Null values found. Filling with '__missing__' for categoricals, -1 for numerics.")
    df[categorical_cols] = df[categorical_cols].fillna('__missing__')
    df[numeric_cols] = df[numeric_cols].fillna(-1)
else:
    logging.info("✅ No missing values detected.")

# --- Encode Target Variable ---
le_target = LabelEncoder()
df['Label_encoded'] = le_target.fit_transform(df[target_col])
joblib.dump(le_target, os.path.join(MODEL_DIR, ENCODER_FILENAME))
logging.info(f"✅ Encoded target '{target_col}' → Classes: {list(le_target.classes_)}")

# --- Define Features and Target ---
X = df[features]
y = df['Label_encoded']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logging.info("📊 Data split into train/test sets.")
logging.info(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
logging.info(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")

# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# --- Model Pipeline ---
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
logging.info("⚙️  Pipeline created: Preprocessing + RandomForestClassifier")

# --- GridSearch Parameters ---
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 20, 30],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}
logging.info(f"🎯 Hyperparameter grid:\n{param_grid}")

# --- Cross-Validated Grid Search ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)

# --- Train Model ---
logging.info("🔄 Running GridSearchCV...")
start_time = time.time()
grid_search.fit(X_train, y_train)
duration = time.time() - start_time

best_model = grid_search.best_estimator_
logging.info("✅ Grid search complete.")
logging.info(f"🏆 Best Parameters: {grid_search.best_params_}")
logging.info(f"📈 Best CV Accuracy: {grid_search.best_score_:.4f}")
logging.info(f"⏱️ Training duration: {duration:.2f} seconds")

# --- Evaluate Model ---
logging.info("\n--- 📊 Model Evaluation ---")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"✅ Test Accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=le_target.classes_)
logging.info(f"\n📄 Classification Report:\n{report}")

conf_matrix = confusion_matrix(y_test, y_pred)
logging.info(f"\n🧮 Confusion Matrix:\n{conf_matrix}")

# --- Feature Importances ---
try:
    importances = best_model.named_steps['clf'].feature_importances_
    feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    feature_importance = dict(zip(feature_names, importances))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    logging.info("📊 Feature Importances:")
    for feat, val in sorted_importance:
        logging.info(f"  {feat}: {val:.4f}")
except Exception as e:
    logging.warning(f"⚠️ Could not retrieve feature importances: {e}")

# --- Save Model ---
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
joblib.dump(best_model, model_path)
logging.info(f"💾 Saved best model to: {model_path}")

# --- Save Metadata ---
metadata = {
    "model_path": model_path,
    "encoder_path": os.path.join(MODEL_DIR, ENCODER_FILENAME),
    "best_params": grid_search.best_params_,
    "cv_score": grid_search.best_score_,
    "test_accuracy": accuracy,
    "label_classes": list(le_target.classes_),
    "features": features,
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "duration_sec": round(duration, 2)
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)
logging.info("📝 Model metadata saved.")

logging.info("🏁 All tasks complete. Model training finished successfully.")
