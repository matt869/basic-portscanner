import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np

# -----------------------
# Example dataset
# -----------------------
data = {
    "Banner": ["Apache/2.4.41 (Ubuntu)", "nginx/1.18.0", "OpenSSH_7.9", "Microsoft-IIS/10.0", None],
    "Port": [80, 443, 22, 8080, 21],
    "Service": ["http", "https", "ssh", "http", "ftp"]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df, "\n")

# -----------------------
# Custom Feature Functions
# -----------------------
def banner_features(X):
    """Extract length + keyword presence from Banner."""
    X = pd.DataFrame(X, columns=["Banner"]).fillna("")
    keywords = ["Apache", "nginx", "Microsoft"]
    
    features = pd.DataFrame()
    features["Banner_Length"] = X["Banner"].apply(len)
    
    for kw in keywords:
        features[f"Banner_has_{kw}"] = X["Banner"].str.contains(kw, case=False).astype(int)
    
    return features


def interaction_features(X):
    """Create interaction feature Port_Service."""
    X = pd.DataFrame(X, columns=["Port", "Service"])
    features = pd.DataFrame()
    features["Port_Service"] = X["Port"].astype(str) + "_" + X["Service"].astype(str)
    return features


# -----------------------
# Pipelines
# -----------------------
# Banner pipeline
banner_pipeline = Pipeline([
    ("features", FunctionTransformer(banner_features, validate=False))
])

# Numeric pipeline (Port)
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Service categorical pipeline
service_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

# Interaction pipeline
interaction_pipeline = Pipeline([
    ("features", FunctionTransformer(interaction_features, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

# -----------------------
# Column Transformer
# -----------------------
preprocessor = ColumnTransformer(transformers=[
    ("banner", banner_pipeline, ["Banner"]),
    ("numeric", numeric_pipeline, ["Port"]),
    ("service", service_pipeline, ["Service"]),
    ("interaction", interaction_pipeline, ["Port", "Service"])
])

# -----------------------
# Apply Feature Engineering
# -----------------------
X_transformed = preprocessor.fit_transform(df)

# Get feature names
feature_names = (
    preprocessor.named_transformers_["banner"].named_steps["features"].func(df[["Banner"]]).columns.tolist()
    + ["Port_Scaled"]
    + preprocessor.named_transformers_["service"].named_steps["onehot"].get_feature_names_out(["Service"]).tolist()
    + preprocessor.named_transformers_["interaction"].named_steps["onehot"].get_feature_names_out(["Port_Service"]).tolist()
)

# Convert back to DataFrame
df_features = pd.DataFrame(X_transformed, columns=feature_names)

print("Powerful Feature-Engineered Data:")
print(df_features)
