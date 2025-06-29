"""
Configuration settings for the sentiment analysis MLOps pipeline
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data settings
DATA_CONFIG = {
    "input_file": "data.csv",
    "text_column": "Sentence",
    "target_column": "Sentiment",
    "test_size": 0.4,
    "random_state": 42
}

# Model settings
MODEL_CONFIG = {
    "models": {
        "RandomForest": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "params": {"n_estimators": 100, "random_state": 42}
        },
        "LogisticRegression": {
            "class": "sklearn.linear_model.LogisticRegression",
            "params": {"max_iter": 1000, "random_state": 42}
        },
        "LinearSVC": {
            "class": "sklearn.svm.LinearSVC",
            "params": {"max_iter": 10000, "random_state": 42}
        },
        "GaussianNB": {
            "class": "sklearn.naive_bayes.GaussianNB",
            "params": {}
        },
        "MultinomialNB": {
            "class": "sklearn.naive_bayes.MultinomialNB",
            "params": {}
        }
    },
    "dense_models": ["GaussianNB", "MultinomialNB"]
}

# Preprocessing settings
PREPROCESSING_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "batch_size": 50,
    "tfidf_params": {
        "min_df": 1,
        "max_df": 1.0
    }
}

# MLflow settings
MLFLOW_CONFIG = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "experiment_name": "sentiment-analysis",
    "artifact_path": "model",
    "registered_model_name": "sentiment-classifier"
}

# Kubernetes settings
K8S_CONFIG = {
    "namespace": "mlops",
    "training_job_name": "sentiment-training",
    "serving_deployment_name": "sentiment-serving",
    "model_storage_path": "/mnt/models"
}

# Monitoring settings
MONITORING_CONFIG = {
    "prometheus_port": 8000,
    "metrics_path": "/metrics",
    "health_check_path": "/health"
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": {
            "filename": str(LOGS_DIR / "app.log"),
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        },
        "console": {
            "stream": "ext://sys.stdout"
        }
    }
}

# Environment-specific overrides
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    MLFLOW_CONFIG["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    LOGGING_CONFIG["level"] = "WARNING"
elif ENVIRONMENT == "testing":
    DATA_CONFIG["test_size"] = 0.1
    MODEL_CONFIG["models"]["RandomForest"]["params"]["n_estimators"] = 10