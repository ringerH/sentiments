"""
Model training utilities
"""
import pickle
import logging
from pathlib import Path
from importlib import import_module
from sklearn.metrics import classification_report, accuracy_score
from src.config.settings import MODEL_CONFIG, MODELS_DIR
from src.data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = TextPreprocessor()
        self.best_model = None
        self.best_score = 0
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all configured models"""
        logger.info("Starting model training...")
        
        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        results = {}
        
        for model_name, model_config in MODEL_CONFIG["models"].items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Import and instantiate model
                module_path, class_name = model_config["class"].rsplit(".", 1)
                module = import_module(module_path)
                model_class = getattr(module, class_name)
                model = model_class(**model_config["params"])
                
                # Handle sparse/dense matrix requirements
                if model_name in MODEL_CONFIG["dense_models"]:
                    X_train_fit = X_train_processed.toarray()
                    X_test_fit = X_test_processed.toarray()
                else:
                    X_train_fit = X_train_processed
                    X_test_fit = X_test_processed
                
                # Train model
                model.fit(X_train_fit, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_fit)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[model_name] = model
                results[model_name] = {
                    "accuracy": accuracy,
                    "report": classification_report(y_test, y_pred, output_dict=True)
                }
                
                # Track best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model_name
                
                logger.info(f"{model_name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        logger.info(f"Best model: {self.best_model} ({self.best_score:.4f})")
        return results
    
    def save_model(self, model_name=None, output_dir=None):
        """Save trained model and preprocessor"""
        model_name = model_name or self.best_model
        output_dir = Path(output_dir) if output_dir else MODELS_DIR
        output_dir.mkdir(exist_ok=True)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Save model
        model_path = output_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.models[model_name], f)
        
        # Save preprocessor
        preprocessor_path = output_dir / "preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Preprocessor saved: {preprocessor_path}")
        
        return model_path, preprocessor_path