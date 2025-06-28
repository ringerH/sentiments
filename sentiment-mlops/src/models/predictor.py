"""
Model inference utilities
"""
import pickle
import logging
import numpy as np
from pathlib import Path
from src.config.settings import MODELS_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Load and use trained models for prediction"""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        self.model = None
        self.preprocessor = None
        self.model_name = None
        
        if model_path and preprocessor_path:
            self.load_model(model_path, preprocessor_path)
    
    def load_model(self, model_path, preprocessor_path):
        """Load trained model and preprocessor"""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            
            self.model_name = Path(model_path).stem
            logger.info(f"Model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, texts):
        """Predict sentiment for given texts"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        X_processed = self.preprocessor.transform(texts)
        
        # Handle sparse/dense requirements
        if self.model_name in MODEL_CONFIG["dense_models"]:
            X_processed = X_processed.toarray()
        
        # Predict
        predictions = self.model.predict(X_processed)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_processed)
        
        return predictions, probabilities
    
    def predict_single(self, text):
        """Predict sentiment for a single text"""
        predictions, probabilities = self.predict([text])
        
        result = {
            "text": text,
            "prediction": predictions[0],
            "confidence": None
        }
        
        if probabilities is not None:
            max_prob = np.max(probabilities[0])
            result["confidence"] = float(max_prob)
        
        return result


def load_best_model():
    """Load the best available model"""
    models_dir = Path(MODELS_DIR)
    
    # Look for saved models
    model_files = list(models_dir.glob("*.pkl"))
    model_files = [f for f in model_files if f.name != "preprocessor.pkl"]
    
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    # Use the first available model (could be improved with metadata)
    model_path = model_files[0]
    preprocessor_path = models_dir / "preprocessor.pkl"
    
    if not preprocessor_path.exists():
        raise FileNotFoundError("Preprocessor not found")
    
    predictor = ModelPredictor()
    predictor.load_model(model_path, preprocessor_path)
    
    return predictor