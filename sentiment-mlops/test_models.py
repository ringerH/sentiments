"""
Test script to verify trainer and predictor work together
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.trainer import ModelTrainer
from src.models.predictor import ModelPredictor, load_best_model


def test_training_pipeline():
    """Test the complete training pipeline"""
    print("Testing training pipeline...")
    
    # Load data
    loader = DataLoader()
    try:
        loader.load_data()
    except FileNotFoundError:
        print("Using sample data")
        from src.data.loader import load_sample_data
        loader.df = load_sample_data()
    
    # Clean and split data
    loader.clean_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    print("\nTraining Results:")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.4f}")
    
    # Save best model
    model_path, preprocessor_path = trainer.save_model()
    print(f"\nSaved model: {model_path}")
    
    return model_path, preprocessor_path


def test_prediction_pipeline(model_path, preprocessor_path):
    """Test the prediction pipeline"""
    print("\nTesting prediction pipeline...")
    
    # Load model
    predictor = ModelPredictor(model_path, preprocessor_path)
    
    # Test predictions
    test_texts = [
        "The stock market is looking very bullish today!",
        "FUD everywhere, this is bearish AF",
        "Neutral earnings report from the company"
    ]
    
    print("\nPrediction Results:")
    for text in test_texts:
        result = predictor.predict_single(text)
        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.4f}")
        print()


def test_load_best_model():
    """Test loading best model utility"""
    print("Testing load_best_model utility...")
    
    try:
        predictor = load_best_model()
        result = predictor.predict_single("HODL! Going to the moon!")
        print(f"Best model prediction: {result['prediction']}")
        return True
    except Exception as e:
        print(f"Error loading best model: {e}")
        return False


if __name__ == "__main__":
    try:
        # Test training
        model_path, preprocessor_path = test_training_pipeline()
        
        # Test prediction
        test_prediction_pipeline(model_path, preprocessor_path)
        
        # Test utility function
        test_load_best_model()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()