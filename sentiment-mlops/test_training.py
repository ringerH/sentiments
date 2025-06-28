"""
Simple test for metrics and training script
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import compute_metrics, detailed_metrics, compare_models, print_metrics
from src.models.predictor import load_best_model


def test_metrics():
    """Test metrics functions"""
    print("Testing metrics functions...")
    
    # Sample predictions
    y_true = ['positive', 'negative', 'neutral', 'positive', 'negative']
    y_pred = ['positive', 'negative', 'positive', 'positive', 'negative']
    
    # Basic metrics
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Model")
    
    # Detailed metrics
    detailed = detailed_metrics(y_true, y_pred)
    print(f"Per-class F1 scores: {detailed['per_class']['macro avg']['f1-score']:.4f}")
    
    # Model comparison
    fake_results = {
        'ModelA': {'accuracy': 0.85, 'f1_score': 0.83},
        'ModelB': {'accuracy': 0.92, 'f1_score': 0.90},
        'ModelC': {'accuracy': 0.78, 'f1_score': 0.76}
    }
    
    comparison = compare_models(fake_results)
    print("\nFake model comparison:")
    for i, (name, scores) in enumerate(comparison['ranking'], 1):
        print(f"{i}. {name}: {scores['accuracy']:.4f}")


def test_trained_model():
    """Test if we can load and use a trained model"""
    print("\nTesting trained model (if available)...")
    
    try:
        predictor = load_best_model()
        
        test_texts = [
            "HODL! Going to the moon!",
            "FUD everywhere, bear market incoming",
            "Neutral market sentiment today"
        ]
        
        for text in test_texts:
            result = predictor.predict_single(text)
            print(f"'{text}' -> {result['prediction']}")
        
        return True
        
    except FileNotFoundError:
        print("No trained model found. Run train.py first.")
        return False
    except Exception as e:
        print(f"Error testing model: {e}")
        return False


if __name__ == "__main__":
    print("Running training pipeline tests...\n")
    
    # Test metrics
    test_metrics()
    
    # Test trained model if available
    model_available = test_trained_model()
    
    if not model_available:
        print("\n📝 To test with a real model:")
        print("1. Run: python scripts/train.py")
        print("2. Then run this test again")
    
    print("\n✅ Metrics tests completed!")