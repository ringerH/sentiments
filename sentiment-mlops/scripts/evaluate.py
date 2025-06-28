#!/usr/bin/env python3
"""
Evaluation entry point script
"""
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader, load_sample_data
from src.models.predictor import load_best_model
from src.utils.metrics import detailed_metrics, print_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_on_test_data():
    """Evaluate model on test data"""
    logger.info("Loading test data...")
    
    # Load data
    loader = DataLoader()
    try:
        loader.load_data()
        logger.info("Loaded real data")
    except FileNotFoundError:
        logger.info("Using sample data for evaluation")
        loader.df = load_sample_data()
    
    # Clean and split
    loader.clean_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    
    logger.info(f"Evaluating on {len(X_test)} test samples")
    
    # Load model and predict
    try:
        predictor = load_best_model()
        predictions, probabilities = predictor.predict(X_test.tolist())
        
        # Calculate metrics
        metrics = detailed_metrics(y_test.tolist(), predictions.tolist())
        print_metrics(metrics, predictor.model_name)
        
        # Show some examples
        print("\nSample Predictions:")
        print("-" * 50)
        for i in range(min(5, len(X_test))):
            text = X_test.iloc[i]
            true_label = y_test.iloc[i]
            pred_label = predictions[i]
            
            status = "✓" if true_label == pred_label else "✗"
            print(f"{status} '{text[:50]}...'")
            print(f"  True: {true_label}, Predicted: {pred_label}")
        
        return metrics
        
    except FileNotFoundError:
        logger.error("No trained model found. Run train.py first.")
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return None


def evaluate_custom_data(texts, labels):
    """Evaluate model on custom data"""
    if len(texts) != len(labels):
        logger.error("Number of texts and labels must match")
        return None
    
    try:
        predictor = load_best_model()
        predictions, _ = predictor.predict(texts)
        
        metrics = detailed_metrics(labels, predictions.tolist())
        print_metrics(metrics, f"{predictor.model_name} (Custom Data)")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Custom evaluation failed: {str(e)}")
        return None


def quick_evaluation():
    """Quick evaluation with financial examples"""
    print("Quick financial sentiment evaluation...")
    
    test_cases = [
        ("HODL! Bitcoin going to the moon!", "positive"),
        ("FUD everywhere, crypto market crashing", "negative"),
        ("Neutral trading volume in the market", "neutral"),
        ("Bull run continues, very bullish", "positive"),
        ("Bear market fears spreading", "negative"),
        ("Company reported average earnings", "neutral")
    ]
    
    texts = [case[0] for case in test_cases]
    labels = [case[1] for case in test_cases]
    
    return evaluate_custom_data(texts, labels)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--test-data', action='store_true', help='Evaluate on test split')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation with examples')
    
    args = parser.parse_args()
    
    if args.test_data:
        evaluate_on_test_data()
    elif args.quick:
        quick_evaluation()
    else:
        # Default: run both
        print("Running full evaluation...")
        evaluate_on_test_data()
        print("\n" + "="*60 + "\n")
        quick_evaluation()


if __name__ == "__main__":
    main()