#!/usr/bin/env python3
"""
Prediction entry point script
"""
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import load_best_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_text(text):
    """Predict sentiment for a single text"""
    try:
        predictor = load_best_model()
        result = predictor.predict_single(text)
        
        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.4f}")
        
        return result
        
    except FileNotFoundError:
        logger.error("No trained model found. Run train.py first.")
        return None
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None


def predict_batch(texts):
    """Predict sentiment for multiple texts"""
    try:
        predictor = load_best_model()
        predictions, probabilities = predictor.predict(texts)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'prediction': predictions[i],
                'confidence': None
            }
            
            if probabilities is not None:
                result['confidence'] = float(max(probabilities[i]))
            
            results.append(result)
            
            print(f"{i+1}. '{text}' -> {result['prediction']}")
            if result['confidence']:
                print(f"   Confidence: {result['confidence']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return None


def interactive_mode():
    """Interactive prediction mode"""
    print("Interactive prediction mode (type 'quit' to exit)")
    print("-" * 50)
    
    try:
        predictor = load_best_model()
        print(f"Model loaded: {predictor.model_name}")
        
        while True:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            result = predictor.predict_single(text)
            print(f"Sentiment: {result['prediction']}")
            if result['confidence']:
                print(f"Confidence: {result['confidence']:.4f}")
                
    except FileNotFoundError:
        logger.error("No trained model found. Run train.py first.")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Interactive mode failed: {str(e)}")


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Prediction')
    parser.add_argument('--text', '-t', type=str, help='Single text to predict')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--batch', '-b', nargs='+', help='Multiple texts to predict')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.text:
        predict_text(args.text)
    elif args.batch:
        predict_batch(args.batch)
    else:
        # Default: run some examples
        sample_texts = [
            "HODL! This coin is going to the moon!",
            "FUD everywhere, market crashing hard",
            "Neutral earnings report from the company",
            "Bull run continues, very bullish sentiment",
            "Bear market fears are spreading"
        ]
        
        print("Running prediction examples...")
        predict_batch(sample_texts)


if __name__ == "__main__":
    main()