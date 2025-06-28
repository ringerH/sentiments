#!/usr/bin/env python3
"""
Training entry point script
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader, load_sample_data
from src.models.trainer import ModelTrainer
from src.utils.metrics import detailed_metrics, compare_models, print_comparison

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    logger.info("Starting training pipeline...")
    
    # Load data
    loader = DataLoader()
    try:
        df = loader.load_data()
        logger.info("Loaded real data")
    except FileNotFoundError:
        logger.info("Data file not found, using sample data")
        loader.df = load_sample_data()
        df = loader.df
    
    # Basic info
    info = loader.basic_info()
    logger.info(f"Dataset shape: {info['shape']}")
    logger.info(f"Target distribution: {info['target_distribution']}")
    
    # Clean and split
    loader.clean_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Compare results
    comparison = compare_models(results)
    print_comparison(comparison)
    
    # Save best model
    best_model = comparison['best_model']
    model_path, preprocessor_path = trainer.save_model(best_model)
    
    logger.info(f"Training completed. Best model: {best_model}")
    logger.info(f"Model saved to: {model_path}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)