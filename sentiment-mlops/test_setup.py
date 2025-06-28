#!/usr/bin/env python3
"""
Quick test to verify the project setup works
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # Test imports
    from src.config.settings import DATA_CONFIG, MODEL_CONFIG
    from src.data.loader import DataLoader, load_sample_data
    from src.data.preprocessor import TextPreprocessor, AcronymExpander
    
    print("✅ All imports successful!")
    
    # Test data loading
    sample_data = load_sample_data()
    print(f"✅ Sample data created: {sample_data.shape}")
    
    # Test data loader
    loader = DataLoader()
    loader.df = sample_data
    info = loader.basic_info()
    print(f"✅ Data loader works: {info['shape']}")
    
    # Test preprocessor
    preprocessor = TextPreprocessor()
    X_features = preprocessor.fit_transform(sample_data['Sentence'])
    print(f"✅ Text preprocessing works: {X_features.shape}")
    
    print("\n🎉 Setup is complete and working!")
    print("\nNext steps:")
    print("1. Create the training module (src/models/trainer.py)")
    print("2. Create the main training script (scripts/train.py)")
    print("3. Set up Docker containers")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all files are in the correct locations")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check the setup and try again")