"""
Test predict.py and evaluate.py scripts
"""
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def run_command(cmd):
    """Run a command and capture output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def test_predict_script():
    """Test the predict.py script"""
    print("Testing predict.py script...")
    
    # Test single prediction
    print("\n1. Testing single text prediction:")
    success, stdout, stderr = run_command('python scripts/predict.py --text "HODL to the moon!"')
    
    if success:
        print("✓ Single prediction works")
        print(stdout.strip())
    else:
        print("✗ Single prediction failed")
        print(stderr)
    
    # Test batch prediction
    print("\n2. Testing batch prediction:")
    success, stdout, stderr = run_command('python scripts/predict.py --batch "Bull market!" "Bear market crash"')
    
    if success:
        print("✓ Batch prediction works")
        print(stdout.strip())
    else:
        print("✗ Batch prediction failed")
        print(stderr)
    
    # Test default mode
    print("\n3. Testing default examples:")
    success, stdout, stderr = run_command('python scripts/predict.py')
    
    if success:
        print("✓ Default examples work")
        print(stdout.strip())
    else:
        print("✗ Default examples failed")
        print(stderr)


def test_evaluate_script():
    """Test the evaluate.py script"""
    print("\nTesting evaluate.py script...")
    
    # Test quick evaluation
    print("\n1. Testing quick evaluation:")
    success, stdout, stderr = run_command('python scripts/evaluate.py --quick')
    
    if success:
        print("✓ Quick evaluation works")
        print(stdout.strip())
    else:
        print("✗ Quick evaluation failed")
        print(stderr)
    
    # Test test data evaluation
    print("\n2. Testing test data evaluation:")
    success, stdout, stderr = run_command('python scripts/evaluate.py --test-data')
    
    if success:
        print("✓ Test data evaluation works")
        print(stdout.strip())
    else:
        print("✗ Test data evaluation failed")
        print(stderr)
    
    # Test default mode
    print("\n3. Testing default evaluation:")
    success, stdout, stderr = run_command('python scripts/evaluate.py')
    
    if success:
        print("✓ Default evaluation works")
    else:
        print("✗ Default evaluation failed")
        print(stderr)


def test_direct_imports():
    """Test importing the scripts directly"""
    print("\nTesting direct imports...")
    
    try:
        # Test predict functions
        sys.path.append('scripts')
        from predict import predict_text, predict_batch
        
        result = predict_text("Test financial sentiment")
        if result:
            print("✓ Direct predict_text works")
        else:
            print("✗ Direct predict_text failed")
        
        results = predict_batch(["Bull market", "Bear sentiment"])
        if results:
            print("✓ Direct predict_batch works")
        else:
            print("✗ Direct predict_batch failed")
        
        # Test evaluate functions  
        from evaluate import quick_evaluation
        metrics = quick_evaluation()
        if metrics:
            print("✓ Direct quick_evaluation works")
        else:
            print("✗ Direct quick_evaluation failed")
            
    except Exception as e:
        print(f"✗ Direct import test failed: {e}")


def check_model_exists():
    """Check if trained model exists"""
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    model_files = [f for f in model_files if f.name != "preprocessor.pkl"]
    
    has_model = len(model_files) > 0 and (models_dir / "preprocessor.pkl").exists()
    
    if has_model:
        print("✓ Trained model found")
        return True
    else:
        print("✗ No trained model found")
        print("  Run: python scripts/train.py")
        return False


if __name__ == "__main__":
    print("Testing predict and evaluate scripts...\n")
    
    # Check if model exists
    model_exists = check_model_exists()
    
    if not model_exists:
        print("\n📝 Please train a model first:")
        print("   python scripts/train.py")
        print("\nThen run this test again.")
        sys.exit(1)
    
    # Test scripts
    test_predict_script()
    test_evaluate_script()
    test_direct_imports()
    
    print("\n✅ All script tests completed!")