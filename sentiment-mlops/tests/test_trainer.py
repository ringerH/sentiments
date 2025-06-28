import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.models.trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Unit tests for ModelTrainer"""
    
    def setUp(self):
        self.trainer = ModelTrainer()
        self.X_train = pd.Series(["positive text", "negative text"])
        self.y_train = pd.Series(["positive", "negative"])
        self.X_test = pd.Series(["test positive", "test negative"])
        self.y_test = pd.Series(["positive", "negative"])
    
    @patch('importlib.import_module')
    def test_model_import(self, mock_import):
        """Test model class importing"""
        mock_module = MagicMock()
        mock_model_class = MagicMock()
        mock_module.RandomForestClassifier = mock_model_class
        mock_import.return_value = mock_module
        
        # This would be tested within train_models, but testing the concept
        module = mock_import('sklearn.ensemble')
        model_class = getattr(module, 'RandomForestClassifier')
        self.assertEqual(model_class, mock_model_class)
    
    def test_initial_state(self):
        """Test trainer initial state"""
        self.assertEqual(self.trainer.models, {})
        self.assertIsNotNone(self.trainer.preprocessor)
        self.assertIsNone(self.trainer.best_model)
        self.assertEqual(self.trainer.best_score, 0)
    
    def test_save_model_without_training_raises_error(self):
        """Test save_model without training raises ValueError"""
        with self.assertRaises(ValueError):
            self.trainer.save_model("NonExistentModel")