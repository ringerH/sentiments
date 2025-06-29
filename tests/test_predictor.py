import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.models.predictor import ModelPredictor


class TestModelPredictor(unittest.TestCase):
    """Unit tests for ModelPredictor"""
    
    def setUp(self):
        self.predictor = ModelPredictor()
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model(self, mock_open, mock_pickle):
        """Test model loading"""
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        mock_pickle.side_effect = [mock_model, mock_preprocessor]
        
        self.predictor.load_model("model.pkl", "preprocessor.pkl")
        
        self.assertEqual(self.predictor.model, mock_model)
        self.assertEqual(self.predictor.preprocessor, mock_preprocessor)
        self.assertEqual(mock_open.call_count, 2)
    
    def test_predict_without_model_raises_error(self):
        """Test predict without loaded model raises ValueError"""
        with self.assertRaises(ValueError):
            self.predictor.predict(["test"])
    
    def test_predict_single_text_conversion(self):
        """Test single text is converted to list"""
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        mock_model.predict.return_value = np.array(['positive'])
        mock_preprocessor.transform.return_value = MagicMock()
        
        self.predictor.model = mock_model
        self.predictor.preprocessor = mock_preprocessor
        self.predictor.model_name = "TestModel"
        
        result = self.predictor.predict("test text")
        mock_preprocessor.transform.assert_called_with(["test text"])
    
    def test_predict_single_result_format(self):
        """Test predict_single returns correct format"""
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        mock_model.predict.return_value = np.array(['positive'])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_preprocessor.transform.return_value = MagicMock()
        
        self.predictor.model = mock_model
        self.predictor.preprocessor = mock_preprocessor
        self.predictor.model_name = "TestModel"
        
        result = self.predictor.predict_single("test")
        
        self.assertIn('text', result)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['prediction'], 'positive')
        self.assertAlmostEqual(result['confidence'], 0.9)