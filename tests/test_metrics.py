import unittest
import numpy as np
from src.utils.metrics import compute_metrics, detailed_metrics, compare_models


class TestMetrics(unittest.TestCase):
    """Unit tests for metrics utilities"""
    
    def setUp(self):
        self.y_true = ['positive', 'negative', 'neutral', 'positive', 'negative']
        self.y_pred = ['positive', 'negative', 'positive', 'positive', 'negative']
    
    def test_compute_metrics(self):
        """Test basic metrics computation"""
        metrics = compute_metrics(self.y_true, self.y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check values are between 0 and 1
        for value in metrics.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def test_detailed_metrics(self):
        """Test detailed metrics computation"""
        metrics = detailed_metrics(self.y_true, self.y_pred)
        
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('per_class', metrics)
        
        # Confusion matrix should be 2D list
        cm = metrics['confusion_matrix']
        self.assertIsInstance(cm, list)
        self.assertIsInstance(cm[0], list)
    
    def test_compare_models(self):
        """Test model comparison"""
        results = {
            'ModelA': {'accuracy': 0.85, 'f1_score': 0.83},
            'ModelB': {'accuracy': 0.92, 'f1_score': 0.90},
            'ModelC': {'accuracy': 0.78, 'f1_score': 0.76}
        }
        
        comparison = compare_models(results)
        
        self.assertIn('ranking', comparison)
        self.assertIn('best_model', comparison)
        self.assertEqual(comparison['best_model'], 'ModelB')
        
        # Check ranking order
        rankings = comparison['ranking']
        self.assertEqual(rankings[0][0], 'ModelB')  # Highest accuracy first
        self.assertEqual(rankings[-1][0], 'ModelC')  # Lowest accuracy last
