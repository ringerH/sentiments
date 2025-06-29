import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.loader import DataLoader, load_sample_data


class TestDataLoader(unittest.TestCase):
    """Unit tests for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.sample_data = load_sample_data()
    
    def test_load_sample_data(self):
        """Test sample data generation"""
        data = load_sample_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('Sentence', data.columns)
        self.assertIn('Sentiment', data.columns)
        self.assertGreater(len(data), 0)
    
    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading"""
        mock_read_csv.return_value = self.sample_data
        result = self.loader.load_data()
        self.assertIsInstance(result, pd.DataFrame)
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test FileNotFoundError handling"""
        mock_read_csv.side_effect = FileNotFoundError()
        with self.assertRaises(FileNotFoundError):
            self.loader.load_data()
    
    def test_basic_info(self):
        """Test basic_info method"""
        self.loader.df = self.sample_data
        info = self.loader.basic_info()
        
        self.assertIn('shape', info)
        self.assertIn('columns', info)
        self.assertIn('target_distribution', info)
        self.assertEqual(info['shape'], self.sample_data.shape)
    
    def test_clean_data(self):
        """Test data cleaning"""
        # Add duplicate and null data
        dirty_data = self.sample_data.copy()
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[0:1]], ignore_index=True)
        dirty_data.loc[len(dirty_data)] = [None, 'positive']
        
        self.loader.df = dirty_data
        cleaned = self.loader.clean_data()
        
        # Should remove duplicates and nulls
        self.assertEqual(len(cleaned), len(self.sample_data))
        self.assertFalse(cleaned.isnull().any().any())
    
    def test_split_data(self):
        """Test data splitting"""
        self.loader.df = self.sample_data
        X_train, X_test, y_train, y_test = self.loader.split_data(test_size=0.3)
        
        total_samples = len(self.sample_data)
        self.assertAlmostEqual(len(X_test) / total_samples, 0.3, delta=0.1)
        self.assertEqual(len(X_train) + len(X_test), total_samples)
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
