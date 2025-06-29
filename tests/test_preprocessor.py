import unittest
from src.data.preprocessor import AcronymExpander, SpacyCleaner, TextPreprocessor


class TestAcronymExpander(unittest.TestCase):
    """Unit tests for AcronymExpander"""
    
    def setUp(self):
        self.expander = AcronymExpander()
    
    def test_expand_acronyms(self):
        """Test acronym expansion"""
        texts = ["HODL strong!", "FUD everywhere", "Normal text"]
        result = self.expander.fit_transform(texts)
        
        self.assertIn("Hold On For Dear Life", result[0])
        self.assertIn("Fear Uncertainty Doubt", result[1])
        self.assertEqual(result[2], "Normal text")
    
    def test_case_insensitive(self):
        """Test case insensitive matching"""
        texts = ["hodl strong", "Fud everywhere"]
        result = self.expander.fit_transform(texts)
        
        self.assertIn("Hold On For Dear Life", result[0])
        self.assertIn("Fear Uncertainty Doubt", result[1])


class TestSpacyCleaner(unittest.TestCase):
    """Unit tests for SpacyCleaner"""
    
    def setUp(self):
        self.cleaner = SpacyCleaner()
    
    def test_fit_loads_model(self):
        """Test that fit loads spaCy model"""
        self.cleaner.fit([])
        self.assertIsNotNone(self.cleaner.nlp)
    
    def test_transform_without_fit_raises_error(self):
        """Test transform without fit raises ValueError"""
        with self.assertRaises(ValueError):
            self.cleaner.transform(["test"])
    
    def test_text_cleaning(self):
        """Test text cleaning and lemmatization"""
        self.cleaner.fit([])
        texts = ["The stocks are performing very well!"]
        result = self.cleaner.transform(texts)
        
        # Should remove stop words and punctuation
        self.assertNotIn("the", result[0].lower())
        self.assertNotIn("are", result[0].lower())
        self.assertNotIn("!", result[0])


class TestTextPreprocessor(unittest.TestCase):
    """Unit tests for TextPreprocessor"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_build_pipeline(self):
        """Test pipeline building"""
        pipeline = self.preprocessor.build_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.steps), 4)
    
    def test_fit_transform(self):
        """Test fit_transform functionality"""
        texts = ["HODL! The market is bullish", "FUD everywhere, very bearish"]
        result = self.preprocessor.fit_transform(texts)
        
        # Should return sparse matrix
        self.assertTrue(hasattr(result, 'toarray'))
        self.assertEqual(result.shape[0], 2)
    
    def test_transform_without_fit_raises_error(self):
        """Test transform without fit raises ValueError"""
        with self.assertRaises(ValueError):
            self.preprocessor.transform(["test"])
