"""
Data loading and basic preprocessing utilities
"""
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config.settings import DATA_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and basic preprocessing"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or (DATA_DIR / DATA_CONFIG["input_file"])
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from CSV file"""
        path = file_path or self.data_path
        
        try:
            logger.info(f"Loading data from {path}")
            self.df = pd.read_csv(path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Data file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def basic_info(self) -> dict:
        """Get basic information about the dataset"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "null_counts": self.df.isnull().sum().to_dict(),
            "duplicate_count": self.df.duplicated().sum(),
            "target_distribution": self.df[DATA_CONFIG["target_column"]].value_counts().to_dict()
        }
        
        logger.info(f"Dataset info: {info}")
        return info
    
    def clean_data(self) -> pd.DataFrame:
        """Basic data cleaning"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_shape = self.df.shape
        
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
        # Remove rows with missing values in key columns
        key_columns = [DATA_CONFIG["text_column"], DATA_CONFIG["target_column"]]
        self.df.dropna(subset=key_columns, inplace=True)
        
        final_shape = self.df.shape
        logger.info(f"Data cleaned. Shape changed from {initial_shape} to {final_shape}")
        
        return self.df
    
    def split_data(self, test_size: float = None, random_state: int = None):
        """Split data into train and test sets"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        test_size = test_size or DATA_CONFIG["test_size"]
        random_state = random_state or DATA_CONFIG["random_state"]
        
        X = self.df[DATA_CONFIG["text_column"]]
        y = self.df[DATA_CONFIG["target_column"]]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        logger.info(f"Data split completed. Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_splits(self):
        """Get train/test splits (must call split_data first)"""
        if any(x is None for x in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Data not split. Call split_data() first.")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, output_dir: str = None):
        """Save processed data splits"""
        if any(x is None for x in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Data not split. Call split_data() first.")
        
        output_dir = Path(output_dir) if output_dir else DATA_DIR / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # Save train data
        train_df = pd.DataFrame({
            DATA_CONFIG["text_column"]: self.X_train,
            DATA_CONFIG["target_column"]: self.y_train
        })
        train_df.to_csv(output_dir / "train.csv", index=False)
        
        # Save test data
        test_df = pd.DataFrame({
            DATA_CONFIG["text_column"]: self.X_test,
            DATA_CONFIG["target_column"]: self.y_test
        })
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        logger.info(f"Processed data saved to {output_dir}")


def load_sample_data():
    """Load sample data for testing"""
    # Create sample data if no real data is available
    sample_data = pd.DataFrame({
        'Sentence': [
            'The stock market is performing very well today',
            'I am worried about the economic downturn',
            'The company reported neutral earnings this quarter',
            'HODL to the moon! This coin is bullish',
            'FUD spreading everywhere, market looks bearish'
        ],
        'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
    })
    
    return sample_data


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Try to load real data, fallback to sample
    try:
        df = loader.load_data()
    except FileNotFoundError:
        print("No data file found, using sample data")
        loader.df = load_sample_data()
        df = loader.df
    
    # Basic processing
    info = loader.basic_info()
    print("Dataset Info:", info)
    
    cleaned_df = loader.clean_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")