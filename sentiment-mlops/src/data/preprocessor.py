import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import spacy
from emoji import replace_emoji
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Financial chat acronyms dictionary
CHAT_WORD = {
    'FOMO': 'Fear Of Missing Out',
    'FUD': 'Fear Uncertainty Doubt',
    'DYOR': 'Do Your Own Research',
    'BTFD': 'Buy The Fucking Dip',
    'HODL': 'Hold On For Dear Life',
    'ATH': 'All Time High',
    'ATL': 'All Time Low',
    'IPO': 'Initial Public Offering',
    'ROI': 'Return On Investment',
    'EPS': 'Earnings Per Share',
    'P/E': 'Price To Earnings Ratio',
    'YTD': 'Year To Date',
    'YOY': 'Year Over Year',
    'QoQ': 'Quarter Over Quarter',
    'SL': 'Stop Loss',
    'TP': 'Take Profit',
    'PT': 'Price Target',
    'MCAP': 'Market Capitalization',
    'VOL': 'Trading Volume',
    'ETF': 'Exchange Traded Fund',
    'CFD': 'Contract For Difference',
    'MOON': 'To The Moon',
    'BEAR': 'Bearish Sentiment',
    'BULL': 'Bullish Sentiment',
}

# Compile regex pattern for acronym matching
_ACRO_PAT = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CHAT_WORD) + r')\b',
                       flags=re.IGNORECASE)


class AcronymExpander(BaseEstimator, TransformerMixin):
    """Expand financial chat acronyms to full phrases"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [
            _ACRO_PAT.sub(lambda m: CHAT_WORD[m.group(1).upper()], txt)
            for txt in X
        ]


class SpacyCleaner(BaseEstimator, TransformerMixin):
    """Clean and lemmatize text using spaCy"""
    
    def __init__(self, model_name="en_core_web_sm", batch_size=50):
        self.model_name = model_name
        self.batch_size = batch_size
        self.nlp = None
    
    def fit(self, X, y=None):
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.model_name, disable=["parser", "ner"])
        except OSError:
            raise OSError(f"spaCy model '{self.model_name}' not found. "
                         "Install with: python -m spacy download en_core_web_sm")
        return self
    
    def transform(self, X, y=None):
        if self.nlp is None:
            raise ValueError("SpacyCleaner not fitted. Call fit() first.")
        
        out = []
        for doc in self.nlp.pipe(X, batch_size=self.batch_size):
            toks = [
                tok.lemma_.lower()
                for tok in doc
                if (
                    not tok.is_stop         # drop "the", "and", etc.
                    and not tok.is_punct    # drop punctuation
                    and not tok.like_url    # drop URLs
                    and not tok.like_email  # drop emails
                    and tok.is_alpha        # only letters
                    and tok.lemma_ != "-PRON-"
                )
            ]
            out.append(" ".join(toks))
        return out


def strip_html_emoji(texts):
    """Remove HTML tags and emojis from text"""
    cleaned = []
    for t in texts:
        no_html = re.sub(r'<.*?>', ' ', t)
        no_emoji = replace_emoji(no_html, replace='')
        cleaned.append(no_emoji)
    return cleaned


def vader_scores(texts):
    """Compute VADER sentiment scores"""
    sia = SentimentIntensityAnalyzer()
    return np.array([[sia.polarity_scores(t)['compound']] for t in texts])

def identity_tokenizer(x):
    return str.split(x)

def identity_preprocessor(x):
    return x

class TextPreprocessor:
    """Complete text preprocessing pipeline"""
    
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df
        self.pipeline = None
        
    def build_pipeline(self):
        """Build the preprocessing pipeline"""
        self.pipeline = Pipeline([
            ("strip", FunctionTransformer(strip_html_emoji, validate=False)),
            ("acro", AcronymExpander()),
            ("clean", SpacyCleaner()),
            ("tfidf", TfidfVectorizer(
                tokenizer=identity_tokenizer,
                preprocessor=identity_preprocessor,
                token_pattern=None,
                min_df=self.min_df,
                max_df=self.max_df
            ))
        ])
        return self.pipeline
    
    def fit_transform(self, texts):
        """Fit pipeline and transform texts"""
        if self.pipeline is None:
            self.build_pipeline()
        return self.pipeline.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")
        return self.pipeline.transform(texts)