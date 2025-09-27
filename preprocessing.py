import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean text by removing special characters, numbers, and converting to lowercase"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokenized text"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_words(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        # Stem words
        tokens = self.stem_words(tokens)
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_reviews(self, reviews):
        """Preprocess a list of reviews"""
        return [self.preprocess_text(review) for review in reviews]