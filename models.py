from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
from preprocessing import TextPreprocessor

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Try to load from CSV file first, otherwise use built-in data
        csv_path = 'sentiment_dataset.csv'
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded dataset from {csv_path} with {len(df)} samples")
                return df
            except Exception as e:
                print(f"Error loading CSV: {e}. Using built-in data.")
        
        # Sample positive, negative, and neutral reviews
        positive_reviews = [
            "This product is amazing! I love it so much.",
            "Excellent service and great quality. Highly recommend!",
            "Best purchase I've ever made. Fantastic experience!",
            "Outstanding product quality and fast delivery.",
            "Absolutely wonderful! Exceeded my expectations.",
            "Great value for money. Very satisfied with my purchase.",
            "Perfect product! Exactly what I was looking for.",
            "Amazing customer service and product quality.",
            "Love this product! Will definitely buy again.",
            "Excellent experience from start to finish.",
            "Fantastic quality and amazing customer support.",
            "Could not be happier with this purchase!",
            "Top-notch product with premium features.",
            "Brilliant design and exceptional functionality.",
            "Superb quality that exceeds all expectations."
        ]
        
        negative_reviews = [
            "Terrible product. Very disappointed with the quality.",
            "Worst purchase ever. Complete waste of money.",
            "Horrible experience. Would not recommend to anyone.",
            "Poor quality and bad customer service.",
            "Product broke after one day. Very frustrating.",
            "Not worth the money. Very poor construction.",
            "Awful experience. Regret buying this product.",
            "Cheap materials and poor design. Very disappointed.",
            "Product doesn't work as advertised. Waste of time.",
            "Very poor quality. Would give zero stars if possible.",
            "Complete disaster. Product failed immediately.",
            "Extremely poor build quality and durability.",
            "Shocking customer service and defective product.",
            "Pathetic quality control and manufacturing.",
            "Unacceptable performance and reliability issues."
        ]
        
        neutral_reviews = [
            "The product is okay. Nothing special but works fine.",
            "Average quality. Does what it's supposed to do.",
            "It's fine for the price. Not great, not terrible.",
            "Decent product. Meets basic expectations.",
            "Standard quality. Nothing to complain about.",
            "The product works as expected. Average experience.",
            "Okay product. Could be better, could be worse.",
            "Fair quality for the price paid.",
            "Product is acceptable. Nothing extraordinary.",
            "Average product with standard features.",
            "Mediocre performance but functional.",
            "Standard product with basic functionality.",
            "Adequate quality for everyday use.",
            "Reasonable product with typical features.",
            "Satisfactory performance at a fair price."
        ]
        
        # Create dataset
        reviews = positive_reviews + negative_reviews + neutral_reviews
        labels = [1] * len(positive_reviews) + [0] * len(negative_reviews) + [2] * len(neutral_reviews)
        
        return pd.DataFrame({'review': reviews, 'sentiment': labels})
    
    def prepare_data(self, data=None):
        """Prepare data for training"""
        if data is None:
            data = self.load_sample_data()
        
        # Preprocess reviews
        print("Preprocessing text data...")
        data['processed_review'] = data['review'].apply(self.preprocessor.preprocess_text)
        
        # Split data
        X = data['processed_review']
        y = data['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Vectorize text
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        best_accuracy = 0
        
        print("Evaluating models...")
        for name, model in self.trained_models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Confusion Matrix:\n{conf_matrix}")
            print("-" * 50)
            
            # Update best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        return results
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        # Predict
        prediction = self.best_model.predict(text_vectorized)[0]
        
        # Get prediction probability
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        else:
            confidence = None
        
        # Map prediction to sentiment label
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        
        return {
            'sentiment': sentiment_map[prediction],
            'confidence': confidence,
            'processed_text': processed_text
        }
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance for the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.best_model, 'feature_importances_'):
            # For tree-based models
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            return [(feature_names[i], importances[i]) for i in indices]
        elif hasattr(self.best_model, 'coef_'):
            # For linear models
            coef = np.abs(self.best_model.coef_).mean(axis=0)
            indices = np.argsort(coef)[::-1][:top_n]
            
            return [(feature_names[i], coef[i]) for i in indices]
        else:
            return []