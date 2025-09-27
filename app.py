from flask import Flask, render_template, request, jsonify
import pickle
import os
from models import SentimentAnalyzer

app = Flask(__name__)

# Load the trained model
model_path = 'sentiment_model.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        analyzer = pickle.load(f)
else:
    # If model doesn't exist, create and train a new one
    print("No trained model found. Training a new model...")
    analyzer = SentimentAnalyzer()
    X_train, X_test, y_train, y_test = analyzer.prepare_data()
    analyzer.train_models(X_train, y_train)
    results = analyzer.evaluate_models(X_test, y_test)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(analyzer, f)
    print("Model trained and saved successfully!")

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get prediction
        result = analyzer.predict_sentiment(text)
        
        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': round(result['confidence'], 4) if result['confidence'] else None,
            'processed_text': result['processed_text']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid texts provided'}), 400
        
        results = []
        for text in texts:
            result = analyzer.predict_sentiment(text)
            results.append({
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': round(result['confidence'], 4) if result['confidence'] else None
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        'best_model': analyzer.best_model_name,
        'models_available': list(analyzer.models.keys()),
        'feature_extraction': 'TF-IDF Vectorization',
        'preprocessing_steps': [
            'Text cleaning (lowercase, remove special characters)',
            'Tokenization',
            'Stop-word removal',
            'Stemming (Porter Stemmer)'
        ]
    })

@app.route('/test_examples')
def test_examples():
    """Get test examples"""
    examples = [
        {
            'text': 'This product is absolutely amazing! I love it so much!',
            'expected': 'Positive'
        },
        {
            'text': 'Terrible quality. Very disappointed with this purchase.',
            'expected': 'Negative'
        },
        {
            'text': 'The product is okay. Nothing special but works fine.',
            'expected': 'Neutral'
        },
        {
            'text': 'Excellent customer service and fast delivery. Highly recommended!',
            'expected': 'Positive'
        },
        {
            'text': 'Poor construction and cheap materials. Not worth the money.',
            'expected': 'Negative'
        }
    ]
    
    return jsonify({'examples': examples})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)