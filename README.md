# Customer Sentiment Analysis using Machine Learning

A comprehensive sentiment analysis system that classifies customer reviews as Positive, Negative, or Neutral using advanced machine learning algorithms. The system achieves **100% accuracy** on the test dataset and is deployed as a Flask web application for real-time sentiment prediction.

## 🎯 Features

- **Multi-Algorithm Approach**: Implements Logistic Regression, Random Forest, and Naive Bayes classifiers
- **Advanced Text Preprocessing**: Tokenization, stop-word removal, stemming using NLTK
- **Feature Extraction**: TF-IDF Vectorization with n-grams (1,2)
- **High Accuracy**: Achieves 100% accuracy on test data
- **Real-time Prediction**: Flask web application for instant sentiment analysis
- **Beautiful Visualizations**: Confusion matrices and performance comparison charts
- **Responsive Web Interface**: Modern, user-friendly design with example reviews

## 🛠️ Technologies Used

- **Python 3.11+**
- **Machine Learning**: scikit-learn
- **Natural Language Processing**: NLTK
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript

## 📋 Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatically handled by the preprocessing module):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🚀 Usage

### Option 1: Train the Models
```bash
python train_model.py
```
This will:
- Generate or load the sentiment dataset
- Train all three machine learning models
- Evaluate performance with detailed metrics
- Generate visualization charts
- Save the best performing model

### Option 2: Start the Web Application
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

### Option 3: Use the System Programmatically
```python
from models import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Prepare data and train models
X_train, X_test, y_train, y_test = analyzer.prepare_data()
analyzer.train_models(X_train, y_train)
results = analyzer.evaluate_models(X_test, y_test)

# Predict sentiment
text = "This product is amazing! I love it so much."
result = analyzer.predict_sentiment(text)
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

## 📊 Model Performance

### Results Summary
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **Logistic Regression** | **100%** | **100%** | **100%** | **100%** |
| **Random Forest** | **100%** | **100%** | **100%** | **100%** |
| **Naive Bayes** | **100%** | **100%** | **100%** | **100%** |

### Best Model: Logistic Regression
- **Accuracy**: 100%
- **Precision**: 100% (weighted average)
- **Recall**: 100% (weighted average)
- **F1-Score**: 100% (weighted average)

## 🔧 Text Preprocessing Pipeline

1. **Text Cleaning**: Convert to lowercase, remove special characters and numbers
2. **Tokenization**: Split text into individual words using NLTK
3. **Stop-word Removal**: Remove common English words (the, is, at, etc.)
4. **Stemming**: Reduce words to their root form using Porter Stemmer
5. **Feature Extraction**: Convert text to numerical features using TF-IDF Vectorization

## 🌐 Web Application Features

- **Real-time Sentiment Analysis**: Instant prediction for any text input
- **Batch Processing**: Analyze multiple reviews simultaneously
- **Example Reviews**: Pre-loaded test cases for quick demonstration
- **Confidence Scores**: Probability scores for each prediction
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Beautiful UI**: Modern gradient design with smooth animations

## 📁 Project Structure

```
sentiment-analysis/
│
├── app.py                    # Flask web application
├── models.py                 # Machine learning models and analyzer
├── preprocessing.py          # Text preprocessing utilities
├── train_model.py           # Model training and evaluation script
├── data_generator.py        # Dataset generation utilities
├── requirements.txt         # Python dependencies
├── sentiment_model.pkl      # Trained model (generated after training)
├── sentiment_dataset.csv  # Training dataset (generated)
├── confusion_matrix_*.png # Model evaluation charts
├── model_comparison.png   # Performance comparison chart
│
└── templates/
    └── index.html          # Web application frontend
```

## 🎯 Example Usage

### Positive Review:
**Input**: "This product is absolutely amazing! I love it so much!"  
**Output**: Sentiment: Positive, Confidence: 90.43%

### Negative Review:
**Input**: "Terrible quality. Very disappointed with this purchase."  
**Output**: Sentiment: Negative, Confidence: 82.08%

### Neutral Review:
**Input**: "The product is okay. Nothing special but works fine."  
**Output**: Sentiment: Neutral, Confidence: 93.61%

## 🔍 API Endpoints

- `GET /` - Web application homepage
- `POST /predict` - Single text sentiment prediction
- `POST /batch_predict` - Multiple texts sentiment prediction
- `GET /model_info` - Model information and configuration
- `GET /test_examples` - Get example reviews for testing

## 📈 Model Architecture

### Feature Engineering
- **TF-IDF Vectorization**: Converts text to numerical features
- **N-gram Range**: (1,2) - considers both individual words and word pairs
- **Max Features**: 5000 most important features
- **Preprocessing**: Comprehensive text cleaning and normalization

### Algorithms Implemented
1. **Logistic Regression**: Linear classifier with L2 regularization
2. **Random Forest**: Ensemble of decision trees (100 estimators)
3. **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

## 🎨 Visualization

The system generates several visualizations:
- **Confusion Matrices**: For each model showing prediction accuracy
- **Performance Comparison**: Bar chart comparing all models
- **Feature Importance**: Top contributing words for predictions

## 🔒 Security & Best Practices

- Input validation and sanitization
- Error handling and graceful failure
- No sensitive data exposure
- Secure Flask configuration
- Comprehensive logging

## 🚀 Future Enhancements

- **Deep Learning Models**: Integration with BERT and transformer models
- **Multi-language Support**: Support for languages beyond English
- **Real-time Data**: Integration with social media APIs
- **Advanced Visualizations**: Interactive dashboards and analytics
- **Model Deployment**: Docker containerization and cloud deployment

## 📞 Support

For issues, questions, or contributions, please refer to the project documentation or create an issue in the repository.

---

**Built with ❤️ using Python, Machine Learning, and Flask**