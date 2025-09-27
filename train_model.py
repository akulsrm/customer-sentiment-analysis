import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import SentimentAnalyzer
import pickle
import os

def plot_confusion_matrix(cm, model_name, class_names=['Negative', 'Positive', 'Neutral']):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def plot_model_comparison(results):
    """Plot model comparison"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 2*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def main():
    print("Customer Sentiment Analysis Model Training")
    print("=" * 50)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data()
    
    # Train models
    analyzer.train_models(X_train, y_train)
    
    # Evaluate models
    results = analyzer.evaluate_models(X_test, y_test)
    
    # Print summary
    print("\nModel Performance Summary:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print()
    
    print(f"Best Model: {analyzer.best_model_name}")
    print(f"Best Accuracy: {results[analyzer.best_model_name]['accuracy']:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot confusion matrices for all models
    for model_name, metrics in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive', 'Neutral'],
                   yticklabels=['Negative', 'Positive', 'Neutral'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        values = [results[model][metric] for model in models]
        plt.bar(x + i*width, values, width, label=metric_labels[i], alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Save the trained model
    print("\nSaving trained model...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(analyzer, f)
    
    print("Model training completed!")
    print("Files saved:")
    print("- sentiment_model.pkl (trained model)")
    print("- confusion_matrix_*.png (confusion matrices)")
    print("- model_comparison.png (model comparison chart)")
    
    # Test the model with some examples
    print("\nTesting the best model with sample reviews:")
    test_reviews = [
        "This product is absolutely amazing! I love it so much!",
        "Terrible quality. Very disappointed with this purchase.",
        "The product is okay. Nothing special but works fine."
    ]
    
    for review in test_reviews:
        result = analyzer.predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Predicted Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()