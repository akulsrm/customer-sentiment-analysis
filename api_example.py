#!/usr/bin/env python3
"""
API Usage Example for Customer Sentiment Analysis System
Demonstrates how to interact with the Flask API programmatically
"""

import requests
import json

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🌐 Testing Flask API Endpoints")
    print("=" * 50)
    
    # Test single prediction
    print("\n1. Testing Single Prediction Endpoint")
    print("-" * 40)
    
    single_review = {
        "review": "This product is absolutely fantastic! Best purchase I've ever made!"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=single_review)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction successful")
            print(f"Review: {single_review['review']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing single prediction: {e}")
    
    # Test batch prediction
    print("\n2. Testing Batch Prediction Endpoint")
    print("-" * 40)
    
    batch_reviews = {
        "reviews": [
            "Excellent product! Highly recommend to everyone.",
            "Terrible quality. Complete waste of money.",
            "Average product. Nothing special but works fine.",
            "Amazing! Exceeded all expectations!",
            "Poor construction. Very disappointed."
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/predict_batch", json=batch_reviews)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Batch prediction successful")
            for i, result in enumerate(results['predictions'], 1):
                print(f"{i}. {result['sentiment']} (Confidence: {result['confidence']:.4f})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing batch prediction: {e}")
    
    # Test model info
    print("\n3. Testing Model Info Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Model info retrieved successfully")
            print(f"Model Type: {info['model_type']}")
            print(f"Accuracy: {info['accuracy']:.4f}")
            print(f"Training Samples: {info['training_samples']}")
            print(f"Classes: {', '.join(info['classes'])}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    # Test example reviews
    print("\n4. Testing Example Reviews Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/examples")
        if response.status_code == 200:
            examples = response.json()
            print(f"✅ Example reviews retrieved successfully")
            print("Available examples:")
            for category, reviews in examples.items():
                print(f"\n{category.title()} Examples:")
                for review in reviews:
                    print(f"  • {review}")
        else:
            print(f"❌ Examples failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting examples: {e}")

def main():
    """Main function"""
    print("🔌 Flask API Testing Tool")
    print("=" * 60)
    print("Make sure the Flask app is running on http://localhost:5000")
    print("You can start it with: python app.py")
    print("=" * 60)
    
    # Check if Flask app is running
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            print("✅ Flask app is running!")
            test_api_endpoints()
        else:
            print("❌ Flask app is not responding correctly")
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running!")
        print("Please start the Flask app first with: python app.py")
        print("Then run this script again to test the API endpoints.")

if __name__ == "__main__":
    main()