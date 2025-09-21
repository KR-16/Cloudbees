"""
Test script for the Iris Classification API
Demonstrates how to interact with the deployed model
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    # Test data (setosa)
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=test_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    response = requests.get(f"{API_BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_metrics():
    """Test the metrics endpoint"""
    print("\nTesting metrics endpoint...")
    response = requests.get(f"{API_BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_multiple_predictions():
    """Test multiple predictions for observability"""
    print("\nTesting multiple predictions...")
    
    test_cases = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},  # setosa
        {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},  # versicolor
        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},  # virginica
    ]
    
    for i, test_data in enumerate(test_cases):
        response = requests.post(f"{API_BASE_URL}/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            class_names = ["setosa", "versicolor", "virginica"]
            predicted_class = class_names[result["prediction"]]
            print(f"Test {i+1}: Predicted {predicted_class} with confidence {result['confidence']:.3f}")
        else:
            print(f"Test {i+1}: Failed with status {response.status_code}")
        time.sleep(0.5)  # Small delay between requests

def main():
    """Run all tests"""
    print("Starting API tests...")
    print("=" * 50)
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("API is ready!")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        print("API not ready after 30 seconds. Please check if the server is running.")
        return
    
    # Run tests
    tests = [
        test_health,
        test_model_info,
        test_prediction,
        test_multiple_predictions,
        test_metrics
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with error: {e}")
        print("-" * 30)
    
    print(f"\nTests completed: {passed}/{len(tests)} passed")

if __name__ == "__main__":
    main()
