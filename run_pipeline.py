"""
Complete MLOps Pipeline Runner
This script demonstrates the end-to-end MLOps workflow:
1. Train and log model with MLflow
2. Start the API server
3. Test the deployed model
"""

import subprocess
import time
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("OUTPUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    """Run the complete MLOps pipeline"""
    print("🚀 Starting Complete MLOps Pipeline")
    print("This will demonstrate:")
    print("1. Model training with MLflow tracking")
    print("2. API deployment with FastAPI")
    print("3. Model testing and observability")
    
    # Step 1: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies. Please check requirements.txt")
        return
    
    # Step 2: Train model
    if not run_command("python train_model.py", "Training model with MLflow"):
        print("Failed to train model. Please check train_model.py")
        return
    
    # Step 3: Start API server in background
    print(f"\n{'='*60}")
    print("STEP: Starting API server")
    print('='*60)
    
    try:
        # Start the server in background
        server_process = subprocess.Popen(
            ["python", "api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ API server started in background")
        print("Waiting for server to be ready...")
        
        # Wait for server to be ready
        import requests
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API server is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            print("❌ API server not ready after 30 seconds")
            server_process.terminate()
            return
        
        # Step 4: Test the API
        if not run_command("python test_api.py", "Testing deployed API"):
            print("API tests failed, but server is running")
        
        # Step 5: Show MLflow UI info
        print(f"\n{'='*60}")
        print("STEP: MLflow Tracking")
        print('='*60)
        print("✅ Model training completed with MLflow tracking")
        print("📊 To view MLflow UI, run: mlflow ui --backend-store-uri file:./mlruns")
        print("🌐 Then open: http://localhost:5000")
        
        # Step 6: Show API info
        print(f"\n{'='*60}")
        print("STEP: API Deployment")
        print('='*60)
        print("✅ API server is running")
        print("🌐 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("📊 Metrics: http://localhost:8000/metrics")
        
        # Keep server running
        print(f"\n{'='*60}")
        print("🎉 MLOps Pipeline Complete!")
        print('='*60)
        print("The API server is running. Press Ctrl+C to stop.")
        print("You can test the API using the /docs endpoint or test_api.py")
        
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping API server...")
            server_process.terminate()
            server_process.wait()
            print("✅ API server stopped")
    
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        if 'server_process' in locals():
            server_process.terminate()

if __name__ == "__main__":
    main()
