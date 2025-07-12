from fastapi.testclient import TestClient
from main import app # Assuming your FastAPI app instance is named 'app' in main.py

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    # Replace 200 with the actual expected status code for your root path
    # If your root path doesn't exist or returns something else, adjust the assertion.
    assert response.status_code == 200
    # You can add more assertions here, e.g., checking the response body:
    # assert response.json() == {"message": "Hello World"} 