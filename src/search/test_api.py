import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Append current directory to path to prevent absolute import resolution errors
sys.path.append(str(Path(__file__).resolve().parent.parent))

from search.main import app

# Initialize the FastAPI TestClient
client = TestClient(app)


def test_search_endpoint():
    print("Testing HTTP POST /search endpoint...")

    # 1. Arrange payload
    request_payload = {
        "query": "black leather office chair",
        "limit": 5,
        "alpha": 0.5
    }

    # 2. Act
    response = client.post("/search", json=request_payload)

    # 3. Assert HTTP status code
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

    # 4. Assert payload response contents
    json_data = response.json()
    assert isinstance(json_data, list), "Response payload should be a JSON array"
    assert len(json_data) > 0, "Response array should not be empty"

    # Verify our overlap winner item is structurally complete and on top
    top_hit = json_data[0]
    assert top_hit["product_id"] == "B00YQ6X8EO"
    assert "rrf_score" in top_hit
    assert "metadata" in top_hit
    assert top_hit["metadata"]["title"] == "Scented Room Spray - Lavender & Chamomile"

    print("FastAPI /search endpoint integration test passed successfully!")
    print("\nSample JSON response chunk received:")
    import json
    print(json.dumps(json_data[0], indent=2))


if __name__ == "__main__":
    test_search_endpoint()