from fastapi.testclient import TestClient
from search.main import app

client = TestClient(app)


def run_search(alpha_value: float):
    response = client.post("/search", json={
        "query": "leather chair",
        "limit": 5,
        "alpha": alpha_value
    })
    return response.json()


def test_alpha_tuning():
    print("Testing alpha tuning effects on ranking...")

    # Scenario 1: Heavy Keyword Bias (alpha = 0.9)
    sparse_heavy = run_search(alpha_value=0.9)
    print(f"\n[Alpha = 0.9] Top Rank: {sparse_heavy[0]['product_id']} | Title: {sparse_heavy[0]['metadata']['title']}")

    # Scenario 2: Near-Pure Semantic Bias (alpha = 0.01)
    # Pushing alpha low enough so that the single-list Dense Rank 1 can beat the Overlap Rank 2
    dense_heavy = run_search(alpha_value=0.01)
    print(f"[Alpha = 0.01] Top Rank: {dense_heavy[0]['product_id']} | Title: {dense_heavy[0]['metadata']['title']}")

    # Assertions to ensure alpha shifts rankings correctly
    assert sparse_heavy[0][
               'product_id'] == "B00YQ6X8EO", "Keyword-heavy search failed to surface the sparse match first."
    assert dense_heavy[0][
               'product_id'] == "B088XYZ789", "Semantic-heavy search failed to surface the dense match first."

    print("\n✅ Alpha weighting logic successfully alters rank outcomes!")

if __name__ == "__main__":
    test_alpha_tuning()