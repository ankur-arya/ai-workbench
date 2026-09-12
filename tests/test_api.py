from fastapi.testclient import TestClient

from app.main import app
from workbench.config import DEFAULT_DATASET_ID, DEFAULT_EXPERIMENT


def test_health_and_full_http_lifecycle():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["primary_metric"] == "f1_macro"
    assert health.json()["openai_configured"] is False

    created = client.post("/api/experiments", json={"name": DEFAULT_EXPERIMENT})
    assert created.status_code == 200
    experiment_id = created.json()["experiment_id"]

    datasets = client.get("/api/datasets")
    assert datasets.status_code == 200
    assert datasets.json()["datasets"]

    bundled = client.post("/api/datasets", json={"mode": "bundled", "dataset_id": "sentiment-v1"})
    assert bundled.status_code == 200, bundled.text
    bundled_body = bundled.json()
    assert bundled_body["dataset_id"] == DEFAULT_DATASET_ID
    assert bundled_body["n_train"] >= 24
    assert bundled_body["n_test"] >= 9
    assert bundled_body["preview"]
    assert bundled_body["loaded_at"]

    generated = client.post("/api/datasets", json={"mode": "generated", "dataset_id": "sentiment-generated"})
    assert generated.status_code == 200, generated.text
    assert generated.json()["n_train"] > bundled_body["n_train"]
    assert generated.json()["preview"]

    unknown = client.post("/api/datasets", json={"mode": "nope", "dataset_id": "x"})
    assert unknown.status_code == 400

    train = client.post(
        "/api/runs/train",
        json={
            "experiment_name": DEFAULT_EXPERIMENT,
            "dataset_id": DEFAULT_DATASET_ID,
            "model": "local-heuristic",
            "prompt_variant": "concise",
        },
    )
    assert train.status_code == 200, train.text
    run_id = train.json()["run_id"]
    assert train.json()["metrics"]["f1_macro"] > 0.5

    runs = client.get("/api/runs", params={"experiment_id": experiment_id})
    assert runs.status_code == 200
    assert any(row["run_id"] == run_id for row in runs.json()["runs"])

    promoted = client.post(
        "/api/models/promote",
        json={"run_id": run_id, "alias": "champion"},
    )
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["alias"] == "champion"

    predicted = client.post(
        "/api/production/predict",
        json={"texts": ["I love this wonderful product.", "Minutes from the standup are attached."]},
    )
    assert predicted.status_code == 200, predicted.text
    labels = [row["prediction"] for row in predicted.json()["predictions"]]
    assert labels[0] == "positive"
    assert labels[1] == "neutral"
