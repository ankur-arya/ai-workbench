from fastapi.testclient import TestClient

from app.main import app
from workbench.config import DEFAULT_EXPERIMENT, LABELS
from workbench.datasets import DatasetParseError, parse_classification_csv


VALID_CSV = """split,text,label
train,I love this wonderful product,positive
train,I hate this terrible crash,negative
train,The report covers office locations,neutral
test,I am delighted and grateful,positive
test,This is dreadful and awful,negative
test,Minutes from the standup are attached,neutral
"""


def test_parse_rejects_missing_columns():
    try:
        parse_classification_csv("text,label\nhi,positive\n", dataset_id="x", name="bad")
    except DatasetParseError as exc:
        assert "Missing: split" in str(exc)
    else:
        raise AssertionError("expected DatasetParseError")


def test_parse_rejects_bad_label_and_empty_test():
    try:
        parse_classification_csv(
            "split,text,label\ntrain,hello,cheerful\n",
            dataset_id="x",
            name="bad",
        )
    except DatasetParseError as exc:
        assert "cheerful" in str(exc)
    else:
        raise AssertionError("expected DatasetParseError")

    try:
        parse_classification_csv(
            "split,text,label\ntrain,hello,positive\n",
            dataset_id="x",
            name="bad",
        )
    except DatasetParseError as exc:
        assert "no test rows" in str(exc)
    else:
        raise AssertionError("expected DatasetParseError")


def test_upload_csv_and_train_saved_config():
    client = TestClient(app)
    client.post("/api/experiments", json={"name": DEFAULT_EXPERIMENT})

    uploaded = client.post(
        "/api/datasets/upload",
        files={"file": ("tiny.csv", VALID_CSV.encode("utf-8"), "text/csv")},
    )
    assert uploaded.status_code == 200, uploaded.text
    body = uploaded.json()
    assert body["n_train"] == 3
    assert body["n_test"] == 3
    assert body["preview"]
    assert body["dataset_id"].startswith("tiny-")
    dataset_id = body["dataset_id"]

    bad = client.post(
        "/api/datasets/upload",
        files={"file": ("bad.csv", b"foo,bar\n1,2\n", "text/csv")},
    )
    assert bad.status_code == 400
    assert "split, text, label" in bad.json()["detail"]

    saved = client.post(
        "/api/classifier-configs",
        json={
            "name": "offline-heuristic",
            "llm": "local-heuristic",
            "prompt_variant": "concise",
            "prompt_template": "Classify into {labels}. Reply with only the label.",
            "temperature": 0,
        },
    )
    assert saved.status_code == 200, saved.text
    config_id = saved.json()["config_id"]
    assert "{labels}" in saved.json()["prompt_template"]

    listed = client.get("/api/classifier-configs")
    assert any(item["config_id"] == config_id for item in listed.json()["configs"])

    trained = client.post(
        "/api/runs/train",
        json={
            "experiment_name": DEFAULT_EXPERIMENT,
            "dataset_id": dataset_id,
            "config_id": config_id,
        },
    )
    assert trained.status_code == 200, trained.text
    assert trained.json()["dataset_id"] == dataset_id
    assert trained.json()["config_id"] == config_id
    assert trained.json()["metrics"]["f1_macro"] > 0.5
    assert set(LABELS) == {"positive", "negative", "neutral"}
