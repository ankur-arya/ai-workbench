import pytest

from workbench.classifier import (
    TextClassifierModel,
    classify_texts,
    heuristic_classify,
    require_openai_credentials,
)


def test_heuristic_labels_clear_sentiment():
    assert heuristic_classify("I love this wonderful excellent product") == "positive"
    assert heuristic_classify("I hate this terrible awful experience") == "negative"
    assert heuristic_classify("The meeting is scheduled for Friday") == "neutral"


def test_local_batch_and_pyfunc_predict():
    texts = [
        "Fantastic value and excellent results.",
        "This is the worst and I hate it.",
        "Version 2.1 shipped with notes.",
    ]
    preds, model = classify_texts(texts, model="local-heuristic")
    assert model == "local-heuristic"
    assert preds == ["positive", "negative", "neutral"]

    wrapped = TextClassifierModel({"model": "local-heuristic", "labels": ["positive", "negative", "neutral"]})
    assert wrapped.predict({"text": texts}) == preds
    assert wrapped.predict(texts) == preds


def test_openai_requires_key_org_and_project(monkeypatch):
    from workbench.config import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_ORG", "")
    monkeypatch.setenv("OPENAI_PROJECT", "")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY, OPENAI_ORG, and OPENAI_PROJECT"):
        require_openai_credentials()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="OPENAI_ORG"):
        require_openai_credentials()

    monkeypatch.setenv("OPENAI_ORG", "org-test")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="OPENAI_PROJECT"):
        require_openai_credentials()

    monkeypatch.setenv("OPENAI_PROJECT", "proj-test")
    get_settings.cache_clear()
    key, org, project = require_openai_credentials()
    assert (key, org, project) == ("sk-test", "org-test", "proj-test")
    get_settings.cache_clear()
