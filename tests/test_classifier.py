from workbench.classifier import TextClassifierModel, classify_texts, heuristic_classify


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
