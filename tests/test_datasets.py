from workbench.datasets import generate_synthetic_dataset, load_bundled_dataset, resolve_bundled_csv


def test_bundled_split_has_all_labels():
    dataset = load_bundled_dataset()
    assert dataset.n_train >= 24
    assert dataset.n_test >= 9
    assert set(dataset.train_labels) == {"positive", "negative", "neutral"}
    assert set(dataset.test_labels) == {"positive", "negative", "neutral"}


def test_generated_dataset_grows():
    bundled = load_bundled_dataset()
    generated = generate_synthetic_dataset()
    assert generated.n_train > bundled.n_train
    assert generated.n_test > bundled.n_test


def test_bundled_csv_resolves_from_other_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = resolve_bundled_csv()
    assert path.is_file()
    assert path.name == "sentiment.csv"
    dataset = load_bundled_dataset()
    assert dataset.n_train >= 24
    assert "loaded_at" in dataset.to_summary()
