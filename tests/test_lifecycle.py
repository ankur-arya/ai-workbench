from workbench.config import DEFAULT_DATASET_ID, DEFAULT_EXPERIMENT
from workbench.datasets import list_datasets
from workbench.mlflow_ops import get_or_create_experiment, list_runs
from workbench.pipeline import production_predict, promote_run, train_and_evaluate


def test_train_compare_promote_and_champion_inference():
    list_datasets()
    experiment = get_or_create_experiment(DEFAULT_EXPERIMENT)
    strong = train_and_evaluate(
        experiment_name=DEFAULT_EXPERIMENT,
        dataset_id=DEFAULT_DATASET_ID,
        model="local-heuristic",
        prompt_variant="concise",
    )
    weak = train_and_evaluate(
        experiment_name=DEFAULT_EXPERIMENT,
        dataset_id=DEFAULT_DATASET_ID,
        model="local-heuristic-weak",
        prompt_variant="concise",
        run_name="weak-baseline",
    )
    assert strong["metrics"]["f1_macro"] >= weak["metrics"]["f1_macro"]
    assert 0.0 <= strong["metrics"]["f1_macro"] <= 1.0

    runs = list_runs(experiment["experiment_id"])
    assert len(runs) >= 2
    assert runs[0]["f1_macro"] >= runs[1]["f1_macro"]

    promoted = promote_run(run_id=strong["run_id"], alias="champion")
    assert promoted["alias"] == "champion"
    assert promoted["model_uri"] == "models:/sentiment-classifier@champion"

    inferred = production_predict(
        texts=[
            "I am delighted and this is wonderful.",
            "I hate this terrible awful crash.",
            "The report covers office locations.",
        ]
    )
    assert inferred["model_uri"] == "models:/sentiment-classifier@champion"
    assert [row["prediction"] for row in inferred["predictions"]] == ["positive", "negative", "neutral"]
