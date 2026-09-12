#!/usr/bin/env python3
"""Headless lifecycle demo: experiment → dataset → train → promote → infer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workbench.config import DEFAULT_DATASET_ID, DEFAULT_EXPERIMENT, LOCAL_HEURISTIC_MODEL
from workbench.datasets import generate_synthetic_dataset, list_datasets, put_dataset
from workbench.mlflow_ops import get_or_create_experiment, list_runs
from workbench.pipeline import production_predict, promote_run, train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic sentiment workbench demo.")
    parser.add_argument("--model", default=LOCAL_HEURISTIC_MODEL)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    experiment = get_or_create_experiment(args.experiment)
    list_datasets()
    put_dataset(generate_synthetic_dataset("sentiment-generated"))

    strong = train_and_evaluate(
        experiment_name=args.experiment,
        dataset_id=DEFAULT_DATASET_ID,
        model=args.model,
        prompt_variant="concise",
    )
    weak = train_and_evaluate(
        experiment_name=args.experiment,
        dataset_id=DEFAULT_DATASET_ID,
        model="local-heuristic-weak",
        prompt_variant="concise",
        run_name="weak-baseline",
    )
    winner = strong if strong["metrics"]["f1_macro"] >= weak["metrics"]["f1_macro"] else weak
    promoted = promote_run(run_id=winner["run_id"], alias=args.alias)
    inferred = production_predict(
        texts=[
            "I am delighted with this update.",
            "This is a dumpster fire and I want a refund.",
            "The shipment includes two adapters.",
        ]
    )
    payload = {
        "experiment": experiment,
        "strong": {"run_id": strong["run_id"], "metrics": strong["metrics"]},
        "weak": {"run_id": weak["run_id"], "metrics": weak["metrics"]},
        "promoted": promoted,
        "production": inferred,
        "runs": list_runs(experiment["experiment_id"]),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
