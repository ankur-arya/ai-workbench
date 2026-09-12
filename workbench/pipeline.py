"""Train → evaluate → register → production inference."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from workbench.classifier import TextClassifierModel, classify_texts, resolve_prompt_template
from workbench.config import (
    DEFAULT_EXPERIMENT,
    LABELS,
    PRIMARY_METRIC,
    REGISTERED_MODEL_DESCRIPTION,
    get_settings,
)
from workbench.datasets import ClassificationDataset, get_dataset
from workbench.metrics import compute_metrics, per_class_report
from workbench.mlflow_ops import (
    champion_status,
    configure_mlflow,
    experiment_url,
    get_or_create_experiment,
    model_url,
    register_and_alias,
    run_url,
)


def _log_pyfunc(model: TextClassifierModel, example_texts: list[str]) -> str:
    example = pd.DataFrame({"text": example_texts[:3]})
    signature = infer_signature(example, model.predict(example))
    kwargs: dict[str, Any] = {
        "python_model": model,
        "signature": signature,
        "input_example": example,
        "pip_requirements": [
            "mlflow",
            "openai",
            "pandas",
            "scikit-learn",
            "python-dotenv",
        ],
    }
    try:
        info = mlflow.pyfunc.log_model(name="classifier", **kwargs)
    except TypeError:
        info = mlflow.pyfunc.log_model(artifact_path="classifier", **kwargs)
    return getattr(info, "model_uri", "runs:/{run}/classifier")


def train_and_evaluate(
    *,
    experiment_name: str,
    dataset_id: str,
    model: str,
    prompt_variant: str = "concise",
    prompt_template: str | None = None,
    temperature: float = 0.0,
    run_name: str | None = None,
    config_id: str | None = None,
) -> dict[str, Any]:
    """Fit-less LLM/heuristic classifier: predict on the test split and log metrics."""
    configure_mlflow()
    experiment = get_or_create_experiment(experiment_name)
    dataset = get_dataset(dataset_id)
    settings = get_settings()

    try:
        mlflow.openai.autolog(disable=False, log_traces=True)
    except Exception:
        pass

    resolved_template = resolve_prompt_template(prompt_variant, prompt_template)
    display_name = run_name or f"{model}-{prompt_variant}"
    with mlflow.start_run(experiment_id=experiment["experiment_id"], run_name=display_name) as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                "task": "text-classification",
                "dataset_id": dataset.dataset_id,
                "workbench": "ai-workbench",
                "primary_metric": PRIMARY_METRIC,
                **({"config_id": config_id} if config_id else {}),
            }
        )
        mlflow.log_params(
            {
                "model": model,
                "prompt_variant": prompt_variant,
                "temperature": temperature,
                "n_train": dataset.n_train,
                "n_test": dataset.n_test,
                "labels": ",".join(dataset.labels),
                "dataset_id": dataset.dataset_id,
                **({"config_id": config_id} if config_id else {}),
            }
        )

        predictions, resolved_model = classify_texts(
            dataset.test_texts,
            model=model,
            prompt_variant=prompt_variant,
            prompt_template=resolved_template,
            temperature=temperature,
            labels=dataset.labels,
        )
        if resolved_model != model:
            mlflow.log_param("resolved_model", resolved_model)

        metrics = compute_metrics(dataset.test_labels, predictions, labels=dataset.labels)
        mlflow.log_metrics(metrics)
        report = per_class_report(dataset.test_labels, predictions, labels=dataset.labels)

        rows = [
            {
                "text": text,
                "label": label,
                "prediction": pred,
                "correct": label == pred,
            }
            for text, label, pred in zip(dataset.test_texts, dataset.test_labels, predictions, strict=True)
        ]
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preds_path = tmp_path / "predictions.csv"
            frame.to_csv(preds_path, index=False)
            mlflow.log_artifact(str(preds_path), artifact_path="eval")
            report_path = tmp_path / "classification_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="eval")
            config_path = tmp_path / "classifier_config.json"
            config = {
                "model": resolved_model,
                "prompt_variant": prompt_variant,
                "prompt_template": resolved_template,
                "temperature": temperature,
                "labels": list(dataset.labels),
                "requested_model": model,
                "config_id": config_id,
            }
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(config_path), artifact_path="eval")
            prompt_path = tmp_path / "prompt_template.txt"
            prompt_path.write_text(resolved_template, encoding="utf-8")
            mlflow.log_artifact(str(prompt_path), artifact_path="eval")

        pyfunc = TextClassifierModel(
            {
                "model": resolved_model,
                "prompt_variant": prompt_variant,
                "prompt_template": resolved_template,
                "temperature": temperature,
                "labels": list(dataset.labels),
            }
        )
        model_uri = _log_pyfunc(pyfunc, dataset.test_texts[:3])

        return {
            "run_id": run_id,
            "experiment_id": experiment["experiment_id"],
            "experiment_name": experiment["name"],
            "dataset_id": dataset.dataset_id,
            "model": model,
            "resolved_model": resolved_model,
            "prompt_variant": prompt_variant,
            "prompt_template": resolved_template,
            "config_id": config_id,
            "metrics": metrics,
            "primary_metric": PRIMARY_METRIC,
            "predictions": rows,
            "model_uri": model_uri.replace("{run}", run_id),
            "registered_model_name": settings.mlflow_registered_model_name,
            "url": run_url(experiment["experiment_id"], run_id),
            "experiment_url": experiment_url(experiment["experiment_id"]),
        }


def promote_run(
    *,
    run_id: str,
    model_name: str | None = None,
    alias: str = "champion",
) -> dict[str, Any]:
    configure_mlflow()
    settings = get_settings()
    name = model_name or settings.mlflow_registered_model_name
    try:
        mlflow.set_registered_model(
            name=name,
            description=REGISTERED_MODEL_DESCRIPTION,
        )
    except Exception:
        try:
            from mlflow import MlflowClient

            MlflowClient().create_registered_model(name, description=REGISTERED_MODEL_DESCRIPTION)
        except Exception:
            pass
    result = register_and_alias(run_id=run_id, model_name=name, alias=alias)
    result["promotion_model"] = (
        "Same MLflow instance Model Registry alias "
        f"({alias}). Production is not a new experiment or a second MLflow server."
    )
    return result


def production_predict(
    *,
    texts: list[str],
    model_name: str | None = None,
    alias: str = "champion",
) -> dict[str, Any]:
    """Load the registry alias and score texts. Traces go to the prod-trace experiment."""
    configure_mlflow()
    settings = get_settings()
    name = model_name or settings.mlflow_registered_model_name
    model_uri = f"models:/{name}@{alias}"
    status = champion_status(name) or {}

    get_or_create_experiment(settings.mlflow_prod_trace_experiment)
    mlflow.set_experiment(settings.mlflow_prod_trace_experiment)

    loaded = mlflow.pyfunc.load_model(model_uri)
    with mlflow.start_span(name="production_inference") as span:
        span.set_inputs({"texts": texts, "model_uri": model_uri})
        predictions = [str(value) for value in loaded.predict(pd.DataFrame({"text": texts}))]
        span.set_outputs({"predictions": predictions})
        trace_id = getattr(span, "trace_id", None) or getattr(span, "request_id", None)

    return {
        "model_uri": model_uri,
        "alias": alias,
        "name": name,
        "version": status.get("champion_version") if alias == "champion" else status.get("aliases", {}).get(alias),
        "predictions": [
            {"text": text, "prediction": pred} for text, pred in zip(texts, predictions, strict=True)
        ],
        "trace_id": trace_id,
        "trace_experiment": settings.mlflow_prod_trace_experiment,
        "registry_url": model_url(name),
        "note": (
            "Production served the registered alias; traces are observability only "
            "and are not a second training experiment."
        ),
    }


def ensure_default_experiment() -> dict[str, Any]:
    configure_mlflow()
    return get_or_create_experiment(DEFAULT_EXPERIMENT)
