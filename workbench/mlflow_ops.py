"""MLflow tracking, registry aliases, and UI deep-links."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import mlflow
from mlflow.tracking import MlflowClient

from workbench.config import get_settings


def configure_mlflow(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    return uri


def client() -> MlflowClient:
    configure_mlflow()
    return MlflowClient()


def ui_url() -> str:
    return get_settings().mlflow_ui_url.rstrip("/")


def experiment_url(experiment_id: str) -> str:
    return f"{ui_url()}/#/experiments/{experiment_id}"


def run_url(experiment_id: str, run_id: str) -> str:
    return f"{ui_url()}/#/experiments/{experiment_id}/runs/{run_id}"


def model_url(name: str) -> str:
    return f"{ui_url()}/#/models/{quote(name, safe='')}"


def list_experiments() -> list[dict[str, Any]]:
    experiments = []
    for exp in client().search_experiments():
        if exp.lifecycle_stage != "active":
            continue
        experiments.append(
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "url": experiment_url(exp.experiment_id),
            }
        )
    return experiments


def get_or_create_experiment(name: str) -> dict[str, Any]:
    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        experiment = mlflow.get_experiment(experiment_id)
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "url": experiment_url(experiment.experiment_id),
    }


def _metric_value(run, key: str) -> float | None:
    metric = run.data.metrics.get(key)
    return float(metric) if metric is not None else None


def list_runs(experiment_id: str, max_results: int = 50) -> list[dict[str, Any]]:
    runs = client().search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=max_results,
    )
    rows = []
    for run in runs:
        params = dict(run.data.params)
        metrics = {key: float(value) for key, value in run.data.metrics.items()}
        rows.append(
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": params,
                "metrics": metrics,
                "f1_macro": _metric_value(run, "f1_macro"),
                "accuracy": _metric_value(run, "accuracy"),
                "model": params.get("model"),
                "prompt_variant": params.get("prompt_variant"),
                "url": run_url(run.info.experiment_id, run.info.run_id),
            }
        )
    return rows


def get_run(run_id: str) -> dict[str, Any]:
    run = client().get_run(run_id)
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "params": dict(run.data.params),
        "metrics": {key: float(value) for key, value in run.data.metrics.items()},
        "url": run_url(run.info.experiment_id, run.info.run_id),
    }


def register_and_alias(
    *,
    run_id: str,
    model_name: str,
    alias: str,
    artifact_path: str = "classifier",
) -> dict[str, Any]:
    """Register a run's pyfunc model and point an alias at the new version."""
    configure_mlflow()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    api = client()
    api.set_registered_model_alias(model_name, alias, int(result.version))
    if alias == "champion":
        try:
            previous = api.get_model_version_by_alias(model_name, "challenger")
            if str(previous.version) == str(result.version):
                api.delete_registered_model_alias(model_name, "challenger")
        except Exception:
            pass
    return serialize_model_version(model_name, int(result.version), alias)


def serialize_model_version(name: str, version: int, just_set_alias: str | None = None) -> dict[str, Any]:
    api = client()
    mv = api.get_model_version(name, str(version))
    registered = api.get_registered_model(name)
    aliases = {alias: int(ver) for alias, ver in (registered.aliases or {}).items()}
    return {
        "name": name,
        "version": int(mv.version),
        "run_id": mv.run_id,
        "source": mv.source,
        "aliases": aliases,
        "alias": just_set_alias,
        "model_uri": f"models:/{name}@{just_set_alias}" if just_set_alias else f"models:/{name}/{version}",
        "url": model_url(name),
        "description": registered.description,
    }


def list_registered_models() -> list[dict[str, Any]]:
    models = []
    for rm in client().search_registered_models():
        aliases = {alias: int(ver) for alias, ver in (rm.aliases or {}).items()}
        versions = [
            {
                "version": int(mv.version),
                "run_id": mv.run_id,
                "status": mv.status,
            }
            for mv in rm.latest_versions or []
        ]
        models.append(
            {
                "name": rm.name,
                "description": rm.description,
                "aliases": aliases,
                "versions": versions,
                "champion_uri": f"models:/{rm.name}@champion" if "champion" in aliases else None,
                "url": model_url(rm.name),
            }
        )
    return models


def champion_status(name: str) -> dict[str, Any] | None:
    api = client()
    try:
        rm = api.get_registered_model(name)
    except Exception:
        return None
    aliases = {alias: int(ver) for alias, ver in (rm.aliases or {}).items()}
    champion_version = aliases.get("champion")
    payload = {
        "name": name,
        "aliases": aliases,
        "champion_uri": f"models:/{name}@champion" if champion_version else None,
        "url": model_url(name),
    }
    if champion_version:
        mv = api.get_model_version(name, str(champion_version))
        payload["champion_version"] = champion_version
        payload["champion_run_id"] = mv.run_id
    return payload


def tracking_reachable() -> tuple[bool, str]:
    uri = configure_mlflow()
    try:
        client().search_experiments(max_results=1)
        return True, uri
    except Exception as exc:  # noqa: BLE001
        return False, f"{uri} ({exc})"
