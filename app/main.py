"""FastAPI workbench: orchestrate the MLflow lifecycle and serve the UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workbench.config import (
    DEFAULT_EXPERIMENT,
    LABELS,
    LOCAL_HEURISTIC_MODEL,
    PRIMARY_METRIC,
    get_settings,
)
from workbench.classifier import PROMPT_VARIANTS
from workbench.datasets import (
    DatasetParseError,
    dataset_id_from_filename,
    generate_synthetic_dataset,
    get_dataset,
    list_datasets,
    load_bundled_dataset,
    parse_classification_csv,
    put_dataset,
)
from workbench.model_configs import get_config, list_configs, prompt_templates, save_config
from workbench.mlflow_ops import (
    champion_status,
    experiment_url,
    get_or_create_experiment,
    list_experiments,
    list_registered_models,
    list_runs,
    tracking_reachable,
    ui_url,
)
from workbench.pipeline import production_predict, promote_run, train_and_evaluate

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AI Workbench", version="0.1.0")


class ExperimentCreate(BaseModel):
    name: str = DEFAULT_EXPERIMENT


class DatasetCreate(BaseModel):
    mode: str = "bundled"
    dataset_id: str = "sentiment-generated"


class TrainRequest(BaseModel):
    experiment_name: str = DEFAULT_EXPERIMENT
    dataset_id: str
    config_id: str | None = None
    model: str | None = None
    prompt_variant: str | None = None
    prompt_template: str | None = None
    temperature: float | None = None
    run_name: str | None = None


class ClassifierConfigCreate(BaseModel):
    name: str
    llm: str = LOCAL_HEURISTIC_MODEL
    prompt_variant: str = "custom"
    prompt_template: str
    temperature: float = 0.0
    config_id: str | None = None


class PromoteRequest(BaseModel):
    run_id: str
    model_name: str | None = None
    alias: str = Field(default="champion", pattern="^(champion|challenger)$")


class PredictRequest(BaseModel):
    texts: list[str]
    model_name: str | None = None
    alias: str = "champion"


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    reachable, detail = tracking_reachable()
    return {
        "ok": reachable,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "mlflow_ui_url": ui_url(),
        "mlflow_reachable": reachable,
        "mlflow_detail": detail,
        "openai_configured": bool(
            settings.openai_api_key and settings.openai_org and settings.openai_project
        ),
        "openai_org_configured": bool(settings.openai_org),
        "openai_project_configured": bool(settings.openai_project),
        "default_openai_model": settings.openai_model,
        "fallback_openai_model": settings.openai_fallback_model,
        "registered_model_name": settings.mlflow_registered_model_name,
        "primary_metric": PRIMARY_METRIC,
        "labels": list(LABELS),
        "local_heuristic_model": LOCAL_HEURISTIC_MODEL,
        "prompt_variants": list(PROMPT_VARIANTS.keys()),
    }


@app.get("/api/experiments")
def api_list_experiments() -> dict[str, Any]:
    try:
        return {"experiments": list_experiments()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"MLflow is not reachable: {exc}") from exc


@app.post("/api/experiments")
def api_create_experiment(body: ExperimentCreate) -> dict[str, Any]:
    try:
        return get_or_create_experiment(body.name.strip())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not create experiment: {exc}") from exc


@app.get("/api/datasets")
def api_list_datasets() -> dict[str, Any]:
    return {"datasets": [dataset.to_summary() for dataset in list_datasets()]}


@app.post("/api/datasets")
def api_create_dataset(body: DatasetCreate) -> dict[str, Any]:
    try:
        if body.mode == "generated":
            dataset = put_dataset(generate_synthetic_dataset(body.dataset_id or "sentiment-generated"))
        elif body.mode == "bundled":
            dataset = put_dataset(load_bundled_dataset())
        else:
            raise HTTPException(status_code=400, detail=f"Unknown dataset mode: {body.mode}")
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not load dataset: {exc}") from exc
    return dataset.to_summary()


@app.post("/api/datasets/upload")
async def api_upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.") from exc
    dataset_id = dataset_id_from_filename(filename)
    try:
        dataset = parse_classification_csv(
            text,
            dataset_id=dataset_id,
            name=filename,
            source=f"upload:{filename}",
        )
    except DatasetParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return put_dataset(dataset).to_summary()


@app.get("/api/datasets/{dataset_id}")
def api_get_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        dataset = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found") from exc
    return dataset.to_summary()


@app.get("/api/classifier-configs")
def api_list_classifier_configs() -> dict[str, Any]:
    return {
        "configs": [item.to_dict() for item in list_configs()],
        "prompt_templates": prompt_templates(),
    }


@app.post("/api/classifier-configs")
def api_save_classifier_config(body: ClassifierConfigCreate) -> dict[str, Any]:
    if not body.prompt_template.strip():
        raise HTTPException(status_code=400, detail="Prompt template cannot be empty.")
    config = save_config(
        name=body.name,
        llm=body.llm,
        prompt_variant=body.prompt_variant,
        prompt_template=body.prompt_template,
        temperature=body.temperature,
        config_id=body.config_id,
    )
    return config.to_dict()


@app.get("/api/classifier-configs/{config_id}")
def api_get_classifier_config(config_id: str) -> dict[str, Any]:
    try:
        return get_config(config_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model config not found") from exc


@app.post("/api/runs/train")
def api_train(body: TrainRequest) -> dict[str, Any]:
    model = body.model
    prompt_variant = body.prompt_variant
    prompt_template = body.prompt_template
    temperature = 0.0 if body.temperature is None else body.temperature
    config_id = body.config_id
    if config_id:
        try:
            saved = get_config(config_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown model config: {config_id}") from exc
        model = model or saved.llm
        prompt_variant = prompt_variant or saved.prompt_variant
        prompt_template = prompt_template or saved.prompt_template
        if body.temperature is None:
            temperature = saved.temperature
    model = model or LOCAL_HEURISTIC_MODEL
    prompt_variant = prompt_variant or "concise"
    try:
        return train_and_evaluate(
            experiment_name=body.experiment_name.strip(),
            dataset_id=body.dataset_id,
            model=model,
            prompt_variant=prompt_variant,
            prompt_template=prompt_template,
            temperature=temperature,
            run_name=body.run_name,
            config_id=config_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runs")
def api_list_runs(experiment_id: str) -> dict[str, Any]:
    try:
        return {
            "experiment_id": experiment_id,
            "experiment_url": experiment_url(experiment_id),
            "primary_metric": PRIMARY_METRIC,
            "runs": list_runs(experiment_id),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/models/promote")
def api_promote(body: PromoteRequest) -> dict[str, Any]:
    try:
        return promote_run(run_id=body.run_id, model_name=body.model_name, alias=body.alias)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    settings = get_settings()
    try:
        return {
            "models": list_registered_models(),
            "champion": champion_status(settings.mlflow_registered_model_name),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/production/predict")
def api_predict(body: PredictRequest) -> dict[str, Any]:
    if not body.texts:
        raise HTTPException(status_code=400, detail="Provide at least one text.")
    try:
        return production_predict(
            texts=body.texts,
            model_name=body.model_name,
            alias=body.alias,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
