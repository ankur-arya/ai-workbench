"""Environment and MLflow settings. Secrets come from a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_org: str = ""
    openai_project: str = ""
    openai_model: str = "gpt-5-nano"
    openai_fallback_model: str = "gpt-4o-mini"

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_ui_url: str = "http://127.0.0.1:5000"
    mlflow_registered_model_name: str = "sentiment-classifier"
    mlflow_prod_trace_experiment: str = "workbench-production-traces"

    workbench_host: str = "127.0.0.1"
    workbench_port: int = 8000

    data_dir: Path = Field(default_factory=lambda: ROOT / "data")


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Cheap nano / small models we try if the configured id is unavailable.
OPENAI_FALLBACK_CHAIN = (
    "gpt-5-nano",
    "gpt-5.4-nano",
    "gpt-4.1-nano",
    "gpt-4o-mini",
)

LOCAL_HEURISTIC_MODEL = "local-heuristic"

LABELS = ("positive", "negative", "neutral")
PRIMARY_METRIC = "f1_macro"
DEFAULT_DATASET_ID = "sentiment-v1"
DEFAULT_EXPERIMENT = "sentiment-workbench"
REGISTERED_MODEL_DESCRIPTION = (
    "Prompt-based text classifier promoted from AI Workbench runs. "
    "Production loads this model by alias (models:/name@champion), "
    "not by starting a second training experiment."
)
