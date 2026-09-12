"""In-memory store for prompt + LLM classifier definitions (the Model step)."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field

from workbench.classifier import CONCISE_SYSTEM, PROMPT_VARIANTS
from workbench.config import LABELS, LOCAL_HEURISTIC_MODEL


@dataclass
class ClassifierConfig:
    config_id: str
    name: str
    llm: str
    prompt_variant: str
    prompt_template: str
    temperature: float = 0.0
    labels: list[str] = field(default_factory=lambda: list(LABELS))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["note"] = (
            "This is the prompt+LLM definition. Train/evaluate runs it and logs metrics to MLflow."
        )
        return payload


_STORE: dict[str, ClassifierConfig] = {}


def _slug(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "classifier").strip().lower()).strip("-")
    return safe or "classifier"


def prompt_templates() -> dict[str, str]:
    return dict(PROMPT_VARIANTS)


def save_config(
    *,
    name: str,
    llm: str,
    prompt_variant: str,
    prompt_template: str,
    temperature: float = 0.0,
    labels: list[str] | None = None,
    config_id: str | None = None,
) -> ClassifierConfig:
    template = (prompt_template or "").strip() or PROMPT_VARIANTS.get(prompt_variant, CONCISE_SYSTEM)
    variant = (prompt_variant or "custom").strip() or "custom"
    ident = config_id or f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    config = ClassifierConfig(
        config_id=ident,
        name=name.strip() or ident,
        llm=llm.strip() or LOCAL_HEURISTIC_MODEL,
        prompt_variant=variant,
        prompt_template=template,
        temperature=float(temperature),
        labels=list(labels or LABELS),
    )
    _STORE[ident] = config
    return config


def get_config(config_id: str) -> ClassifierConfig:
    if config_id not in _STORE:
        raise KeyError(config_id)
    return _STORE[config_id]


def list_configs() -> list[ClassifierConfig]:
    if not _STORE:
        save_config(
            name="Default heuristic",
            llm=LOCAL_HEURISTIC_MODEL,
            prompt_variant="concise",
            prompt_template=CONCISE_SYSTEM,
            temperature=0.0,
            config_id="default-heuristic",
        )
    return list(_STORE.values())
