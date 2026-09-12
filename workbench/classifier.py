"""Prompt-based sentiment classifiers (OpenAI nano-class + local heuristic)."""

from __future__ import annotations

import os
import re
from typing import Any

import mlflow

from workbench.config import (
    LABELS,
    LOCAL_HEURISTIC_MODEL,
    OPENAI_FALLBACK_CHAIN,
    get_settings,
)

CONCISE_SYSTEM = (
    "You are a text classifier. Classify the user text into exactly one label: "
    "{labels}. Reply with only the label, lowercase, no punctuation or explanation."
)

FEWSHOT_SYSTEM = (
    "You are a text classifier. Choose exactly one label from: {labels}.\n"
    "Examples:\n"
    '- "I am so happy with this" -> positive\n'
    '- "This is terrible and I hate it" -> negative\n'
    '- "The meeting is at 3pm" -> neutral\n'
    "Reply with only the label."
)

PROMPT_VARIANTS = {
    "concise": CONCISE_SYSTEM,
    "few-shot": FEWSHOT_SYSTEM,
}

_POS = {
    "love",
    "great",
    "excellent",
    "amazing",
    "happy",
    "wonderful",
    "fantastic",
    "delighted",
    "joyful",
    "brilliant",
    "thrilled",
    "outstanding",
    "impressive",
    "recommend",
    "pleasant",
    "grateful",
    "enjoyable",
    "uplifting",
    "overjoyed",
    "charming",
}
_NEG = {
    "hate",
    "terrible",
    "awful",
    "worst",
    "disappointed",
    "horrible",
    "furious",
    "dreadful",
    "letdown",
    "broken",
    "frustrating",
    "regret",
    "useless",
    "miserable",
    "angry",
    "dumpster",
    "crashes",
    "rude",
    "denied",
    "buggy",
}
_WEAK_POS = {"love", "great", "excellent", "happy"}
_WEAK_NEG = {"hate", "terrible", "worst", "awful"}


def _normalize_label(raw: str, labels: tuple[str, ...] = LABELS) -> str:
    token = (raw or "").strip().lower()
    token = re.sub(r"[^a-z]+", " ", token).strip()
    for label in labels:
        if token == label or token.startswith(label) or label in token.split():
            return label
    return token.split()[0] if token else labels[-1]


def heuristic_classify(text: str, *, weak: bool = False) -> str:
    words = set(re.findall(r"[a-z']+", text.lower()))
    pos = _WEAK_POS if weak else _POS
    neg = _WEAK_NEG if weak else _NEG
    pos_hits = len(words & pos)
    neg_hits = len(words & neg)
    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


def _is_gpt5_family(model: str) -> bool:
    return model.startswith("gpt-5")


def _chat_complete(client: Any, model: str, messages: list[dict[str, str]], temperature: float) -> str:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if _is_gpt5_family(model):
        kwargs["max_completion_tokens"] = 16
    else:
        kwargs["max_tokens"] = 16
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return content.strip()


def require_openai_credentials() -> tuple[str, str, str]:
    """Return (api_key, organization, project) or raise a clear setup error."""
    settings = get_settings()
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    organization = settings.openai_org or os.environ.get("OPENAI_ORG", "")
    project = settings.openai_project or os.environ.get("OPENAI_PROJECT", "")
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_ORG", organization),
            ("OPENAI_PROJECT", project),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "OpenAI client requires OPENAI_API_KEY, OPENAI_ORG, and OPENAI_PROJECT. "
            f"Missing: {', '.join(missing)}. Copy .env.example to .env and fill all three."
        )
    return api_key, organization, project


def openai_classify(
    text: str,
    *,
    model: str | None = None,
    prompt_variant: str = "concise",
    temperature: float = 0.0,
    labels: tuple[str, ...] = LABELS,
) -> tuple[str, str]:
    """Return (predicted_label, model_id_used)."""
    settings = get_settings()
    api_key, organization, project = require_openai_credentials()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, organization=organization, project=project)
    template = PROMPT_VARIANTS.get(prompt_variant, CONCISE_SYSTEM)
    messages = [
        {"role": "system", "content": template.format(labels=", ".join(labels))},
        {"role": "user", "content": text},
    ]
    preferred = model or settings.openai_model
    chain: list[str] = []
    for candidate in (preferred, settings.openai_fallback_model, *OPENAI_FALLBACK_CHAIN):
        if candidate and candidate not in chain and candidate != LOCAL_HEURISTIC_MODEL:
            chain.append(candidate)

    last_error: Exception | None = None
    for candidate in chain:
        try:
            raw = _chat_complete(client, candidate, messages, temperature)
            return _normalize_label(raw, labels), candidate
        except Exception as exc:  # noqa: BLE001 — we want any SDK / HTTP failure to try the next id
            last_error = exc
            continue
    raise RuntimeError(f"All OpenAI model ids failed. Last error: {last_error}") from last_error


def classify_texts(
    texts: list[str],
    *,
    model: str,
    prompt_variant: str = "concise",
    temperature: float = 0.0,
    labels: tuple[str, ...] = LABELS,
) -> tuple[list[str], str]:
    """Classify a batch. Returns (predictions, resolved_model_id)."""
    if model == LOCAL_HEURISTIC_MODEL or model.startswith("local-heuristic"):
        weak = model.endswith("-weak")
        return [heuristic_classify(text, weak=weak) for text in texts], model

    predictions: list[str] = []
    resolved = model
    for text in texts:
        label, resolved = openai_classify(
            text,
            model=model,
            prompt_variant=prompt_variant,
            temperature=temperature,
            labels=labels,
        )
        predictions.append(label)
    return predictions, resolved


class TextClassifierModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc classifier. Config is pickled with the model for production loads."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def _texts(self, model_input: Any) -> list[str]:
        if isinstance(model_input, dict) and "text" in model_input:
            value = model_input["text"]
            if isinstance(value, str):
                return [value]
            return [str(item) for item in list(value)]
        if isinstance(model_input, list):
            if model_input and isinstance(model_input[0], dict):
                return [str(row.get("text", row)) for row in model_input]
            return [str(item) for item in model_input]
        if hasattr(model_input, "columns") and "text" in getattr(model_input, "columns", []):
            return [str(value) for value in model_input["text"].tolist()]
        if hasattr(model_input, "tolist"):
            values = model_input.tolist()
            if values and isinstance(values[0], list):
                return [str(row[0]) for row in values]
            return [str(value) for value in values]
        return [str(model_input)]

    def predict(self, model_input, params=None):
        texts = self._texts(model_input)
        model = self.config.get("model", LOCAL_HEURISTIC_MODEL)
        prompt_variant = self.config.get("prompt_variant", "concise")
        temperature = float(self.config.get("temperature", 0.0))
        labels = tuple(self.config.get("labels", LABELS))
        predictions, _ = classify_texts(
            texts,
            model=model,
            prompt_variant=prompt_variant,
            temperature=temperature,
            labels=labels,
        )
        return predictions
