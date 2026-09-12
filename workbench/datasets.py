"""Synthetic sentiment dataset helpers."""

from __future__ import annotations

import csv
import io
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from workbench.config import DEFAULT_DATASET_ID, LABELS, ROOT, get_settings

REQUIRED_CSV_COLUMNS = ("split", "text", "label")

BUNDLED_CSV = ROOT / "data" / "sentiment.csv"


@dataclass
class ClassificationDataset:
    dataset_id: str
    name: str
    labels: tuple[str, ...]
    train_texts: list[str]
    train_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]
    source: str = "bundled-csv"
    description: str = "Synthetic 3-class sentiment (positive / negative / neutral)."
    loaded_at: str = ""

    @property
    def n_train(self) -> int:
        return len(self.train_texts)

    @property
    def n_test(self) -> int:
        return len(self.test_texts)

    def test_rows(self) -> list[dict[str, str]]:
        return [
            {"text": text, "label": label}
            for text, label in zip(self.test_texts, self.test_labels, strict=True)
        ]

    def preview(self, limit: int = 6) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for text, label in zip(self.train_texts[:limit], self.train_labels[:limit], strict=False):
            rows.append({"split": "train", "text": text, "label": label})
        for text, label in zip(self.test_texts[: max(0, limit // 2)], self.test_labels, strict=False):
            rows.append({"split": "test", "text": text, "label": label})
        return rows

    def to_summary(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "labels": list(self.labels),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "source": self.source,
            "description": self.description,
            "preview": self.preview(),
            "primary_metric": "f1_macro",
            "loaded_at": self.loaded_at or datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }


class DatasetParseError(ValueError):
    """User-facing CSV validation error."""


def dataset_id_from_filename(filename: str) -> str:
    stem = Path(filename or "upload").stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").lower() or "upload"
    return f"{safe}-{uuid.uuid4().hex[:6]}"


def parse_classification_csv(
    raw: str,
    *,
    dataset_id: str,
    name: str,
    source: str = "upload",
    labels: tuple[str, ...] = LABELS,
) -> ClassificationDataset:
    """Parse an uploaded CSV. Requires split, text, label and both train and test rows."""
    sample = raw.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(sample))
    if not reader.fieldnames:
        raise DatasetParseError("CSV is empty or has no header row.")
    fields = {name.strip().lower() for name in reader.fieldnames if name}
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in fields]
    if missing:
        raise DatasetParseError(
            "CSV must include columns: split, text, label. "
            f"Missing: {', '.join(missing)}."
        )

    train_texts: list[str] = []
    train_labels: list[str] = []
    test_texts: list[str] = []
    test_labels: list[str] = []
    allowed = set(labels)
    for index, row in enumerate(reader, start=2):
        normalized = {(key or "").strip().lower(): (value or "").strip() for key, value in row.items()}
        split = normalized.get("split", "").lower()
        text = normalized.get("text", "")
        label = normalized.get("label", "").lower()
        if not text and not label and not split:
            continue
        if split not in {"train", "test"}:
            raise DatasetParseError(
                f"Row {index} has split '{normalized.get('split', '') or '(empty)'}'; use train or test."
            )
        if not text:
            raise DatasetParseError(f"Row {index} is missing text.")
        if label not in allowed:
            raise DatasetParseError(
                f"Row {index} has label '{normalized.get('label', '') or '(empty)'}'; "
                f"allowed labels are {', '.join(labels)}."
            )
        if split == "test":
            test_texts.append(text)
            test_labels.append(label)
        else:
            train_texts.append(text)
            train_labels.append(label)

    if not train_texts:
        raise DatasetParseError("CSV has no train rows (split=train).")
    if not test_texts:
        raise DatasetParseError("CSV has no test rows (split=test).")

    return ClassificationDataset(
        dataset_id=dataset_id,
        name=name,
        labels=labels,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        source=source,
        description=f"Uploaded classification CSV ({len(train_texts)} train / {len(test_texts)} test).",
        loaded_at=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    )


def _read_csv(path: Path) -> ClassificationDataset:
    train_texts: list[str] = []
    train_labels: list[str] = []
    test_texts: list[str] = []
    test_labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = (row.get("split") or "train").strip().lower()
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text or label not in LABELS:
                continue
            if split == "test":
                test_texts.append(text)
                test_labels.append(label)
            else:
                train_texts.append(text)
                train_labels.append(label)
    if not train_texts or not test_texts:
        raise ValueError(f"Dataset at {path} is missing train or test rows.")
    return ClassificationDataset(
        dataset_id=DEFAULT_DATASET_ID,
        name="Synthetic sentiment",
        labels=LABELS,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        source=str(path.resolve()),
        loaded_at=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    )


# Extra generated rows used when the UI asks to regenerate a tiny split in-memory.
_GENERATED_TRAIN = [
    ("I am overjoyed with the results. This is wonderful.", "positive"),
    ("A charming, uplifting experience from the first click.", "positive"),
    ("Utterly miserable. I will never use this again.", "negative"),
    ("Broken, buggy, and a complete waste of an afternoon.", "negative"),
    ("The checklist has four items remaining for tomorrow.", "neutral"),
    ("I copied the notes into the shared folder.", "neutral"),
]
_GENERATED_TEST = [
    ("What a pleasant surprise. I am genuinely grateful.", "positive"),
    ("This is dreadful. I am angry and done with it.", "negative"),
    ("The calendar invite is set for 3pm next week.", "neutral"),
]


def resolve_bundled_csv() -> Path:
    """Find sentiment.csv even if the process cwd is not the repo root."""
    settings_dir = Path(get_settings().data_dir)
    candidates = [
        settings_dir / "sentiment.csv",
        BUNDLED_CSV,
        ROOT / "data" / "sentiment.csv",
        Path(__file__).resolve().parents[1] / "data" / "sentiment.csv",
        Path.cwd() / "data" / "sentiment.csv",
    ]
    seen: set[str] = set()
    for path in candidates:
        resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Could not find data/sentiment.csv. Looked next to the package, "
        f"under {settings_dir}, and under the current working directory ({Path.cwd()})."
    )


def load_bundled_dataset() -> ClassificationDataset:
    return _read_csv(resolve_bundled_csv())


def generate_synthetic_dataset(dataset_id: str = "sentiment-generated") -> ClassificationDataset:
    """Build a slightly larger in-memory split from the bundled CSV plus extras."""
    base = load_bundled_dataset()
    train_texts = list(base.train_texts)
    train_labels = list(base.train_labels)
    test_texts = list(base.test_texts)
    test_labels = list(base.test_labels)
    for text, label in _GENERATED_TRAIN:
        train_texts.append(text)
        train_labels.append(label)
    for text, label in _GENERATED_TEST:
        test_texts.append(text)
        test_labels.append(label)
    return ClassificationDataset(
        dataset_id=dataset_id,
        name="Generated synthetic sentiment",
        labels=LABELS,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        source="generated",
        description="Bundled synthetic sentiment plus extra generated rows.",
        loaded_at=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    )


_STORE: dict[str, ClassificationDataset] = {}


def list_datasets() -> list[ClassificationDataset]:
    if DEFAULT_DATASET_ID not in _STORE:
        _STORE[DEFAULT_DATASET_ID] = load_bundled_dataset()
    return list(_STORE.values())


def get_dataset(dataset_id: str) -> ClassificationDataset:
    if dataset_id not in _STORE:
        if dataset_id == DEFAULT_DATASET_ID:
            _STORE[dataset_id] = load_bundled_dataset()
        else:
            raise KeyError(dataset_id)
    return _STORE[dataset_id]


def put_dataset(dataset: ClassificationDataset) -> ClassificationDataset:
    _STORE[dataset.dataset_id] = dataset
    return dataset
