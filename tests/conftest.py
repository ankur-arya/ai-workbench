from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workbench.config import get_settings


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    tracking = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
