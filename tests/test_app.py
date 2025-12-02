# tests/test_app.py
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
import importlib

# Ensure project root (one level above tests/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_MODULE_NAME = "app"  # your app.py


def import_app_with_test_env():
    """Import the app module with RUNNING_TESTS set, and a clean module state."""
    os.environ["RUNNING_TESTS"] = "1"

    # Make sure we re-import with the env var in effect
    if APP_MODULE_NAME in sys.modules:
        del sys.modules[APP_MODULE_NAME]

    return importlib.import_module(APP_MODULE_NAME)


def test_app_imports_without_error():
    """Basic sanity check: app module imports without crashing."""
    import_app_with_test_env()


def test_main_runs_without_streamlit_errors(monkeypatch):
    """
    Smoke test: app.main() runs without raising an exception.

    We mock `app.st` so no real Streamlit runtime is needed.
    """
    app = import_app_with_test_env()

    # Fake streamlit object
    fake_st = MagicMock()
    fake_st.session_state = {}

    # Replace the `st` imported in app.py with our fake
    monkeypatch.setattr(app, "st", fake_st)

    # If main() raises, pytest will fail the test
    app.main()
