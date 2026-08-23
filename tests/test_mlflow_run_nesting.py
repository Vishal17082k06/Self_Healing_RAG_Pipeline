import os
import pathlib
import tempfile
from unittest.mock import patch, MagicMock

import pytest

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "mock-openai-key")

# main.py sets its MLflow tracking URI at import time, before any pytest fixture can
# run — so this has to happen before `import main`, not inside a fixture. Needs the
# canonical file:///C:/... form (three slashes, forward slashes) — a bare Windows path
# or a hand-built file:// string both get misparsed as a remote URI scheme.
_TEST_MLRUNS_DIR = tempfile.mkdtemp(prefix="test_mlflow_run_nesting_")
os.environ["MLFLOW_TRACKING_URI"] = pathlib.Path(_TEST_MLRUNS_DIR).as_uri()

import main  # noqa: E402 — must import before mlflow, see Case 02 in DEBUGGING_LOG.md
import mlflow  # noqa: E402


@pytest.fixture
def local_mlflow_tracking():
    yield
    mlflow.end_run()


def test_build_vector_db_does_not_crash_when_a_run_is_already_active(local_mlflow_tracking, tmp_path):
    """Regression test: monitor.py's healing chain calls build_vector_db() from inside
    two already-active mlflow runs (monitor_check -> healing_attempt_*). Without
    nested=True on build_vector_db's own start_run, MLflow raises 'Run ... is already
    active' instead of the healing rebuild actually happening."""
    mock_doc = MagicMock()
    mock_doc.page_content = "some content"

    with patch("main.TextLoader") as mock_loader_cls, \
         patch("main.CharacterTextSplitter") as mock_splitter_cls, \
         patch("main.Chroma") as mock_chroma_cls, \
         patch("main.CHROMA_BASE_DIR", str(tmp_path / "chroma_db")):

        mock_loader_cls.return_value.load.return_value = [mock_doc]
        mock_splitter_cls.return_value.split_documents.return_value = [mock_doc]
        mock_chroma_cls.from_documents.return_value = MagicMock()

        with mlflow.start_run(run_name="monitor_check"):
            with mlflow.start_run(run_name="healing_attempt_test", nested=True):
                # This must not raise "Run with UUID ... is already active"
                main.build_vector_db(force_rebuild=True)
