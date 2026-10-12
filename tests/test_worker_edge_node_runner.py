import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js is required for Worker regression tests"
)
def test_worker_edge_node_suite_passes():
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = repo_root / "tests"

    result = subprocess.run(
        [
            "node",
            "--test",
            str(suite_path / "test_worker_edge.mjs"),
            str(suite_path / "test_regression_guards.mjs"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Node Worker regression suite failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
