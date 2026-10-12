import json
from types import SimpleNamespace

import audit


def _completed_process(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_model_readiness_gate_fails_when_scorecard_is_not_production_ready(monkeypatch):
    payload = {
        "report": {
            "can_benchmark": True,
            "dataset_is_informative": True,
            "sample_count": 34,
            "group_count": 10,
            "best_model": "hist_gradient_boosting",
            "baseline_model": "random_forest_baseline",
            "best_improvement_pct": 8.9,
            "best_model_metrics": {"r2": 0.31},
            "time_split": {
                "best_model_metrics": {
                    "test_r2": -3.62,
                }
            },
            "scorecard": {
                "grade": "C",
                "production_ready": False,
                "state_of_the_art": False,
                "assessment": "No, not yet.",
            },
        }
    }

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: _completed_process(stdout=json.dumps(payload)),
    )

    result = audit.run_model_readiness_gate(
        require_uplift=False,
        min_uplift_pct=1.0,
        require_min_r2=False,
        min_best_r2=0.05,
        require_production_ready=True,
    )

    assert result.passed is False
    assert "production-ready scorecard" in result.details
    assert "scorecard_grade=C" in result.details


def test_model_readiness_gate_fails_when_scorecard_grade_is_below_required_floor(monkeypatch):
    payload = {
        "report": {
            "can_benchmark": True,
            "dataset_is_informative": True,
            "sample_count": 34,
            "group_count": 10,
            "best_model": "hist_gradient_boosting",
            "baseline_model": "random_forest_baseline",
            "best_improvement_pct": 8.9,
            "best_model_metrics": {"r2": 0.31},
            "time_split": {
                "best_model_metrics": {
                    "test_r2": 0.18,
                }
            },
            "scorecard": {
                "grade": "C",
                "production_ready": False,
                "state_of_the_art": False,
                "assessment": "No, not yet.",
            },
        }
    }

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: _completed_process(stdout=json.dumps(payload)),
    )

    result = audit.run_model_readiness_gate(
        require_uplift=False,
        min_uplift_pct=1.0,
        require_min_r2=False,
        min_best_r2=0.05,
        require_production_ready=False,
        min_scorecard_grade="B",
    )

    assert result.passed is False
    assert "below required grade B" in result.details
    assert "min_scorecard_grade=B" in result.details


def test_model_readiness_gate_passes_when_scorecard_is_production_ready(monkeypatch):
    payload = {
        "report": {
            "can_benchmark": True,
            "dataset_is_informative": True,
            "sample_count": 48,
            "group_count": 12,
            "best_model": "hist_gradient_boosting",
            "baseline_model": "random_forest_baseline",
            "best_improvement_pct": 6.3,
            "best_model_metrics": {"r2": 0.24},
            "time_split": {
                "best_model_metrics": {
                    "test_r2": 0.18,
                }
            },
            "scorecard": {
                "grade": "B",
                "production_ready": True,
                "state_of_the_art": False,
                "assessment": (
                    "No, not yet cutting-edge. Model is production-ready but still leaves headroom."
                ),
            },
        }
    }

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: _completed_process(stdout=json.dumps(payload)),
    )

    result = audit.run_model_readiness_gate(
        require_uplift=False,
        min_uplift_pct=1.0,
        require_min_r2=False,
        min_best_r2=0.05,
        require_production_ready=True,
        min_scorecard_grade="B",
    )

    assert result.passed is True
    assert "Model readiness thresholds satisfied" in result.details
    assert "production_ready=True" in result.details


def test_gates_use_this_interpreter_and_every_js_test(monkeypatch):
    import sys
    from pathlib import Path

    gates = dict(audit.command_gates())
    assert gates["python regression tests"][0] == sys.executable
    js_files = {p.name for p in (Path(audit.ROOT_DIR) / "tests").glob("*.mjs")}
    assert js_files and js_files <= {Path(arg).name for arg in gates["javascript tests"]}

    seen = []
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda command, **kwargs: seen.append(command) or _completed_process(stdout="{}"),
    )
    audit.run_model_readiness_gate(
        require_uplift=False,
        min_uplift_pct=0.0,
        require_min_r2=False,
        min_best_r2=0.0,
        require_production_ready=False,
    )
    assert seen[0][0] == sys.executable
