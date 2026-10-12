from pathlib import Path

import pytest

from app.ml.benchmarking import (
    benchmark_real_world_wildfire_dataset,
    benchmark_real_world_wildfire_dataset_suite,
    build_fire_samples_from_algerian_rows,
    build_fire_samples_from_uci_rows,
    fine_tune_spread_models,
    load_uci_algerian_forest_fires_rows,
    load_uci_forest_fires_rows,
)


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "uci_forest_fires_sample.csv"


def _algerian_fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "uci_algerian_forest_fires_sample.csv"


def test_fixture_provenance_markers_are_official_rows():
    forest_lines = _fixture_path().read_text(encoding="utf-8").splitlines()
    assert forest_lines[0] == "X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area"
    assert forest_lines[1] == "9,9,jul,tue,85.8,48.3,313.4,3.9,18,42,2.7,0,0.36"
    assert forest_lines[-1] == "8,6,aug,wed,85.6,90.4,609.6,6.6,17.4,50,4,0,2.69"

    algerian_lines = _algerian_fixture_path().read_text(encoding="utf-8").splitlines()
    assert algerian_lines[0] == "Bejaia Region Dataset"
    assert (
        algerian_lines[1]
        == "day,month,year,Temperature, RH, Ws,Rain ,FFMC,DMC,DC,ISI,BUI,FWI,Classes"
    )
    assert algerian_lines[2] == "01,06,2012,29,57,18,0,65.7,3.4,7.6,1.3,3.4,0.5,not fire"
    assert algerian_lines[-1] == "30,06,2012,33,50,14,0,88.7,22.9,92.8,7.2,28.3,12.9,fire"


def test_load_uci_rows_from_fixture():
    rows = load_uci_forest_fires_rows(str(_fixture_path()), download_if_missing=False)

    assert len(rows) >= 30
    assert rows[0]["month"] == "jul"
    assert rows[0]["area"] == pytest.approx(0.36)


def test_build_fire_samples_from_real_rows_filters_zero_area():
    rows = load_uci_forest_fires_rows(str(_fixture_path()), download_if_missing=False)
    samples = build_fire_samples_from_uci_rows(rows, include_zero_area=False)

    assert len(samples) == len(rows)
    assert all(sample["spread_rate"] > 0.0 for sample in samples)
    assert all("features" in sample for sample in samples)
    assert all("metadata" in sample for sample in samples)


def test_load_algerian_rows_from_fixture():
    rows = load_uci_algerian_forest_fires_rows(
        str(_algerian_fixture_path()), download_if_missing=False
    )

    assert len(rows) >= 12
    assert rows[0]["month_index"] == 6
    assert rows[0]["temp"] == pytest.approx(29.0)


def test_build_fire_samples_from_algerian_rows_filters_non_fire():
    rows = load_uci_algerian_forest_fires_rows(
        str(_algerian_fixture_path()), download_if_missing=False
    )
    samples = build_fire_samples_from_algerian_rows(rows, include_zero_area=False)

    assert len(samples) < len(rows)
    assert len(samples) >= 8
    assert all(sample["spread_rate"] > 0.0 for sample in samples)


def test_real_world_benchmark_reports_uplift_diagnostics():
    summary = benchmark_real_world_wildfire_dataset(
        str(_fixture_path()),
        download_if_missing=False,
        include_zero_area=False,
        cv_splits=3,
        random_state=42,
        require_informative=True,
        run_tuning=True,
    )

    spread = summary["spread_benchmark"]
    bakeoff = summary["bakeoff"]
    tuning = summary["tuning"]

    assert summary["dataset_name"] == "uci_forest_fires"
    assert summary["samples_used"] >= 20
    assert summary["scorecard"]["grade"] in {"A", "B", "C"}
    assert spread["dataset_is_informative"] is True
    assert spread["best_model"] in {result["name"] for result in spread["results"]}
    assert "uplift_diagnosis" in spread
    assert "effective_tie_with_baseline" in spread

    assert bakeoff["bakeoff_ran"] is True
    assert bakeoff["winner_model"] in {result["name"] for result in bakeoff["results"]}
    assert "uplift_diagnosis" in bakeoff

    assert tuning["tuning_ran"] is True
    assert tuning["tuned_winner_model"] in {result["name"] for result in tuning["results"]}


def test_real_world_benchmark_supports_algerian_dataset_name():
    summary = benchmark_real_world_wildfire_dataset(
        str(_algerian_fixture_path()),
        dataset_name="uci_algerian_forest_fires",
        download_if_missing=False,
        include_zero_area=False,
        cv_splits=3,
        random_state=42,
        require_informative=True,
        run_tuning=False,
    )

    assert summary["dataset_name"] == "uci_algerian_forest_fires"
    assert summary["samples_used"] >= 8
    assert summary["scorecard"]["grade"] in {"A", "B", "C"}
    assert summary["spread_benchmark"]["best_model"]
    assert summary["bakeoff"]["bakeoff_ran"] is True


def test_real_world_benchmark_suite_runs_multiple_datasets():
    summary = benchmark_real_world_wildfire_dataset_suite(
        ["uci_forest_fires", "uci_algerian_forest_fires"],
        dataset_paths={
            "uci_forest_fires": str(_fixture_path()),
            "uci_algerian_forest_fires": str(_algerian_fixture_path()),
        },
        download_if_missing=False,
        include_zero_area=False,
        cv_splits=3,
        random_state=42,
        require_informative=True,
        max_rows=24,
        run_tuning=False,
    )

    assert summary["dataset_name"] == "suite"
    assert summary["dataset_count"] == 2
    assert set(summary["dataset_results"].keys()) == {
        "uci_forest_fires",
        "uci_algerian_forest_fires",
    }
    assert summary["best_dataset_by_spread_uplift"] in {
        "uci_forest_fires",
        "uci_algerian_forest_fires",
    }
    assert summary["scorecard"]["grade"] in {"A", "B", "C"}
    assert summary["scorecard"]["key_metrics"]["dataset_count"] == 2
    assert all("scorecard" in payload for payload in summary["dataset_results"].values())


def test_load_uci_rows_downloads_when_missing(monkeypatch, tmp_path):
    destination = tmp_path / "datasets" / "forestfires.csv"

    class FakeResponse:
        text = (
            "X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area\n"
            "7,5,mar,fri,86.2,26.2,94.3,5.1,8.2,51,6.7,0,0\n"
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr("app.ml.benchmarking.httpx.get", lambda *args, **kwargs: FakeResponse())

    rows = load_uci_forest_fires_rows(str(destination), download_if_missing=True)

    assert destination.exists()
    assert len(rows) == 1
    assert rows[0]["ffmc"] == pytest.approx(86.2)


def test_load_algerian_rows_downloads_when_missing(monkeypatch, tmp_path):
    destination = tmp_path / "datasets" / "algerian_forest_fires.csv"

    class FakeResponse:
        text = (
            "Date,Temp,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,FWI,Classes\n"
            "01/06/2012,31,48,14,0,88,25,80,7.8,25,5.2,fire\n"
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr("app.ml.benchmarking.httpx.get", lambda *args, **kwargs: FakeResponse())

    rows = load_uci_algerian_forest_fires_rows(str(destination), download_if_missing=True)

    assert destination.exists()
    assert len(rows) == 1
    assert rows[0]["classes"] == "fire"


def test_fine_tune_requires_minimum_sample_count():
    with pytest.raises(ValueError):
        fine_tune_spread_models([], cv_splits=3)
