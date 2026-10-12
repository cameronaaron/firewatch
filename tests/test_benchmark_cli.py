"""Tests for benchmark CLI helpers."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from app.ml import benchmark_cli
from app.ml.benchmark_cli import capture_live_history, run_benchmark_report, seed_synthetic_history
from app.models.database import DistanceHistory, Fire, Location
from app.services.fire_data import FireDataService


def _add_location(db, location_id: str = "loc-1") -> Location:
    location = Location(
        location_id=location_id,
        address="123 Test St",
        display_name="Test Location",
        icon="home",
        latitude=34.0,
        longitude=-118.0,
        is_active=True,
    )
    db.add(location)
    db.commit()
    db.refresh(location)
    return location


def _add_fire(db, fire_id: str = "fire-1") -> Fire:
    fire = Fire(
        fire_id=fire_id,
        name="Test Fire",
        latitude=34.1,
        longitude=-118.1,
        acres_burned=1000.0,
        percent_contained=15.0,
    )
    db.add(fire)
    db.commit()
    db.refresh(fire)
    return fire


def test_run_benchmark_report_without_enough_samples(db):
    _add_location(db)
    _add_fire(db)

    report = run_benchmark_report(db)

    assert report["can_benchmark"] is False
    assert report["sample_count"] == 0
    assert "Need at least 6 labeled samples" in report["reason"]


def test_seed_synthetic_history_creates_informative_labels(db):
    _add_location(db)
    _add_fire(db)

    seeded = seed_synthetic_history(
        db,
        pair_limit=1,
        points_per_pair=8,
        step_minutes=30,
        random_seed=7,
        replace_existing=True,
    )

    assert seeded["pairs_seeded"] == 1
    assert seeded["history_rows_added"] == 8

    report = run_benchmark_report(db, min_history_points=3, cv_splits=3)
    assert report["can_benchmark"] is True
    assert report["sample_count"] >= 6
    assert report["target_unique_count"] >= 2


def test_capture_live_history_persists_rows(db, monkeypatch):
    location = _add_location(db, location_id="loc-capture")
    _add_fire(db, fire_id="capture-fire")

    state = {"step": 0}

    async def fake_process_fire_data(db_session, locations):
        assert db_session is db
        assert len(locations) == 1
        assert locations[0].location_id == location.location_id
        state["step"] += 1
        distance = 8.0 - (state["step"] * 0.4)
        return [
            {
                "fire_id": "capture-fire",
                "name": "Capture Fire",
                "latitude": 34.2,
                "longitude": -118.2,
                "acres_burned": 1400.0,
                "percent_contained": 12.0,
                "distances": {location.location_id: distance},
                "directions": {location.location_id: "N (0 deg)"},
                "bearings": {location.location_id: 0.0},
                "movement_data": {
                    location.location_id: {
                        "moving_towards": True,
                        "speed": 0.4,
                    }
                },
            }
        ]

    monkeypatch.setattr(FireDataService, "process_fire_data", fake_process_fire_data)

    summary = asyncio.run(capture_live_history(db, iterations=3, interval_seconds=0))

    assert summary["captures_completed"] == 3
    assert summary["history_rows_added"] == 3
    assert db.query(DistanceHistory).count() == 3


def test_persist_processed_fires_updates_existing_and_skips_unknown_location(db):
    location = _add_location(db, location_id="persist-home")
    existing_fire = _add_fire(db, fire_id="persist-fire")

    observed_at = datetime.now(UTC)
    summary = benchmark_cli._persist_processed_fires(
        db,
        locations=[location],
        fires_data=[
            {
                "fire_id": "persist-fire",
                "name": "Updated Name",
                "latitude": 35.1,
                "longitude": -117.2,
                "acres_burned": 2500.0,
                "percent_contained": 8.0,
                "distances": {
                    "persist-home": 4.2,
                    "unknown-location": 9.9,
                },
                "bearings": {"persist-home": 33.0},
                "directions": {"persist-home": "NE (33 deg)"},
                "movement_data": {
                    "persist-home": {
                        "moving_towards": True,
                        "speed": 1.1,
                    }
                },
            }
        ],
        observed_at=observed_at,
    )

    db.refresh(existing_fire)
    history_rows = db.query(DistanceHistory).all()
    assert summary["fires_touched"] == 1
    assert summary["history_rows"] == 1
    assert len(history_rows) == 1
    assert existing_fire.name == "Updated Name"
    assert existing_fire.percent_contained == 8.0


def test_build_parser_and_print_text_paths(capsys):
    parser = benchmark_cli._build_parser()

    report_args = parser.parse_args(["report", "--cv-splits", "3"])
    capture_args = parser.parse_args(["capture", "--iterations", "2", "--run-benchmark"])
    seed_args = parser.parse_args(["seed", "--pair-limit", "2", "--replace-existing"])
    bakeoff_args = parser.parse_args(["bakeoff", "--allow-non-informative"])
    real_eval_args = parser.parse_args(["real-eval", "--download-if-missing", "--run-tuning"])
    real_eval_all_args = parser.parse_args(["real-eval", "--dataset-name", "all"])

    assert report_args.command == "report"
    assert report_args.cv_splits == 3
    assert capture_args.command == "capture"
    assert capture_args.run_benchmark is True
    assert seed_args.command == "seed"
    assert seed_args.replace_existing is True
    assert bakeoff_args.command == "bakeoff"
    assert bakeoff_args.allow_non_informative is True
    assert real_eval_args.command == "real-eval"
    assert real_eval_args.dataset_name == "uci_forest_fires"
    assert real_eval_args.download_if_missing is True
    assert real_eval_args.run_tuning is True
    assert real_eval_all_args.dataset_name == "all"

    benchmark_cli._print_text(
        {
            "command": "report",
            "report": {
                "sample_count": 2,
                "group_count": 1,
                "target_min": 0.1,
                "target_max": 0.9,
                "target_mean": 0.4,
                "target_std": 0.2,
                "target_unique_count": 2,
                "can_benchmark": False,
                "reason": "Need more data",
            },
        }
    )
    out = capsys.readouterr().out
    assert "Benchmark blocked: Need more data" in out

    benchmark_cli._print_text(
        {
            "command": "report",
            "report": {
                "sample_count": 8,
                "group_count": 3,
                "target_min": 0.1,
                "target_max": 0.9,
                "target_mean": 0.4,
                "target_std": 0.2,
                "target_unique_count": 6,
                "can_benchmark": True,
                "best_model": "random_forest",
                "baseline_model": "linear",
                "best_improvement_pct": 3.2,
                "best_mae_delta_vs_baseline": 0.0123,
                "effective_tie_with_baseline": False,
                "uplift_diagnosis": "Alternative model shows measurable uplift over baseline",
                "validation_warning": "tiny gain",
                "scorecard": {
                    "grade": "C",
                    "production_ready": False,
                    "state_of_the_art": False,
                    "assessment": "No, not yet.",
                },
            },
        }
    )
    out = capsys.readouterr().out
    assert "Selected model: random_forest" in out
    assert "Validation warning: tiny gain" in out
    assert "Scorecard: grade=C production_ready=False state_of_the_art=False" in out
    assert "Is it state-of-the-art / cutting edge? No" in out

    benchmark_cli._print_text(
        {
            "command": "capture",
            "capture": {"captures_completed": 1},
            "report": {
                "sample_count": 10,
                "target_unique_count": 4,
                "can_benchmark": True,
            },
        }
    )
    out = capsys.readouterr().out
    assert "captures_completed" in out
    assert "Post-run benchmark samples=10" in out

    benchmark_cli._print_text(
        {
            "command": "bakeoff",
            "bakeoff": {
                "bakeoff_ran": True,
                "sample_count": 12,
                "target_unique_count": 8,
                "winner_model": "xgboost",
                "baseline_model": "random_forest_baseline",
                "winner_improvement_pct_vs_baseline": 2.0,
                "winner_mae_delta_vs_baseline": 0.011,
                "effective_tie_with_baseline": False,
                "uplift_diagnosis": "Winner shows measurable uplift over baseline",
                "unavailable_optional_models": ["lightgbm"],
            },
        }
    )
    out = capsys.readouterr().out
    assert "Winner: xgboost" in out
    assert "Unavailable optional models: lightgbm" in out

    benchmark_cli._print_text(
        {
            "command": "bakeoff",
            "bakeoff": {
                "bakeoff_ran": False,
                "sample_count": 2,
                "target_unique_count": 1,
                "reason": "Not informative",
            },
        }
    )
    out = capsys.readouterr().out
    assert "Bakeoff skipped: Not informative" in out

    benchmark_cli._print_text(
        {
            "command": "real-eval",
            "real_eval": {
                "dataset_name": "uci_forest_fires",
                "dataset_rows_loaded": 35,
                "samples_used": 35,
                "include_zero_area": False,
                "spread_benchmark": {
                    "best_model": "extra_trees_ensemble",
                    "baseline_model": "random_forest_baseline",
                    "best_improvement_pct": 1.2,
                    "best_mae_delta_vs_baseline": 0.002,
                    "effective_tie_with_baseline": False,
                    "uplift_diagnosis": "Alternative model shows measurable uplift over baseline",
                },
                "bakeoff": {
                    "bakeoff_ran": True,
                    "winner_model": "xgboost_regressor",
                    "winner_improvement_pct_vs_baseline": 2.1,
                    "uplift_diagnosis": "Winner shows measurable uplift over baseline",
                },
                "tuning": {
                    "tuned_winner_model": "rf_tuned_deep",
                    "tuned_improvement_pct_vs_default_baseline": 3.4,
                },
                "scorecard": {
                    "grade": "C",
                    "production_ready": False,
                    "state_of_the_art": False,
                    "assessment": "No, not yet.",
                },
            },
        }
    )
    out = capsys.readouterr().out
    assert "Real dataset name=uci_forest_fires rows=35" in out
    assert "Tuning winner=rf_tuned_deep" in out
    assert "Is it state-of-the-art / cutting edge? No" in out
    assert "Assessment: No, not yet." in out

    benchmark_cli._print_text(
        {
            "command": "real-eval",
            "real_eval": {
                "dataset_name": "suite",
                "dataset_count": 2,
                "total_samples_used": 64,
                "best_dataset_by_spread_uplift": "uci_forest_fires",
                "dataset_results": {
                    "uci_forest_fires": {
                        "dataset_rows_loaded": 35,
                        "samples_used": 35,
                        "spread_benchmark": {
                            "best_model": "extra_trees_ensemble",
                            "best_improvement_pct": 1.2,
                        },
                        "bakeoff": {"winner_model": "xgboost_regressor"},
                        "scorecard": {"grade": "C"},
                    },
                    "uci_algerian_forest_fires": {
                        "dataset_rows_loaded": 29,
                        "samples_used": 29,
                        "spread_benchmark": {
                            "best_model": "extra_trees_ensemble",
                            "best_improvement_pct": 0.8,
                        },
                        "bakeoff": {"winner_model": "catboost_regressor"},
                        "scorecard": {"grade": "B"},
                    },
                },
                "scorecard": {
                    "grade": "C",
                    "production_ready": False,
                    "state_of_the_art": False,
                    "assessment": "No, not yet.",
                },
            },
        }
    )
    out = capsys.readouterr().out
    assert "Real dataset suite count=2" in out
    assert "[uci_algerian_forest_fires]" in out
    assert "Scorecard: grade=C production_ready=False state_of_the_art=False" in out
    assert "Assessment: No, not yet." in out


def test_main_routes_commands(monkeypatch, capsys):
    class DummyDb:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    db_instances = []

    def fake_session_local():
        db = DummyDb()
        db_instances.append(db)
        return db

    monkeypatch.setattr(benchmark_cli, "init_db", lambda: None)
    monkeypatch.setattr(benchmark_cli, "SessionLocal", fake_session_local)
    monkeypatch.setattr(
        benchmark_cli,
        "run_benchmark_report",
        lambda db, **kwargs: {"sample_count": 7, "can_benchmark": True},
    )

    async def fake_capture(db, **kwargs):
        return {"captures_completed": 2}

    monkeypatch.setattr(benchmark_cli, "capture_live_history", fake_capture)
    monkeypatch.setattr(
        benchmark_cli,
        "seed_synthetic_history",
        lambda db, **kwargs: {"pairs_seeded": 2},
    )
    monkeypatch.setattr(
        benchmark_cli, "build_historical_spread_samples", lambda db, **kwargs: [{"x": 1}]
    )
    monkeypatch.setattr(
        benchmark_cli,
        "benchmark_algorithm_bakeoff",
        lambda samples, **kwargs: {"bakeoff_ran": False, "reason": "insufficient"},
    )
    monkeypatch.setattr(
        benchmark_cli,
        "benchmark_real_world_wildfire_dataset",
        lambda *args, **kwargs: {"dataset_rows_loaded": 10, "samples_used": 8},
    )
    monkeypatch.setattr(
        benchmark_cli,
        "benchmark_real_world_wildfire_dataset_suite",
        lambda *args, **kwargs: {"dataset_name": "suite", "dataset_count": 2},
    )

    class DummyParser:
        def __init__(self, namespace):
            self._namespace = namespace

        def parse_args(self):
            return self._namespace

    printed_payloads = []
    monkeypatch.setattr(
        benchmark_cli, "_print_text", lambda payload: printed_payloads.append(payload)
    )

    monkeypatch.setattr(
        benchmark_cli,
        "_build_parser",
        lambda: DummyParser(
            SimpleNamespace(
                command="report",
                output="json",
                min_history_points=3,
                cv_splits=4,
                random_state=42,
                holdout_fraction=0.25,
            )
        ),
    )
    assert benchmark_cli.main() == 0
    json_out = capsys.readouterr().out
    assert '"command": "report"' in json_out

    monkeypatch.setattr(
        benchmark_cli,
        "_build_parser",
        lambda: DummyParser(
            SimpleNamespace(
                command="capture",
                output="text",
                iterations=2,
                interval_seconds=0,
                run_benchmark=True,
                min_history_points=3,
                cv_splits=3,
                random_state=7,
            )
        ),
    )
    assert benchmark_cli.main() == 0
    assert printed_payloads[-1]["command"] == "capture"
    assert printed_payloads[-1]["capture"]["captures_completed"] == 2
    assert "report" in printed_payloads[-1]

    monkeypatch.setattr(
        benchmark_cli,
        "_build_parser",
        lambda: DummyParser(
            SimpleNamespace(
                command="seed",
                output="text",
                pair_limit=2,
                points_per_pair=6,
                step_minutes=30,
                random_seed=9,
                replace_existing=False,
                run_benchmark=True,
                min_history_points=3,
                cv_splits=3,
                random_state=9,
            )
        ),
    )
    assert benchmark_cli.main() == 0
    assert printed_payloads[-1]["command"] == "seed"
    assert printed_payloads[-1]["seed"]["pairs_seeded"] == 2

    monkeypatch.setattr(
        benchmark_cli,
        "_build_parser",
        lambda: DummyParser(
            SimpleNamespace(
                command="bakeoff",
                output="text",
                min_history_points=3,
                cv_splits=3,
                random_state=11,
                allow_non_informative=True,
            )
        ),
    )
    assert benchmark_cli.main() == 0
    assert printed_payloads[-1]["command"] == "bakeoff"
    assert printed_payloads[-1]["bakeoff"]["bakeoff_ran"] is False

    monkeypatch.setattr(
        benchmark_cli,
        "_build_parser",
        lambda: DummyParser(
            SimpleNamespace(
                command="real-eval",
                output="text",
                dataset_name="uci_forest_fires",
                dataset_path="tests/fixtures/uci_forest_fires_sample.csv",
                download_if_missing=False,
                include_zero_area=False,
                max_rows=None,
                cv_splits=3,
                random_state=5,
                allow_non_informative=False,
                run_tuning=True,
            )
        ),
    )
    assert benchmark_cli.main() == 0
    assert printed_payloads[-1]["command"] == "real-eval"
    assert printed_payloads[-1]["real_eval"]["dataset_rows_loaded"] == 10

    monkeypatch.setattr(
        benchmark_cli,
        "_build_parser",
        lambda: DummyParser(
            SimpleNamespace(
                command="real-eval",
                output="text",
                dataset_name="all",
                dataset_path=None,
                download_if_missing=False,
                include_zero_area=False,
                max_rows=None,
                cv_splits=3,
                random_state=5,
                allow_non_informative=False,
                run_tuning=False,
            )
        ),
    )
    assert benchmark_cli.main() == 0
    assert printed_payloads[-1]["command"] == "real-eval"
    assert printed_payloads[-1]["real_eval"]["dataset_name"] == "suite"

    assert db_instances
    assert all(db.closed for db in db_instances)


def test_seed_skips_locations_not_yet_geocoded(db):
    _add_location(db)
    db.add(
        Location(location_id="pending", address="Somewhere", display_name="Pending", is_active=True)
    )
    db.commit()
    _add_fire(db)

    seeded = seed_synthetic_history(
        db, pair_limit=5, points_per_pair=4, step_minutes=30, random_seed=7, replace_existing=True
    )

    assert seeded["pairs_seeded"] == 1
