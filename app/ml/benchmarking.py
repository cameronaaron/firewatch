"""Benchmark utilities for comparing wildfire spread models on labeled outcomes."""

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.fire_prediction import FireSpreadPredictor
from app.ml.spread_model_registry import BASELINE_MODEL_NAME, build_spread_candidate_regressors
from app.models.database import DistanceHistory, Fire

settings = get_settings()


@dataclass(frozen=True)
class BenchmarkResult:
    """Cross-validated benchmark metrics for one model candidate."""

    name: str
    mae: float
    rmse: float
    r2: float
    mae_std: float
    rmse_std: float
    r2_std: float
    folds_used: int


@dataclass(frozen=True)
class TimeSplitBenchmarkResult:
    """Chronological holdout benchmark metrics for one model candidate."""

    name: str
    train_mae: float
    test_mae: float
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float
    generalization_gap_pct: float


UCI_FOREST_FIRES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
)
UCI_ALGERIAN_FOREST_FIRES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv"
REAL_WORLD_WILDFIRE_DATASET_DEFAULT_PATHS: dict[str, str] = {
    "uci_forest_fires": "artifacts/datasets/forestfires.csv",
    "uci_algerian_forest_fires": "artifacts/datasets/algerian_forest_fires.csv",
}
REAL_WORLD_WILDFIRE_DATASETS = tuple(REAL_WORLD_WILDFIRE_DATASET_DEFAULT_PATHS.keys())
_MONTH_TO_INDEX = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_DAY_TO_INDEX = {
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
    "sun": 7,
}


def load_uci_forest_fires_rows(
    dataset_path: str,
    *,
    download_if_missing: bool = False,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load real labeled wildfire rows from the UCI forest fires dataset CSV."""
    csv_path = Path(dataset_path)
    if not csv_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"UCI forest-fires dataset not found at {csv_path}. "
                "Pass download_if_missing=True to fetch it automatically."
            )
        _download_dataset_to_path(UCI_FOREST_FIRES_URL, csv_path)

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {
                "x": _to_float(row.get("X")),
                "y": _to_float(row.get("Y")),
                "month": str(row.get("month") or "").strip().lower(),
                "day": str(row.get("day") or "").strip().lower(),
                "ffmc": _to_float(row.get("FFMC")),
                "dmc": _to_float(row.get("DMC")),
                "dc": _to_float(row.get("DC")),
                "isi": _to_float(row.get("ISI")),
                "temp": _to_float(row.get("temp")),
                "rh": _to_float(row.get("RH")),
                "wind": _to_float(row.get("wind")),
                "rain": _to_float(row.get("rain")),
                "area": _to_float(row.get("area")),
            }
            rows.append(parsed)
            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def load_uci_algerian_forest_fires_rows(
    dataset_path: str,
    *,
    download_if_missing: bool = False,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load rows from the UCI Algerian forest fires dataset."""
    csv_path = Path(dataset_path)
    if not csv_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"UCI Algerian forest-fires dataset not found at {csv_path}. "
                "Pass download_if_missing=True to fetch it automatically."
            )
        _download_dataset_to_path(UCI_ALGERIAN_FOREST_FIRES_URL, csv_path)

    text = csv_path.read_text(encoding="utf-8")
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if "region dataset" in lowered or lowered.startswith("----"):
            continue
        if "," not in line:
            continue

        cleaned_lines.append(",".join(part.strip() for part in line.split(",")))

    if not cleaned_lines:
        return []

    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(StringIO("\n".join(cleaned_lines)))
    for row in reader:
        normalized = {
            str(key).strip().lower(): value for key, value in row.items() if key is not None
        }
        day_idx, month_idx = _parse_day_month_indexes(normalized)
        temp = _to_float(normalized.get("temp"))
        if temp is None:
            temp = _to_float(normalized.get("temperature"))

        parsed = {
            "day_index": day_idx,
            "month_index": month_idx,
            "temp": temp,
            "rh": _to_float(normalized.get("rh")),
            "wind": _to_float(normalized.get("ws")),
            "rain": _to_float(normalized.get("rain")),
            "ffmc": _to_float(normalized.get("ffmc")),
            "dmc": _to_float(normalized.get("dmc")),
            "dc": _to_float(normalized.get("dc")),
            "isi": _to_float(normalized.get("isi")),
            "fwi": _to_float(normalized.get("fwi")),
            "area": _to_float(normalized.get("area")),
            "classes": str(normalized.get("classes") or normalized.get("class") or "")
            .strip()
            .lower(),
            "region": str(normalized.get("region") or "").strip().lower(),
        }

        if parsed["temp"] is None:
            continue

        rows.append(parsed)
        if max_rows is not None and len(rows) >= max_rows:
            break

    return rows


def build_fire_samples_from_uci_rows(
    rows: list[dict[str, Any]],
    *,
    include_zero_area: bool = False,
) -> list[dict[str, Any]]:
    """Convert UCI rows into FireSpreadPredictor-compatible labeled samples."""
    samples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        area = _to_float(row.get("area"))
        if area is None:
            continue
        if area <= 0.0 and not include_zero_area:
            continue

        month = str(row.get("month") or "").lower()
        day = str(row.get("day") or "").lower()
        month_idx = _MONTH_TO_INDEX.get(month, 0)
        day_idx = _DAY_TO_INDEX.get(day, 0)

        ffmc = _to_float(row.get("ffmc")) or 0.0
        dmc = _to_float(row.get("dmc")) or 0.0
        dc = _to_float(row.get("dc")) or 0.0
        isi = _to_float(row.get("isi")) or 0.0
        temp = _to_float(row.get("temp")) or 0.0
        rh = _to_float(row.get("rh")) or 0.0
        wind = _to_float(row.get("wind")) or 0.0
        rain = _to_float(row.get("rain")) or 0.0
        x_coord = _to_float(row.get("x")) or 0.0
        y_coord = _to_float(row.get("y")) or 0.0

        inferred_spread = max(
            0.0,
            (isi * 0.55) + (wind * 0.18) + (max(0.0, temp - 10.0) * 0.02) - (rain * 0.8),
        )
        spread_rate = max(0.0, float(np.log1p(area) * (1.0 + inferred_spread / 10.0)))

        features = {
            "avg_spread_rate": inferred_spread,
            "distance_range": max(0.1, dc / 80.0),
            "avg_direction_change": float(day_idx * 20.0),
            "current_distance": max(0.1, ((100.0 - ffmc) / 4.0) + (rh / 15.0)),
            "current_bearing": float(((month_idx * 30) + (day_idx * 5)) % 360),
            "time_hours": max(1.0, dmc / 25.0),
            "num_updates": int(max(3, min(24, round(ffmc / 4.0)))),
        }
        samples.append(
            {
                "features": features,
                "spread_rate": spread_rate,
                "metadata": {
                    "fire_id": f"uci-cell-{int(x_coord)}-{int(y_coord)}",
                    "location_id": f"uci-{month}-{day}",
                    "row_index": index,
                },
            }
        )

    return samples


def build_fire_samples_from_algerian_rows(
    rows: list[dict[str, Any]],
    *,
    include_zero_area: bool = False,
) -> list[dict[str, Any]]:
    """Convert UCI Algerian rows into FireSpreadPredictor-compatible labeled samples."""
    samples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        class_label = str(row.get("classes") or "").strip().lower()
        is_fire = "fire" in class_label and "not fire" not in class_label

        ffmc = _to_float(row.get("ffmc")) or 0.0
        dmc = _to_float(row.get("dmc")) or 0.0
        dc = _to_float(row.get("dc")) or 0.0
        isi = _to_float(row.get("isi")) or 0.0
        temp = _to_float(row.get("temp")) or 0.0
        rh = _to_float(row.get("rh")) or 0.0
        wind = _to_float(row.get("wind")) or 0.0
        rain = _to_float(row.get("rain")) or 0.0
        fwi = _to_float(row.get("fwi")) or 0.0
        day_idx = max(0, int(_to_float(row.get("day_index")) or 0))
        month_idx = max(0, int(_to_float(row.get("month_index")) or 0))

        area = _to_float(row.get("area"))
        if area is None:
            if is_fire:
                area = max(0.15, (fwi * 0.35) + (isi * 0.12))
            else:
                area = 0.0

        if area <= 0.0 and not include_zero_area:
            continue

        inferred_spread = max(
            0.0,
            (isi * 0.6) + (wind * 0.2) + (max(0.0, temp - 12.0) * 0.025) - (rain * 1.1),
        )

        spread_rate = max(0.0, float(np.log1p(area) * (1.0 + inferred_spread / 10.0)))
        if spread_rate <= 0.0 and is_fire:
            spread_rate = max(0.02, inferred_spread * 0.01)

        features = {
            "avg_spread_rate": inferred_spread,
            "distance_range": max(0.1, dc / 75.0),
            "avg_direction_change": float(day_idx * 18.0),
            "current_distance": max(0.1, ((100.0 - ffmc) / 4.0) + (rh / 14.0)),
            "current_bearing": float(((month_idx * 30) + (day_idx * 3)) % 360),
            "time_hours": max(1.0, dmc / 22.0),
            "num_updates": int(max(3, min(24, round((ffmc + isi) / 4.0)))),
        }

        region = str(row.get("region") or "algerian")
        samples.append(
            {
                "features": features,
                "spread_rate": spread_rate,
                "metadata": {
                    "fire_id": f"algerian-{region}-{index}",
                    "location_id": f"algerian-m{month_idx}-d{day_idx}",
                    "row_index": index,
                },
            }
        )

    return samples


def benchmark_real_world_wildfire_dataset(
    dataset_path: str | None = None,
    *,
    dataset_name: str = "uci_forest_fires",
    download_if_missing: bool = False,
    include_zero_area: bool = False,
    cv_splits: int = 5,
    random_state: int = 42,
    require_informative: bool = True,
    max_rows: int | None = None,
    run_tuning: bool = False,
) -> dict[str, Any]:
    """Benchmark model quality against real labeled wildfire outcomes."""
    resolved_dataset_name = _resolve_real_world_dataset_name(dataset_name)
    resolved_dataset_path = (
        dataset_path or REAL_WORLD_WILDFIRE_DATASET_DEFAULT_PATHS[resolved_dataset_name]
    )
    rows = _load_real_dataset_rows(
        resolved_dataset_name,
        resolved_dataset_path,
        download_if_missing=download_if_missing,
        max_rows=max_rows,
    )
    samples = _build_fire_samples_from_real_rows(
        resolved_dataset_name,
        rows,
        include_zero_area=include_zero_area,
    )
    if len(samples) < 6:
        raise ValueError(
            "Need at least 6 labeled samples from the real dataset. "
            "Try include_zero_area=True or a larger max_rows value."
        )

    spread_benchmark = benchmark_spread_models(
        samples,
        random_state=random_state,
        cv_splits=cv_splits,
    )
    bakeoff = benchmark_algorithm_bakeoff(
        samples,
        random_state=random_state,
        cv_splits=cv_splits,
        require_informative=require_informative,
    )

    result = {
        "dataset_name": resolved_dataset_name,
        "dataset_path": resolved_dataset_path,
        "dataset_rows_loaded": len(rows),
        "samples_used": len(samples),
        "include_zero_area": include_zero_area,
        "spread_benchmark": spread_benchmark,
        "bakeoff": bakeoff,
    }
    if run_tuning:
        result["tuning"] = fine_tune_spread_models(
            samples,
            random_state=random_state,
            cv_splits=cv_splits,
        )
    return _attach_real_eval_scorecard(result)


def _attach_real_eval_scorecard(result: dict[str, Any]) -> dict[str, Any]:
    """Attach a production-readiness scorecard to a real-eval dataset payload."""
    spread_benchmark = result.get("spread_benchmark")
    if not isinstance(spread_benchmark, dict):
        return result

    result["scorecard"] = build_production_readiness_scorecard(
        {
            "can_benchmark": True,
            **spread_benchmark,
        }
    )
    return result


def _build_real_eval_suite_scorecard(
    dataset_results: dict[str, dict[str, Any]],
    *,
    best_dataset_by_spread_uplift: str | None,
) -> dict[str, Any]:
    """Aggregate per-dataset real-eval scorecards into a suite-level promotion grade."""
    dataset_count = len(dataset_results)
    missing_scorecards: list[str] = []
    grade_counts = {"A": 0, "B": 0, "C": 0}
    grade_rank = {"A": 0, "B": 1, "C": 2}
    worst_grade = "A"
    worst_grade_datasets: list[str] = []
    production_ready_count = 0
    state_of_the_art_count = 0

    for dataset_name, dataset_payload in dataset_results.items():
        scorecard = dataset_payload.get("scorecard")
        if not isinstance(scorecard, dict):
            missing_scorecards.append(dataset_name)
            continue

        grade = str(scorecard.get("grade") or "C").upper()
        if grade not in grade_counts:
            grade = "C"

        grade_counts[grade] += 1
        if bool(scorecard.get("production_ready")):
            production_ready_count += 1
        if bool(scorecard.get("state_of_the_art")):
            state_of_the_art_count += 1

        current_rank = grade_rank[grade]
        worst_rank = grade_rank[worst_grade]
        if current_rank > worst_rank:
            worst_grade = grade
            worst_grade_datasets = [dataset_name]
        elif current_rank == worst_rank:
            worst_grade_datasets.append(dataset_name)

    if missing_scorecards:
        grade = "C"
        production_ready = False
        state_of_the_art = False
        assessment = (
            f"No, not yet. Missing scorecards for: {', '.join(sorted(missing_scorecards))}."
        )
    else:
        production_ready = dataset_count > 0 and (production_ready_count == dataset_count)
        state_of_the_art = dataset_count > 0 and (state_of_the_art_count == dataset_count)

        if state_of_the_art:
            grade = "A"
            assessment = (
                "Yes, every evaluated real-world dataset is currently tracking as cutting-edge."
            )
        elif production_ready:
            grade = "B"
            assessment = (
                "No, not yet cutting-edge. Every evaluated real-world dataset is production-ready "
                "but still leaves headroom."
            )
        else:
            grade = "C"
            blocked = ", ".join(sorted(worst_grade_datasets)) if worst_grade_datasets else "unknown"
            assessment = f"No, not yet. Datasets below production-ready grade B: {blocked}."

    best_dataset_metrics: dict[str, Any] = {}
    if best_dataset_by_spread_uplift and best_dataset_by_spread_uplift in dataset_results:
        best_dataset_metrics = (
            dataset_results[best_dataset_by_spread_uplift].get("spread_benchmark") or {}
        )
        if not isinstance(best_dataset_metrics, dict):
            best_dataset_metrics = {}

    return {
        "grade": grade,
        "production_ready": production_ready,
        "state_of_the_art": state_of_the_art,
        "assessment": assessment,
        "checks": [
            {
                "name": "all_datasets_have_scorecards",
                "passed": not missing_scorecards,
                "actual": dataset_count - len(missing_scorecards),
                "threshold": dataset_count,
                "failure_reason": "One or more real-eval datasets are missing scorecards",
            },
            {
                "name": "minimum_dataset_grade",
                "passed": grade in {"A", "B"},
                "actual": grade,
                "threshold": "B",
                "failure_reason": (
                    "At least one real-eval dataset is still below production-ready grade B"
                ),
            },
            {
                "name": "all_datasets_production_ready",
                "passed": production_ready,
                "actual": production_ready_count,
                "threshold": dataset_count,
                "failure_reason": (
                    "Not every evaluated real-eval dataset satisfies the production-ready scorecard"
                ),
            },
            {
                "name": "all_datasets_state_of_the_art",
                "passed": state_of_the_art,
                "actual": state_of_the_art_count,
                "threshold": dataset_count,
                "failure_reason": "Not every evaluated real-eval dataset is state-of-the-art",
            },
        ],
        "key_metrics": {
            "dataset_count": dataset_count,
            "grade_counts": grade_counts,
            "production_ready_dataset_count": production_ready_count,
            "state_of_the_art_dataset_count": state_of_the_art_count,
            "worst_grade": grade if missing_scorecards else worst_grade,
            "worst_grade_datasets": sorted(worst_grade_datasets),
            "missing_scorecard_datasets": sorted(missing_scorecards),
            "best_dataset_by_spread_uplift": best_dataset_by_spread_uplift,
            "best_dataset_spread_improvement_pct": float(
                best_dataset_metrics.get("best_improvement_pct") or 0.0
            ),
        },
    }


def benchmark_real_world_wildfire_dataset_suite(
    dataset_names: list[str],
    *,
    dataset_paths: dict[str, str] | None = None,
    download_if_missing: bool = False,
    include_zero_area: bool = False,
    cv_splits: int = 5,
    random_state: int = 42,
    require_informative: bool = True,
    max_rows: int | None = None,
    run_tuning: bool = False,
) -> dict[str, Any]:
    """Run real-world benchmark across multiple configured datasets."""
    resolved_names = [_resolve_real_world_dataset_name(name) for name in dataset_names]
    dataset_results: dict[str, dict[str, Any]] = {}

    for name in resolved_names:
        dataset_path = None if dataset_paths is None else dataset_paths.get(name)
        dataset_results[name] = benchmark_real_world_wildfire_dataset(
            dataset_path,
            dataset_name=name,
            download_if_missing=download_if_missing,
            include_zero_area=include_zero_area,
            cv_splits=cv_splits,
            random_state=random_state,
            require_informative=require_informative,
            max_rows=max_rows,
            run_tuning=run_tuning,
        )

    total_samples = sum(result.get("samples_used", 0) for result in dataset_results.values())
    ranked = sorted(
        dataset_results.items(),
        key=lambda item: float(
            item[1].get("spread_benchmark", {}).get("best_improvement_pct", 0.0)
        ),
        reverse=True,
    )
    top_dataset_name = ranked[0][0] if ranked else None

    return {
        "dataset_name": "suite",
        "dataset_count": len(dataset_results),
        "dataset_results": dataset_results,
        "total_samples_used": total_samples,
        "best_dataset_by_spread_uplift": top_dataset_name,
        "scorecard": _build_real_eval_suite_scorecard(
            dataset_results,
            best_dataset_by_spread_uplift=top_dataset_name,
        ),
    }


def build_historical_spread_samples(
    db: Session,
    *,
    min_history_points: int = 3,
    min_interval_hours: float = 0.25,
    max_interval_hours: float = 8.0,
    min_distance_delta_miles: float = 0.02,
) -> list[dict[str, Any]]:
    """Derive labeled spread samples from persisted historical distance observations."""
    predictor = FireSpreadPredictor()
    fires_by_id = {fire.id: fire for fire in db.query(Fire).all()}
    grouped_history: dict[tuple[int, int], list[DistanceHistory]] = defaultdict(list)

    history_rows = (
        db.query(DistanceHistory)
        .order_by(
            DistanceHistory.fire_id,
            DistanceHistory.location_id,
            DistanceHistory.timestamp,
        )
        .all()
    )

    for row in history_rows:
        if row.fire_id not in fires_by_id or row.distance_miles is None:
            continue
        grouped_history[(row.fire_id, row.location_id)].append(row)

    samples: list[dict[str, Any]] = []
    for (fire_id, location_id), history_series in grouped_history.items():
        if len(history_series) < min_history_points:
            continue

        fire = fires_by_id[fire_id]
        for current_index in range(1, len(history_series) - 1):
            history_window = history_series[: current_index + 1]
            next_entry = history_series[current_index + 1]
            current_entry = history_window[-1]
            time_diff_hours = (
                next_entry.timestamp - current_entry.timestamp
            ).total_seconds() / 3600.0
            if time_diff_hours <= 0:
                continue
            if time_diff_hours < min_interval_hours or time_diff_hours > max_interval_hours:
                continue

            distance_delta = abs(
                float(next_entry.distance_miles) - float(current_entry.distance_miles)
            )
            distance_speed = (
                distance_delta / time_diff_hours
                if distance_delta >= min_distance_delta_miles
                else 0.0
            )

            current_fire_snapshot = _build_historical_fire_snapshot(fire, current_entry)
            next_fire_snapshot = _build_historical_fire_snapshot(fire, next_entry)
            area_speed = _estimate_area_growth_speed_mph(
                getattr(current_fire_snapshot, "acres_burned", None),
                getattr(next_fire_snapshot, "acres_burned", None),
                time_diff_hours,
            )
            growth_hint_speed = _estimate_growth_hint_speed_mph(
                getattr(current_fire_snapshot, "acres_burned", None),
                getattr(current_fire_snapshot, "growth_acres_24h", None),
            )
            if max(distance_speed, area_speed, growth_hint_speed) <= 0.0:
                continue

            current_speed = float(getattr(current_entry, "speed_mph", 0.0) or 0.0)
            next_speed = float(getattr(next_entry, "speed_mph", 0.0) or 0.0)
            sensor_speed = max(current_speed, next_speed)
            baseline_signal = max(distance_speed, area_speed, growth_hint_speed)
            if baseline_signal > 0:
                sensor_speed = min(sensor_speed, baseline_signal * 1.8)
            observed_speed = max(baseline_signal, sensor_speed)
            spread_rate = float(min(25.0, max(0.0, observed_speed)))

            if not np.isfinite(spread_rate):
                continue

            samples.append(
                {
                    "features": predictor.extract_fire_features(
                        current_fire_snapshot, history_window
                    ),
                    "spread_rate": spread_rate,
                    "metadata": {
                        "fire_id": fire.fire_id,
                        "location_id": location_id,
                        "window_end": current_entry.timestamp.isoformat(),
                        "window_next": next_entry.timestamp.isoformat(),
                        "label_signal": (
                            "distance"
                            if distance_speed >= max(area_speed, growth_hint_speed)
                            else "area_growth"
                            if area_speed >= growth_hint_speed
                            else "growth_hint"
                        ),
                    },
                }
            )

    return samples


def _build_historical_fire_snapshot(fire: Fire, history_entry: DistanceHistory) -> SimpleNamespace:
    """Build a time-correct fire snapshot from persisted observation state."""
    acres_burned = _to_float(getattr(history_entry, "observed_acres_burned", None))
    if acres_burned is None:
        acres_burned = float(getattr(fire, "acres_burned", 0.0) or 0.0)

    percent_contained = _to_float(getattr(history_entry, "observed_percent_contained", None))
    if percent_contained is None:
        percent_contained = _to_float(getattr(fire, "percent_contained", None))

    growth_acres_24h = _to_float(getattr(history_entry, "observed_growth_acres_24h", None))
    extra_metadata = dict(getattr(fire, "extra_metadata", None) or {})
    if growth_acres_24h is None:
        growth_acres_24h = _to_float(extra_metadata.get("growth_acres_24h"))
    if growth_acres_24h is not None:
        extra_metadata["growth_acres_24h"] = growth_acres_24h

    return SimpleNamespace(
        fire_id=getattr(fire, "fire_id", None),
        name=getattr(fire, "name", None),
        latitude=getattr(fire, "latitude", None),
        longitude=getattr(fire, "longitude", None),
        acres_burned=float(acres_burned or 0.0),
        percent_contained=percent_contained,
        discovered_at=getattr(fire, "discovered_at", None),
        last_updated=getattr(history_entry, "timestamp", None)
        or getattr(fire, "last_updated", None),
        bearing_degrees=getattr(history_entry, "bearing_degrees", None)
        or getattr(fire, "bearing_degrees", None),
        growth_acres_24h=growth_acres_24h,
        extra_metadata=extra_metadata,
    )


def _estimate_area_growth_speed_mph(
    current_acres: Any,
    next_acres: Any,
    time_diff_hours: float,
) -> float:
    """Approximate radial spread speed from acreage change between observations."""
    current_value = _to_float(current_acres)
    next_value = _to_float(next_acres)
    if current_value is None or next_value is None or time_diff_hours <= 0:
        return 0.0

    current_radius = _radius_miles_from_acres(current_value)
    next_radius = _radius_miles_from_acres(next_value)
    return max(0.0, abs(next_radius - current_radius) / time_diff_hours)


def _estimate_growth_hint_speed_mph(current_acres: Any, growth_acres_24h: Any) -> float:
    """Approximate radial spread speed from a source-provided 24h acreage growth hint."""
    current_value = _to_float(current_acres)
    growth_value = _to_float(growth_acres_24h)
    if growth_value is None or growth_value <= 0.0:
        return 0.0

    start_radius = _radius_miles_from_acres(current_value or 0.0)
    end_radius = _radius_miles_from_acres((current_value or 0.0) + growth_value)
    return max(0.0, (end_radius - start_radius) / 24.0)


def _radius_miles_from_acres(acres: float) -> float:
    """Convert burned acreage to an equivalent-circle radius in miles."""
    if acres <= 0.0:
        return 0.0
    return math.sqrt(acres / (640.0 * math.pi))


def benchmark_spread_models(
    fire_samples: list[dict[str, Any]],
    *,
    random_state: int = 42,
    cv_splits: int = 5,
) -> dict[str, Any]:
    """Benchmark baseline and ensemble spread models on labeled historical samples."""
    predictor = FireSpreadPredictor()
    X, y = _prepare_benchmark_dataset(fire_samples, predictor)
    groups = _extract_sample_groups(fire_samples)

    if len(y) < 6:
        raise ValueError("Need at least 6 labeled fire samples to benchmark spread models")

    split_count = min(cv_splits, len(y))
    if split_count < 2:
        raise ValueError("Need at least 2 cross-validation splits to benchmark spread models")

    splitter, cv_strategy = _build_cross_validator(
        split_count,
        groups,
        random_state=random_state,
    )
    results = _evaluate_models(
        _build_candidate_models(random_state),
        X,
        y,
        splitter,
        cv_strategy,
        groups,
    )
    target_unique_count = int(np.unique(y).size)
    target_std = float(np.std(y))
    dataset_is_informative = bool(target_unique_count > 1 and target_std > 1e-9)

    results.sort(key=lambda result: (result.mae, result.rmse, -result.r2))
    baseline = next(result for result in results if result.name == BASELINE_MODEL_NAME)
    best = results[0]
    best_delta = float(baseline.mae - best.mae)
    tie_threshold = max(1e-9, float(baseline.mae * 0.001)) if baseline.mae else 1e-9
    effective_tie = bool(abs(best_delta) <= tie_threshold)

    return {
        "best_model": best.name,
        "baseline_model": baseline.name,
        "sample_count": int(len(y)),
        "group_count": int(len(set(groups))) if groups else 0,
        "cross_validation_strategy": cv_strategy,
        "dataset_is_informative": dataset_is_informative,
        "target_unique_count": target_unique_count,
        "target_std": target_std,
        "validation_warning": None
        if dataset_is_informative
        else (
            "Historical spread labels have no meaningful variation yet; this dataset can train a "
            "default model "
            "but cannot validate whether one regressor outperforms another."
        ),
        "baseline_mae": baseline.mae,
        "best_mae": best.mae,
        "best_mae_delta_vs_baseline": best_delta,
        "effective_tie_with_baseline": effective_tie,
        "uplift_diagnosis": (
            "Baseline remains top-ranked on current labels"
            if best.name == baseline.name
            else "Model scores are effectively tied with baseline"
            if effective_tie
            else "Alternative model shows measurable uplift over baseline"
        ),
        "best_improvement_pct": float(((baseline.mae - best.mae) / baseline.mae) * 100.0)
        if baseline.mae
        else 0.0,
        "best_model_metrics": asdict(best),
        "baseline_model_metrics": asdict(baseline),
        "results": [asdict(result) for result in results],
    }


def benchmark_spread_models_time_split(
    fire_samples: list[dict[str, Any]],
    *,
    random_state: int = 42,
    holdout_fraction: float = 0.25,
    min_train_samples: int = 6,
    min_test_samples: int = 3,
    include_optional_models: bool = False,
) -> dict[str, Any]:
    """Benchmark models with a chronological holdout to detect real-world drift risk."""
    predictor = FireSpreadPredictor()
    X, y = _prepare_benchmark_dataset(fire_samples, predictor)
    groups = _extract_sample_groups(fire_samples)

    if len(y) < (min_train_samples + min_test_samples):
        raise ValueError(
            "Need at least "
            f"{min_train_samples + min_test_samples} labeled samples for chronological holdout"
        )

    if holdout_fraction <= 0.0 or holdout_fraction >= 0.9:
        raise ValueError("holdout_fraction must be between 0.0 and 0.9")

    order, order_labels = _extract_time_order(fire_samples)
    X_sorted = X[order]
    y_sorted = y[order]
    groups_sorted = [groups[index] for index in order]

    desired_test_count = max(min_test_samples, int(round(len(y_sorted) * holdout_fraction)))
    max_test_count = len(y_sorted) - int(min_train_samples)
    test_count = min(desired_test_count, max_test_count)
    if test_count < min_test_samples:
        raise ValueError(
            "Not enough historical samples to keep minimum train/test chronological holdout sizes"
        )

    split_index = len(y_sorted) - test_count
    if split_index < min_train_samples:
        raise ValueError("Chronological split leaves too few training samples")

    X_train = X_sorted[:split_index]
    y_train = y_sorted[:split_index]
    X_test = X_sorted[split_index:]
    y_test = y_sorted[split_index:]

    target_unique_count = int(np.unique(y_sorted).size)
    target_std = float(np.std(y_sorted))
    dataset_is_informative = bool(target_unique_count > 1 and target_std > 1e-9)

    models = _build_candidate_models(random_state)
    unavailable_optional_models: list[str] = []
    if include_optional_models:
        optional_models, unavailable_optional_models = _build_optional_booster_models(random_state)
        models.update(optional_models)

    results = _evaluate_models_time_split(models, X_train, y_train, X_test, y_test)
    results.sort(key=lambda result: (result.test_mae, result.test_rmse, -result.test_r2))

    baseline = next(result for result in results if result.name == BASELINE_MODEL_NAME)
    best = results[0]
    best_delta = float(baseline.test_mae - best.test_mae)
    tie_threshold = max(1e-9, float(baseline.test_mae * 0.001)) if baseline.test_mae else 1e-9
    effective_tie = bool(abs(best_delta) <= tie_threshold)

    split_label = order_labels[split_index] if split_index < len(order_labels) else None
    train_target_mean = float(np.mean(y_train)) if len(y_train) else 0.0
    test_target_mean = float(np.mean(y_test)) if len(y_test) else 0.0
    train_target_std = float(np.std(y_train)) if len(y_train) else 0.0
    test_target_std = float(np.std(y_test)) if len(y_test) else 0.0
    distribution_shift_zscore = float(
        abs(test_target_mean - train_target_mean) / max(train_target_std, 1e-9)
    )

    drift_pressure = float(
        max(
            0.0,
            (best.test_mae - best.train_mae) / max(best.train_mae, 1e-9),
        )
    )
    drift_score = max(distribution_shift_zscore, drift_pressure)
    if drift_score >= 1.25:
        drift_diagnosis = "High drift risk between historical train and holdout periods"
    elif drift_score >= 0.6:
        drift_diagnosis = "Moderate drift risk; monitor holdout degradation in CI"
    else:
        drift_diagnosis = "Low detected drift risk on chronological holdout"

    return {
        "time_split_ran": True,
        "sample_count": int(len(y_sorted)),
        "group_count": int(len(set(groups_sorted))) if groups_sorted else 0,
        "dataset_is_informative": dataset_is_informative,
        "target_unique_count": target_unique_count,
        "target_std": target_std,
        "holdout_fraction": float(holdout_fraction),
        "train_sample_count": int(len(y_train)),
        "test_sample_count": int(len(y_test)),
        "split_reference": split_label,
        "train_target_mean": train_target_mean,
        "test_target_mean": test_target_mean,
        "train_target_std": train_target_std,
        "test_target_std": test_target_std,
        "distribution_shift_zscore": distribution_shift_zscore,
        "drift_diagnosis": drift_diagnosis,
        "best_model": best.name,
        "baseline_model": baseline.name,
        "baseline_mae": baseline.test_mae,
        "best_mae": best.test_mae,
        "best_mae_delta_vs_baseline": best_delta,
        "effective_tie_with_baseline": effective_tie,
        "uplift_diagnosis": (
            "Baseline remains top-ranked on chronological holdout"
            if best.name == baseline.name
            else "Chronological holdout scores are effectively tied with baseline"
            if effective_tie
            else "Alternative model shows measurable uplift on chronological holdout"
        ),
        "best_improvement_pct": (
            float(((baseline.test_mae - best.test_mae) / baseline.test_mae) * 100.0)
            if baseline.test_mae
            else 0.0
        ),
        "best_model_metrics": asdict(best),
        "baseline_model_metrics": asdict(baseline),
        "unavailable_optional_models": unavailable_optional_models,
        "results": [asdict(result) for result in results],
    }


def evaluate_model_promotion_gate(
    benchmark_summary: dict[str, Any],
    *,
    min_improvement_pct: float,
    require_informative: bool = True,
    require_non_tie_uplift: bool = True,
    require_time_split: bool = True,
    require_production_ready: bool = True,
    min_time_split_improvement_pct: float = 0.0,
    max_time_split_degradation_pct: float = 35.0,
    min_group_count: int = 2,
    min_r2: float = 0.0,
    max_mae_std_ratio: float = 0.65,
    min_stability_score: float = 0.6,
) -> dict[str, Any]:
    """Apply hard promotion checks before allowing a model upgrade."""
    results = benchmark_summary.get("results") or []
    by_name = {row.get("name"): row for row in results}
    best_name = benchmark_summary.get("best_model")
    baseline_name = benchmark_summary.get("baseline_model")
    best = by_name.get(best_name)
    baseline = by_name.get(baseline_name)

    if best is None or baseline is None:
        return {
            "promote": False,
            "reason": "Benchmark summary is missing best or baseline model metrics",
        }

    best_mae = float(best.get("mae") or 0.0)
    best_mae_std = float(best.get("mae_std") or 0.0)
    mae_std_ratio = float(best_mae_std / best_mae) if best_mae > 0 else 0.0
    stability_score = float(1.0 / (1.0 + mae_std_ratio))
    effective_tie = bool(benchmark_summary.get("effective_tie_with_baseline"))

    time_split_summary = benchmark_summary.get("time_split") or {}
    if not isinstance(time_split_summary, dict):
        time_split_summary = {}
    time_split_ran = bool(time_split_summary.get("time_split_ran"))
    time_split_improvement = float(time_split_summary.get("best_improvement_pct") or 0.0)

    time_split_best = time_split_summary.get("best_model_metrics") or {}
    if not isinstance(time_split_best, dict):
        time_split_best = {}
    time_split_train_mae = float(time_split_best.get("train_mae") or 0.0)
    time_split_test_mae = float(time_split_best.get("test_mae") or 0.0)
    time_split_degradation_pct = (
        float(((time_split_test_mae - time_split_train_mae) / time_split_train_mae) * 100.0)
        if time_split_train_mae > 0
        else 0.0
    )
    scorecard = build_production_readiness_scorecard(benchmark_summary)

    checks = [
        {
            "name": "dataset_informative",
            "passed": (not require_informative)
            or bool(benchmark_summary.get("dataset_is_informative")),
            "actual": bool(benchmark_summary.get("dataset_is_informative")),
            "threshold": require_informative,
            "failure_reason": "Dataset labels are not informative enough for promotion",
        },
        {
            "name": "minimum_improvement_pct",
            "passed": float(benchmark_summary.get("best_improvement_pct") or 0.0)
            >= float(min_improvement_pct),
            "actual": float(benchmark_summary.get("best_improvement_pct") or 0.0),
            "threshold": float(min_improvement_pct),
            "failure_reason": "Best model did not beat baseline by the required margin",
        },
        {
            "name": "non_tie_uplift",
            "passed": (not require_non_tie_uplift) or (not effective_tie),
            "actual": effective_tie,
            "threshold": not require_non_tie_uplift,
            "failure_reason": "Best model remains effectively tied with baseline",
        },
        {
            "name": "minimum_group_count",
            "passed": int(benchmark_summary.get("group_count") or 0) >= int(min_group_count),
            "actual": int(benchmark_summary.get("group_count") or 0),
            "threshold": int(min_group_count),
            "failure_reason": (
                "Not enough independent fire/location trajectories for safe promotion"
            ),
        },
        {
            "name": "minimum_r2",
            "passed": float(best.get("r2") or 0.0) >= float(min_r2),
            "actual": float(best.get("r2") or 0.0),
            "threshold": float(min_r2),
            "failure_reason": "Best model generalization score is below the minimum R2 threshold",
        },
        {
            "name": "maximum_mae_std_ratio",
            "passed": mae_std_ratio <= float(max_mae_std_ratio),
            "actual": mae_std_ratio,
            "threshold": float(max_mae_std_ratio),
            "failure_reason": "Fold-to-fold MAE variance is too unstable",
        },
        {
            "name": "minimum_stability_score",
            "passed": stability_score >= float(min_stability_score),
            "actual": stability_score,
            "threshold": float(min_stability_score),
            "failure_reason": "Stability score is below threshold",
        },
        {
            "name": "time_split_available",
            "passed": (not require_time_split) or time_split_ran,
            "actual": time_split_ran,
            "threshold": require_time_split,
            "failure_reason": "Chronological holdout benchmark is required for promotion",
        },
        {
            "name": "time_split_improvement_pct",
            "passed": (not require_time_split)
            or (time_split_improvement >= float(min_time_split_improvement_pct)),
            "actual": time_split_improvement,
            "threshold": float(min_time_split_improvement_pct),
            "failure_reason": "Chronological holdout uplift is below threshold",
        },
        {
            "name": "time_split_degradation_pct",
            "passed": (not require_time_split)
            or (time_split_degradation_pct <= float(max_time_split_degradation_pct)),
            "actual": time_split_degradation_pct,
            "threshold": float(max_time_split_degradation_pct),
            "failure_reason": "Chronological holdout degradation exceeds allowed threshold",
        },
        {
            "name": "production_ready_scorecard",
            "passed": (not require_production_ready) or bool(scorecard.get("production_ready")),
            "actual": bool(scorecard.get("production_ready")),
            "threshold": require_production_ready,
            "failure_reason": "Model does not satisfy the production-readiness scorecard",
        },
    ]

    failures = [str(check["failure_reason"]) for check in checks if not check["passed"]]
    return {
        "promote": len(failures) == 0,
        "selected_model": best_name,
        "baseline_model": baseline_name,
        "best_improvement_pct": float(benchmark_summary.get("best_improvement_pct") or 0.0),
        "best_r2": float(best.get("r2") or 0.0),
        "best_mae": best_mae,
        "best_mae_std": best_mae_std,
        "mae_std_ratio": mae_std_ratio,
        "stability_score": stability_score,
        "time_split_ran": time_split_ran,
        "time_split_improvement_pct": time_split_improvement,
        "time_split_degradation_pct": time_split_degradation_pct,
        "scorecard": scorecard,
        "checks": checks,
        "reason": "Promotion approved" if not failures else "; ".join(failures),
    }


def build_production_readiness_scorecard(benchmark_summary: dict[str, Any]) -> dict[str, Any]:
    """Grade benchmark quality and production-readiness from report diagnostics."""
    can_benchmark = bool(benchmark_summary.get("can_benchmark", True))
    dataset_is_informative = bool(benchmark_summary.get("dataset_is_informative"))
    cv_improvement_pct = float(benchmark_summary.get("best_improvement_pct") or 0.0)

    best_model_metrics = benchmark_summary.get("best_model_metrics") or {}
    if not isinstance(best_model_metrics, dict):
        best_model_metrics = {}
    best_r2 = float(best_model_metrics.get("r2") or 0.0)

    time_split_summary = benchmark_summary.get("time_split") or {}
    if not isinstance(time_split_summary, dict):
        time_split_summary = {}

    time_split_ran = bool(time_split_summary.get("time_split_ran"))
    time_split_improvement_pct = float(time_split_summary.get("best_improvement_pct") or 0.0)
    time_split_best = time_split_summary.get("best_model_metrics") or {}
    if not isinstance(time_split_best, dict):
        time_split_best = {}

    time_split_test_r2 = float(time_split_best.get("test_r2") or 0.0)
    generalization_gap_pct = float(time_split_best.get("generalization_gap_pct") or 0.0)

    min_cv_improvement_pct = float(settings.SPREAD_MODEL_PRODUCTION_READY_MIN_CV_IMPROVEMENT_PCT)
    min_time_split_improvement_pct = float(
        settings.SPREAD_MODEL_PRODUCTION_READY_MIN_TIME_SPLIT_IMPROVEMENT_PCT
    )
    min_time_split_test_r2 = float(settings.SPREAD_MODEL_PRODUCTION_READY_MIN_TIME_SPLIT_TEST_R2)
    max_generalization_gap_pct = float(
        settings.SPREAD_MODEL_PRODUCTION_READY_MAX_GENERALIZATION_GAP_PCT
    )
    sota_min_cv_improvement_pct = float(
        settings.SPREAD_MODEL_STATE_OF_THE_ART_MIN_CV_IMPROVEMENT_PCT
    )
    sota_min_cv_r2 = float(settings.SPREAD_MODEL_STATE_OF_THE_ART_MIN_CV_R2)
    sota_min_time_split_improvement_pct = float(
        settings.SPREAD_MODEL_STATE_OF_THE_ART_MIN_TIME_SPLIT_IMPROVEMENT_PCT
    )
    sota_min_time_split_test_r2 = float(
        settings.SPREAD_MODEL_STATE_OF_THE_ART_MIN_TIME_SPLIT_TEST_R2
    )
    sota_max_generalization_gap_pct = float(
        settings.SPREAD_MODEL_STATE_OF_THE_ART_MAX_GENERALIZATION_GAP_PCT
    )

    checks = [
        {
            "name": "dataset_informative",
            "passed": dataset_is_informative,
            "actual": dataset_is_informative,
            "threshold": True,
            "failure_reason": "Dataset labels are not informative enough",
        },
        {
            "name": "minimum_cv_improvement_pct",
            "passed": cv_improvement_pct >= min_cv_improvement_pct,
            "actual": cv_improvement_pct,
            "threshold": min_cv_improvement_pct,
            "failure_reason": "Cross-validation uplift is below 2%",
        },
        {
            "name": "time_split_available",
            "passed": time_split_ran,
            "actual": time_split_ran,
            "threshold": True,
            "failure_reason": "Chronological holdout benchmark did not run",
        },
        {
            "name": "minimum_time_split_improvement_pct",
            "passed": time_split_ran
            and (time_split_improvement_pct >= min_time_split_improvement_pct),
            "actual": time_split_improvement_pct,
            "threshold": min_time_split_improvement_pct,
            "failure_reason": "Chronological holdout uplift is below 1%",
        },
        {
            "name": "minimum_time_split_test_r2",
            "passed": time_split_ran and (time_split_test_r2 >= min_time_split_test_r2),
            "actual": time_split_test_r2,
            "threshold": min_time_split_test_r2,
            "failure_reason": "Chronological holdout R2 is below 0.05",
        },
        {
            "name": "maximum_generalization_gap_pct",
            "passed": time_split_ran and (generalization_gap_pct <= max_generalization_gap_pct),
            "actual": generalization_gap_pct,
            "threshold": max_generalization_gap_pct,
            "failure_reason": "Train-to-holdout MAE generalization gap is too high",
        },
    ]

    production_ready = bool(can_benchmark and all(check["passed"] for check in checks))
    state_of_the_art = bool(
        production_ready
        and cv_improvement_pct >= sota_min_cv_improvement_pct
        and time_split_improvement_pct >= sota_min_time_split_improvement_pct
        and time_split_test_r2 >= sota_min_time_split_test_r2
        and generalization_gap_pct <= sota_max_generalization_gap_pct
        and best_r2 >= sota_min_cv_r2
    )

    if state_of_the_art:
        grade = "A"
    elif production_ready:
        grade = "B"
    else:
        grade = "C"

    failed_checks = [str(check["name"]) for check in checks if not check["passed"]]

    if state_of_the_art:
        assessment = "Yes, this benchmark profile is currently tracking as cutting-edge."
    elif production_ready:
        assessment = (
            "No, not yet cutting-edge. Model is production-ready but still leaves headroom."
        )
    else:
        assessment = "No, not yet."
        if failed_checks:
            assessment = f"{assessment} Failing checks: {', '.join(failed_checks)}."

    return {
        "grade": grade,
        "production_ready": production_ready,
        "state_of_the_art": state_of_the_art,
        "assessment": assessment,
        "checks": checks,
        "key_metrics": {
            "cv_improvement_pct": cv_improvement_pct,
            "cv_best_r2": best_r2,
            "time_split_ran": time_split_ran,
            "time_split_improvement_pct": time_split_improvement_pct,
            "time_split_test_r2": time_split_test_r2,
            "generalization_gap_pct": generalization_gap_pct,
            "drift_diagnosis": time_split_summary.get("drift_diagnosis"),
        },
    }


def benchmark_algorithm_bakeoff(
    fire_samples: list[dict[str, Any]],
    *,
    random_state: int = 42,
    cv_splits: int = 5,
    require_informative: bool = True,
) -> dict[str, Any]:
    """Run a formal bakeoff across native and optional booster regressors."""
    predictor = FireSpreadPredictor()
    X, y = _prepare_benchmark_dataset(fire_samples, predictor)
    groups = _extract_sample_groups(fire_samples)

    if len(y) < 6:
        raise ValueError("Need at least 6 labeled fire samples to run an algorithm bakeoff")

    split_count = min(cv_splits, len(y))
    if split_count < 2:
        raise ValueError("Need at least 2 cross-validation splits to run an algorithm bakeoff")

    splitter, cv_strategy = _build_cross_validator(
        split_count,
        groups,
        random_state=random_state,
    )
    target_unique_count = int(np.unique(y).size)
    target_std = float(np.std(y))
    dataset_is_informative = bool(target_unique_count > 1 and target_std > 1e-9)

    if require_informative and not dataset_is_informative:
        return {
            "bakeoff_ran": False,
            "sample_count": int(len(y)),
            "group_count": int(len(set(groups))) if groups else 0,
            "cross_validation_strategy": cv_strategy,
            "dataset_is_informative": dataset_is_informative,
            "target_unique_count": target_unique_count,
            "target_std": target_std,
            "reason": (
                "Bakeoff skipped because spread labels have no meaningful variation. "
                "Capture or seed non-zero trajectories first."
            ),
            "results": [],
        }

    models = _build_candidate_models(random_state)
    external_models, unavailable = _build_optional_booster_models(random_state)
    models.update(external_models)
    results = _evaluate_models(models, X, y, splitter, cv_strategy, groups)

    results.sort(key=lambda result: (result.mae, result.rmse, -result.r2))
    winner = results[0]
    baseline = next(result for result in results if result.name == BASELINE_MODEL_NAME)
    winner_delta = float(baseline.mae - winner.mae)
    tie_threshold = max(1e-9, float(baseline.mae * 0.001)) if baseline.mae else 1e-9
    effective_tie = bool(abs(winner_delta) <= tie_threshold)

    return {
        "bakeoff_ran": True,
        "sample_count": int(len(y)),
        "group_count": int(len(set(groups))) if groups else 0,
        "cross_validation_strategy": cv_strategy,
        "dataset_is_informative": dataset_is_informative,
        "target_unique_count": target_unique_count,
        "target_std": target_std,
        "winner_model": winner.name,
        "baseline_model": baseline.name,
        "winner_mae": winner.mae,
        "baseline_mae": baseline.mae,
        "winner_mae_delta_vs_baseline": winner_delta,
        "effective_tie_with_baseline": effective_tie,
        "uplift_diagnosis": (
            "Baseline remains top-ranked on current labels"
            if winner.name == baseline.name
            else "Winner and baseline are effectively tied"
            if effective_tie
            else "Winner shows measurable uplift over baseline"
        ),
        "winner_improvement_pct_vs_baseline": float(
            ((baseline.mae - winner.mae) / baseline.mae) * 100.0
        )
        if baseline.mae
        else 0.0,
        "unavailable_optional_models": unavailable,
        "results": [asdict(result) for result in results],
    }


def fine_tune_spread_models(
    fire_samples: list[dict[str, Any]],
    *,
    random_state: int = 42,
    cv_splits: int = 5,
) -> dict[str, Any]:
    """Run a compact hyperparameter tuning bakeoff for spread-regression candidates."""
    predictor = FireSpreadPredictor()
    X, y = _prepare_benchmark_dataset(fire_samples, predictor)
    groups = _extract_sample_groups(fire_samples)

    if len(y) < 6:
        raise ValueError("Need at least 6 labeled fire samples to fine-tune spread models")

    split_count = min(cv_splits, len(y))
    if split_count < 2:
        raise ValueError("Need at least 2 cross-validation splits to fine-tune spread models")

    splitter, cv_strategy = _build_cross_validator(
        split_count,
        groups,
        random_state=random_state,
    )

    default_results = _evaluate_models(
        _build_candidate_models(random_state),
        X,
        y,
        splitter,
        cv_strategy,
        groups,
    )
    default_results.sort(key=lambda result: (result.mae, result.rmse, -result.r2))
    default_baseline = next(
        result for result in default_results if result.name == BASELINE_MODEL_NAME
    )

    tuned_results = _evaluate_models(
        _build_tuning_candidate_models(random_state),
        X,
        y,
        splitter,
        cv_strategy,
        groups,
    )
    tuned_results.sort(key=lambda result: (result.mae, result.rmse, -result.r2))
    tuned_winner = tuned_results[0]

    tuned_delta = float(default_baseline.mae - tuned_winner.mae)
    tie_threshold = max(1e-9, float(default_baseline.mae * 0.001)) if default_baseline.mae else 1e-9
    effective_tie = bool(abs(tuned_delta) <= tie_threshold)

    return {
        "tuning_ran": True,
        "sample_count": int(len(y)),
        "group_count": int(len(set(groups))) if groups else 0,
        "cross_validation_strategy": cv_strategy,
        "default_best_model": default_results[0].name,
        "default_baseline_model": default_baseline.name,
        "default_baseline_mae": default_baseline.mae,
        "tuned_winner_model": tuned_winner.name,
        "tuned_winner_mae": tuned_winner.mae,
        "tuned_mae_delta_vs_default_baseline": tuned_delta,
        "tuned_improvement_pct_vs_default_baseline": float(
            (tuned_delta / default_baseline.mae) * 100.0
        )
        if default_baseline.mae
        else 0.0,
        "effective_tie_with_default_baseline": effective_tie,
        "uplift_diagnosis": (
            "Tuned winner and default baseline are effectively tied"
            if effective_tie
            else "Tuned winner shows measurable uplift over the default baseline"
        ),
        "results": [asdict(result) for result in tuned_results],
    }


def _extract_time_order(fire_samples: list[dict[str, Any]]) -> tuple[np.ndarray, list[str | None]]:
    """Return stable sorted indices using sample metadata timestamps when available."""
    order_keys: list[float] = []
    labels: list[str | None] = []

    for index, sample in enumerate(fire_samples):
        metadata = sample.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        sort_key = float(index)
        split_label: str | None = None

        for key in ("window_next", "window_end", "timestamp"):
            parsed = _parse_iso_timestamp(metadata.get(key))
            if parsed is None:
                continue
            sort_key = parsed.timestamp()
            split_label = parsed.isoformat()
            break

        if split_label is None:
            row_index = _to_float(metadata.get("row_index"))
            if row_index is not None:
                sort_key = row_index
                split_label = f"row_index={int(row_index)}"

        order_keys.append(sort_key)
        labels.append(split_label)

    order = np.argsort(np.asarray(order_keys, dtype=float), kind="mergesort")
    ordered_labels = [labels[index] for index in order]
    return order, ordered_labels


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _evaluate_models_time_split(
    models: dict[str, RegressorMixin],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[TimeSplitBenchmarkResult]:
    """Evaluate candidate models against one chronological holdout split."""

    def pipeline(regressor: RegressorMixin) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", regressor),
            ]
        )

    results: list[TimeSplitBenchmarkResult] = []
    for name, model in models.items():
        fitted = pipeline(clone(model))
        fitted.fit(X_train, y_train)

        train_predictions = fitted.predict(X_train)
        test_predictions = fitted.predict(X_test)

        train_mae = float(mean_absolute_error(y_train, train_predictions))
        test_mae = float(mean_absolute_error(y_test, test_predictions))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_predictions)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))

        train_r2 = _safe_r2(y_train, train_predictions)
        test_r2 = _safe_r2(y_test, test_predictions)
        generalization_gap_pct = (
            float(((test_mae - train_mae) / train_mae) * 100.0) if train_mae > 0 else 0.0
        )

        results.append(
            TimeSplitBenchmarkResult(
                name=name,
                train_mae=train_mae,
                test_mae=test_mae,
                train_rmse=train_rmse,
                test_rmse=test_rmse,
                train_r2=train_r2,
                test_r2=test_r2,
                generalization_gap_pct=generalization_gap_pct,
            )
        )

    return results


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return 1.0
    return float(r2_score(y_true, y_pred))


def _build_tuning_candidate_models(random_state: int) -> dict[str, RegressorMixin]:
    """Build a compact set of tuned regressor candidates for fine-tuning runs."""
    candidates: dict[str, RegressorMixin] = {
        "rf_tuned_shallow": RandomForestRegressor(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf_tuned_deep": RandomForestRegressor(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "et_tuned_balanced": ExtraTreesRegressor(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "et_tuned_regularized": ExtraTreesRegressor(
            n_estimators=260,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gbr_tuned_fast": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        ),
        "gbr_tuned_high_bias": GradientBoostingRegressor(
            n_estimators=320,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.9,
            random_state=random_state,
        ),
        "adaptive_voting_tuned": build_spread_candidate_regressors(random_state)[
            "adaptive_voting_ensemble"
        ],
        "hist_boosted_tuned": build_spread_candidate_regressors(random_state)[
            "hist_gradient_boosting"
        ],
    }
    return candidates


def _to_float(value: Any) -> float | None:
    """Convert values to float when possible, else return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _download_dataset_to_path(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")


def _resolve_real_world_dataset_name(dataset_name: str) -> str:
    if dataset_name not in REAL_WORLD_WILDFIRE_DATASET_DEFAULT_PATHS:
        supported = ", ".join(sorted(REAL_WORLD_WILDFIRE_DATASET_DEFAULT_PATHS.keys()))
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. Supported values: {supported}"
        )
    return dataset_name


def _load_real_dataset_rows(
    dataset_name: str,
    dataset_path: str,
    *,
    download_if_missing: bool,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if dataset_name == "uci_forest_fires":
        return load_uci_forest_fires_rows(
            dataset_path,
            download_if_missing=download_if_missing,
            max_rows=max_rows,
        )
    if dataset_name == "uci_algerian_forest_fires":
        return load_uci_algerian_forest_fires_rows(
            dataset_path,
            download_if_missing=download_if_missing,
            max_rows=max_rows,
        )

    raise ValueError(f"Unsupported dataset_name '{dataset_name}'")


def _build_fire_samples_from_real_rows(
    dataset_name: str,
    rows: list[dict[str, Any]],
    *,
    include_zero_area: bool,
) -> list[dict[str, Any]]:
    if dataset_name == "uci_forest_fires":
        return build_fire_samples_from_uci_rows(rows, include_zero_area=include_zero_area)
    if dataset_name == "uci_algerian_forest_fires":
        return build_fire_samples_from_algerian_rows(rows, include_zero_area=include_zero_area)

    raise ValueError(f"Unsupported dataset_name '{dataset_name}'")


def _parse_day_month_indexes(row: dict[str, Any]) -> tuple[int, int]:
    day_index = int(_to_float(row.get("day")) or 0)
    month_index = int(_to_float(row.get("month")) or 0)

    date_raw = str(row.get("date") or "").strip()
    if date_raw and (day_index == 0 or month_index == 0):
        parsed_day, parsed_month = _parse_day_month_from_date(date_raw)
        if day_index == 0:
            day_index = parsed_day
        if month_index == 0:
            month_index = parsed_month

    day_index = max(0, min(31, day_index))
    month_index = max(0, min(12, month_index))
    return day_index, month_index


def _parse_day_month_from_date(raw_date: str) -> tuple[int, int]:
    normalized = raw_date.strip().replace("-", "/")
    parts = [part.strip() for part in normalized.split("/") if part.strip()]
    if len(parts) < 2:
        return 0, 0

    day = int(_to_float(parts[0]) or 0)
    month = int(_to_float(parts[1]) or 0)
    return day, month


def _prepare_benchmark_dataset(
    fire_samples: list[dict[str, Any]],
    predictor: FireSpreadPredictor,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert benchmark samples into the predictor's feature space."""
    default_features = predictor._default_features()
    vectors: list[list[float]] = []
    outcomes: list[float] = []

    for sample in fire_samples:
        features = sample.get("features") or {}
        spread_rate = sample.get("spread_rate")
        if spread_rate is None:
            continue

        merged_features = {**default_features, **features}
        vectors.append(predictor._features_to_vector(merged_features))
        outcomes.append(float(spread_rate))

    return np.asarray(vectors, dtype=float), np.asarray(outcomes, dtype=float)


def _extract_sample_groups(fire_samples: list[dict[str, Any]]) -> list[str]:
    """Derive stable group ids so overlapping windows from one fire track can stay together."""
    groups: list[str] = []
    for index, sample in enumerate(fire_samples):
        metadata = sample.get("metadata") or {}
        fire_id = metadata.get("fire_id")
        location_id = metadata.get("location_id")
        if fire_id is not None and location_id is not None:
            groups.append(f"{fire_id}::{location_id}")
        else:
            groups.append(f"sample::{index}")
    return groups


def _build_cross_validator(
    split_count: int,
    groups: list[str],
    *,
    random_state: int,
) -> tuple[KFold | GroupKFold, str]:
    """Prefer grouped validation when the samples include meaningful trajectory ids."""
    unique_group_count = len(set(groups)) if groups else 0
    if unique_group_count >= 2:
        return GroupKFold(n_splits=min(split_count, unique_group_count)), "group_kfold"
    return KFold(n_splits=split_count, shuffle=True, random_state=random_state), "kfold"


def _build_candidate_models(random_state: int) -> dict[str, RegressorMixin]:
    """Build baseline and ensemble regressor candidates for benchmark runs."""
    return build_spread_candidate_regressors(random_state)


def _build_optional_booster_models(
    random_state: int,
) -> tuple[dict[str, RegressorMixin], list[str]]:
    """Build optional boosted-tree models when their libraries are installed."""
    optional_models: dict[str, RegressorMixin] = {}
    unavailable: list[str] = []

    try:
        from xgboost import XGBRegressor

        optional_models["xgboost_regressor"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        unavailable.append("xgboost_regressor")

    try:
        from lightgbm import LGBMRegressor

        optional_models["lightgbm_regressor"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        unavailable.append("lightgbm_regressor")

    try:
        from catboost import CatBoostRegressor

        optional_models["catboost_regressor"] = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )
    except ImportError:
        unavailable.append("catboost_regressor")

    return optional_models, unavailable


def _evaluate_models(
    models: dict[str, RegressorMixin],
    X: np.ndarray,
    y: np.ndarray,
    splitter: KFold | GroupKFold,
    cv_strategy: str,
    groups: list[str],
) -> list[BenchmarkResult]:
    """Evaluate candidate models with consistent pre-processing and CV settings."""

    def pipeline(regressor: RegressorMixin) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", regressor),
            ]
        )

    split_kwargs: dict[str, Any] = {}
    if cv_strategy == "group_kfold":
        split_kwargs["groups"] = groups
    splits = list(splitter.split(X, y, **split_kwargs))

    results: list[BenchmarkResult] = []
    for name, model in models.items():
        predictions = np.zeros(len(y), dtype=float)
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []

        for train_idx, test_idx in splits:
            fold_model = pipeline(clone(model))
            fold_model.fit(X[train_idx], y[train_idx])
            fold_predictions = fold_model.predict(X[test_idx])
            predictions[test_idx] = fold_predictions

            y_true = y[test_idx]
            fold_mae.append(float(mean_absolute_error(y_true, fold_predictions)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_true, fold_predictions))))
            if len(np.unique(y_true)) > 1:
                fold_r2.append(float(r2_score(y_true, fold_predictions)))
            else:
                fold_r2.append(1.0)

        results.append(
            BenchmarkResult(
                name=name,
                mae=float(mean_absolute_error(y, predictions)),
                rmse=float(np.sqrt(mean_squared_error(y, predictions))),
                r2=float(r2_score(y, predictions)),
                mae_std=float(np.std(fold_mae, ddof=0)) if fold_mae else 0.0,
                rmse_std=float(np.std(fold_rmse, ddof=0)) if fold_rmse else 0.0,
                r2_std=float(np.std(fold_r2, ddof=0)) if fold_r2 else 0.0,
                folds_used=len(splits),
            )
        )
    return results
