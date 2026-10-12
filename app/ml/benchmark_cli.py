"""CLI helpers for spread-model benchmarking and dataset capture/seeding."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from random import Random
from statistics import mean, pstdev
from typing import Any

from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db
from app.ml.benchmarking import (
    REAL_WORLD_WILDFIRE_DATASETS,
    benchmark_algorithm_bakeoff,
    benchmark_real_world_wildfire_dataset,
    benchmark_real_world_wildfire_dataset_suite,
    benchmark_spread_models,
    benchmark_spread_models_time_split,
    build_historical_spread_samples,
    build_production_readiness_scorecard,
)
from app.models.database import DistanceHistory, Fire, Location
from app.services.fire_data import FireDataService


def run_benchmark_report(
    db: Session,
    *,
    min_history_points: int = 3,
    cv_splits: int = 5,
    random_state: int = 42,
    holdout_fraction: float = 0.25,
) -> dict[str, Any]:
    """Build spread samples and run the benchmark when enough labels exist."""
    samples = build_historical_spread_samples(db, min_history_points=min_history_points)
    spread_rates = [
        float(rate) for sample in samples if (rate := sample.get("spread_rate")) is not None
    ]
    group_ids = {
        "{}::{}".format(
            sample.get("metadata", {}).get("fire_id"),
            sample.get("metadata", {}).get("location_id"),
        )
        for sample in samples
    }

    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "group_count": len(group_ids),
        "min_history_points": min_history_points,
        "cv_splits": cv_splits,
        "random_state": random_state,
        "holdout_fraction": holdout_fraction,
        "target_min": min(spread_rates) if spread_rates else 0.0,
        "target_max": max(spread_rates) if spread_rates else 0.0,
        "target_mean": mean(spread_rates) if spread_rates else 0.0,
        "target_std": pstdev(spread_rates) if len(spread_rates) > 1 else 0.0,
        "target_unique_count": len({round(value, 6) for value in spread_rates}),
        "can_benchmark": len(samples) >= 6,
    }

    if not summary["can_benchmark"]:
        summary["reason"] = (
            "Need at least 6 labeled samples before cross-validated benchmarking can run"
        )
        return summary

    benchmark = benchmark_spread_models(
        samples,
        random_state=random_state,
        cv_splits=cv_splits,
    )
    summary.update(benchmark)
    try:
        summary["time_split"] = benchmark_spread_models_time_split(
            samples,
            random_state=random_state,
            holdout_fraction=holdout_fraction,
        )
    except ValueError as exc:
        summary["time_split"] = {
            "time_split_ran": False,
            "reason": str(exc),
            "holdout_fraction": holdout_fraction,
        }
    summary["scorecard"] = build_production_readiness_scorecard(summary)
    return summary


def _persist_processed_fires(
    db: Session,
    *,
    locations: list[Location],
    fires_data: list[dict[str, Any]],
    observed_at: datetime,
) -> dict[str, int]:
    """Persist processed fire snapshots as normalized fire and history rows."""
    location_by_public_id = {location.location_id: location for location in locations}
    touched_fire_ids: set[int] = set()
    history_rows = 0

    for fire_data in fires_data:
        fire_id = fire_data.get("fire_id")
        if not fire_id:
            continue

        db_fire = db.query(Fire).filter(Fire.fire_id == fire_id).first()
        if db_fire is None:
            db_fire = Fire(
                fire_id=str(fire_id),
                name=fire_data.get("name") or "Unknown",
                latitude=float(fire_data.get("latitude") or 0.0),
                longitude=float(fire_data.get("longitude") or 0.0),
                acres_burned=float(fire_data.get("acres_burned") or 0.0),
                percent_contained=float(fire_data.get("percent_contained") or 0.0),
                polygon_data=fire_data.get("polygon_data"),
                extra_metadata=fire_data.get("extra_metadata") or {},
            )
            db.add(db_fire)
            db.flush()
        else:
            db_fire.name = fire_data.get("name") or db_fire.name
            db_fire.latitude = float(fire_data.get("latitude") or db_fire.latitude)
            db_fire.longitude = float(fire_data.get("longitude") or db_fire.longitude)
            db_fire.acres_burned = float(
                fire_data.get("acres_burned") or db_fire.acres_burned or 0.0
            )
            db_fire.percent_contained = float(
                fire_data.get("percent_contained") or db_fire.percent_contained or 0.0
            )
            db_fire.polygon_data = fire_data.get("polygon_data") or db_fire.polygon_data
            db_fire.extra_metadata = {
                **(db_fire.extra_metadata or {}),
                **(fire_data.get("extra_metadata") or {}),
            }
            db_fire.last_updated = observed_at

        touched_fire_ids.add(db_fire.id)

        distances = fire_data.get("distances") or {}
        bearings = fire_data.get("bearings") or {}
        directions = fire_data.get("directions") or {}
        movement_data = fire_data.get("movement_data") or {}
        growth_24h = (fire_data.get("extra_metadata") or {}).get("growth_acres_24h")

        for location_id, distance in distances.items():
            location = location_by_public_id.get(location_id)
            if location is None or location.id is None:
                continue

            movement = movement_data.get(location_id) or {}
            history_entry = DistanceHistory(
                fire_id=db_fire.id,
                location_id=location.id,
                distance_miles=float(distance),
                bearing_degrees=float(bearings.get(location_id) or 0.0),
                direction=str(directions.get(location_id) or ""),
                timestamp=observed_at,
                moving_towards=bool(movement.get("moving_towards", False)),
                speed_mph=float(movement.get("speed", 0.0)),
                observed_acres_burned=float(fire_data.get("acres_burned") or 0.0),
                observed_percent_contained=float(fire_data.get("percent_contained") or 0.0),
                observed_growth_acres_24h=None if growth_24h is None else float(growth_24h),
            )
            db.add(history_entry)
            history_rows += 1

    db.commit()
    return {
        "fires_touched": len(touched_fire_ids),
        "history_rows": history_rows,
    }


async def capture_live_history(
    db: Session,
    *,
    iterations: int,
    interval_seconds: int,
) -> dict[str, Any]:
    """Capture live fire snapshots into distance history for benchmark training."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if interval_seconds < 0:
        raise ValueError("interval_seconds cannot be negative")

    locations = db.query(Location).filter(Location.is_active.is_(True)).all()
    if not locations:
        return {
            "iterations": iterations,
            "interval_seconds": interval_seconds,
            "captures_completed": 0,
            "fires_touched": 0,
            "history_rows_added": 0,
            "reason": "No active locations found",
        }

    fires_touched = 0
    history_rows_added = 0
    captures_completed = 0

    for iteration in range(iterations):
        fires_data = await FireDataService.process_fire_data(db, locations)
        observed_at = datetime.now(UTC)
        persisted = _persist_processed_fires(
            db,
            locations=locations,
            fires_data=fires_data,
            observed_at=observed_at,
        )
        fires_touched += persisted["fires_touched"]
        history_rows_added += persisted["history_rows"]
        captures_completed += 1

        if iteration < iterations - 1 and interval_seconds > 0:
            await asyncio.sleep(interval_seconds)

    return {
        "iterations": iterations,
        "interval_seconds": interval_seconds,
        "captures_completed": captures_completed,
        "fires_touched": fires_touched,
        "history_rows_added": history_rows_added,
    }


def seed_synthetic_history(
    db: Session,
    *,
    pair_limit: int,
    points_per_pair: int,
    step_minutes: int,
    random_seed: int,
    replace_existing: bool,
) -> dict[str, Any]:
    """Seed synthetic fire trajectories with varied spread labels for benchmarking."""
    if pair_limit < 1:
        raise ValueError("pair_limit must be at least 1")
    if points_per_pair < 3:
        raise ValueError("points_per_pair must be at least 3")
    if step_minutes < 1:
        raise ValueError("step_minutes must be at least 1")

    if replace_existing:
        db.query(DistanceHistory).delete()
        db.commit()

    locations = db.query(Location).filter(Location.is_active.is_(True)).all()
    fires = db.query(Fire).all()
    if not locations or not fires:
        return {
            "pair_limit": pair_limit,
            "points_per_pair": points_per_pair,
            "step_minutes": step_minutes,
            "pairs_seeded": 0,
            "history_rows_added": 0,
            "reason": "Need at least one active location and one fire before seeding",
        }

    rng = Random(random_seed)
    candidate_pairs: list[tuple[Fire, Location, float]] = []
    for fire in fires:
        for location in locations:
            distance = FireDataService.haversine_distance(
                fire.latitude,
                fire.longitude,
                location.latitude,
                location.longitude,
            )
            candidate_pairs.append((fire, location, distance))

    candidate_pairs.sort(key=lambda pair: pair[2])
    selected_pairs = candidate_pairs[:pair_limit]

    history_rows_added = 0
    now = datetime.now(UTC)
    step_hours = step_minutes / 60.0

    for pair_index, (fire, location, baseline_distance) in enumerate(selected_pairs):
        start_distance = max(0.5, baseline_distance + rng.uniform(3.0, 15.0))
        drift_mph = rng.uniform(0.2, 1.8)
        wave_amplitude = rng.uniform(0.1, 0.9)
        wave_period_hours = rng.uniform(2.0, 8.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        noise_amplitude = rng.uniform(0.02, 0.25)

        series_start = now - timedelta(minutes=(pair_index + 1) * points_per_pair * step_minutes)
        previous_distance: float | None = None

        for point_index in range(points_per_pair):
            elapsed_hours = point_index * step_hours
            trend_distance = max(0.2, start_distance - (drift_mph * elapsed_hours))
            wave_offset = wave_amplitude * math.sin(
                (elapsed_hours / wave_period_hours) * 2.0 * math.pi + phase
            )
            noise_offset = rng.uniform(-noise_amplitude, noise_amplitude)
            distance_miles = max(0.1, trend_distance + wave_offset + noise_offset)

            timestamp = series_start + timedelta(minutes=point_index * step_minutes)
            bearing = FireDataService.calculate_bearing(
                fire.latitude,
                fire.longitude,
                location.latitude,
                location.longitude,
            )
            direction = FireDataService.get_cardinal_direction(bearing)

            moving_towards = False
            speed_mph = 0.0
            if previous_distance is not None:
                moving_towards = distance_miles < previous_distance
                speed_mph = abs(previous_distance - distance_miles) / step_hours

            db.add(
                DistanceHistory(
                    fire_id=fire.id,
                    location_id=location.id,
                    distance_miles=round(distance_miles, 3),
                    bearing_degrees=round(bearing, 1),
                    direction=f"{direction} ({int(bearing)} deg)",
                    timestamp=timestamp,
                    moving_towards=moving_towards,
                    speed_mph=round(speed_mph, 3),
                    observed_acres_burned=float(fire.acres_burned or 0.0),
                    observed_percent_contained=float(fire.percent_contained or 0.0),
                    observed_growth_acres_24h=0.0,
                )
            )
            previous_distance = distance_miles
            history_rows_added += 1

    db.commit()

    return {
        "pair_limit": pair_limit,
        "points_per_pair": points_per_pair,
        "step_minutes": step_minutes,
        "pairs_seeded": len(selected_pairs),
        "history_rows_added": history_rows_added,
        "random_seed": random_seed,
        "replace_existing": replace_existing,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Firewatch benchmark and dataset tooling")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Run benchmark report against current history"
    )
    report_parser.add_argument("--min-history-points", type=int, default=3)
    report_parser.add_argument("--cv-splits", type=int, default=5)
    report_parser.add_argument("--random-state", type=int, default=42)
    report_parser.add_argument("--holdout-fraction", type=float, default=0.25)

    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture live snapshots into distance history",
    )
    capture_parser.add_argument("--iterations", type=int, default=6)
    capture_parser.add_argument("--interval-seconds", type=int, default=120)
    capture_parser.add_argument("--run-benchmark", action="store_true")
    capture_parser.add_argument("--min-history-points", type=int, default=3)
    capture_parser.add_argument("--cv-splits", type=int, default=5)
    capture_parser.add_argument("--random-state", type=int, default=42)
    capture_parser.add_argument("--holdout-fraction", type=float, default=0.25)

    seed_parser = subparsers.add_parser(
        "seed",
        help="Seed synthetic trajectories for benchmark validation",
    )
    seed_parser.add_argument("--pair-limit", type=int, default=20)
    seed_parser.add_argument("--points-per-pair", type=int, default=16)
    seed_parser.add_argument("--step-minutes", type=int, default=30)
    seed_parser.add_argument("--random-seed", type=int, default=42)
    seed_parser.add_argument("--replace-existing", action="store_true")
    seed_parser.add_argument("--run-benchmark", action="store_true")
    seed_parser.add_argument("--min-history-points", type=int, default=3)
    seed_parser.add_argument("--cv-splits", type=int, default=5)
    seed_parser.add_argument("--random-state", type=int, default=42)
    seed_parser.add_argument("--holdout-fraction", type=float, default=0.25)

    bakeoff_parser = subparsers.add_parser(
        "bakeoff",
        help="Run formal model bakeoff including optional XGBoost/LightGBM/CatBoost",
    )
    bakeoff_parser.add_argument("--min-history-points", type=int, default=3)
    bakeoff_parser.add_argument("--cv-splits", type=int, default=5)
    bakeoff_parser.add_argument("--random-state", type=int, default=42)
    bakeoff_parser.add_argument(
        "--allow-non-informative",
        action="store_true",
        help="Run bakeoff even when spread labels show no variation",
    )

    real_eval_parser = subparsers.add_parser(
        "real-eval",
        help="Benchmark against one or more real labeled wildfire datasets",
    )
    real_eval_parser.add_argument(
        "--dataset-name",
        choices=[*REAL_WORLD_WILDFIRE_DATASETS, "all"],
        default="uci_forest_fires",
        help="Dataset to evaluate. Use 'all' to run every supported dataset",
    )
    real_eval_parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset CSV path override (single dataset mode only)",
    )
    real_eval_parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download the real labeled dataset when dataset-path does not exist",
    )
    real_eval_parser.add_argument(
        "--include-zero-area",
        action="store_true",
        help="Keep rows with zero burned area labels (defaults to filtering them out)",
    )
    real_eval_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on loaded rows for quicker debugging",
    )
    real_eval_parser.add_argument("--cv-splits", type=int, default=5)
    real_eval_parser.add_argument("--random-state", type=int, default=42)
    real_eval_parser.add_argument(
        "--allow-non-informative",
        action="store_true",
        help="Allow bakeoff to run when labels are non-informative",
    )
    real_eval_parser.add_argument(
        "--run-tuning",
        action="store_true",
        help="Run compact model fine-tuning on top of the real labeled benchmark",
    )

    return parser


def _print_text(payload: dict[str, Any]) -> None:
    command = payload.get("command")
    print(f"Command: {command}")

    if command == "report":
        report = payload["report"]
        print(f"Labeled samples: {report['sample_count']} across {report['group_count']} tracks")
        print(
            "Target distribution: "
            f"min={report['target_min']:.4f}, max={report['target_max']:.4f}, "
            f"mean={report['target_mean']:.4f}, std={report['target_std']:.4f}, "
            f"unique={report['target_unique_count']}"
        )
        if report.get("can_benchmark"):
            print(
                f"Selected model: {report.get('best_model')} (baseline: "
                f"{report.get('baseline_model')})"
            )
            print(f"Improvement vs baseline MAE: {report.get('best_improvement_pct', 0.0):.2f}%")
            print(
                "MAE delta vs baseline: "
                f"{report.get('best_mae_delta_vs_baseline', 0.0):.6f} "
                f"(tie={report.get('effective_tie_with_baseline')})"
            )
            if report.get("uplift_diagnosis"):
                print(f"Uplift diagnosis: {report['uplift_diagnosis']}")
            if report.get("validation_warning"):
                print(f"Validation warning: {report['validation_warning']}")
            time_split = (
                report.get("time_split") if isinstance(report.get("time_split"), dict) else None
            )
            if time_split and time_split.get("time_split_ran"):
                print(
                    "Time-split holdout: "
                    f"train={time_split.get('train_sample_count')} "
                    f"test={time_split.get('test_sample_count')} "
                    f"best={time_split.get('best_model')} "
                    f"improvement={time_split.get('best_improvement_pct', 0.0):.2f}%"
                )
                print(
                    "Time-split MAE delta vs baseline: "
                    f"{time_split.get('best_mae_delta_vs_baseline', 0.0):.6f} "
                    f"(tie={time_split.get('effective_tie_with_baseline')})"
                )
                if time_split.get("drift_diagnosis"):
                    print(f"Time-split drift diagnosis: {time_split['drift_diagnosis']}")
            elif time_split and time_split.get("reason"):
                print(f"Time-split skipped: {time_split['reason']}")
            scorecard = (
                report.get("scorecard") if isinstance(report.get("scorecard"), dict) else None
            )
            if scorecard:
                print(
                    "Scorecard: "
                    f"grade={scorecard.get('grade')} "
                    f"production_ready={scorecard.get('production_ready')} "
                    f"state_of_the_art={scorecard.get('state_of_the_art')}"
                )
                print(
                    "Is it state-of-the-art / cutting edge? "
                    f"{'Yes' if scorecard.get('state_of_the_art') else 'No'}"
                )
                if scorecard.get("assessment"):
                    print(f"Assessment: {scorecard['assessment']}")
        else:
            print(f"Benchmark blocked: {report.get('reason')}")
        return

    if command in {"capture", "seed"}:
        operation = payload[command]
        print(json.dumps(operation, indent=2, sort_keys=True))
        if payload.get("report"):
            report = payload["report"]
            print(
                f"Post-run benchmark samples={report['sample_count']} "
                f"unique_targets={report['target_unique_count']} "
                f"can_benchmark={report['can_benchmark']}"
            )

    if command == "bakeoff":
        bakeoff = payload["bakeoff"]
        print(
            f"Bakeoff ran: {bakeoff.get('bakeoff_ran')} | samples={bakeoff.get('sample_count')} "
            f"unique_targets={bakeoff.get('target_unique_count')}"
        )
        if bakeoff.get("bakeoff_ran"):
            print(
                f"Winner: {bakeoff.get('winner_model')} vs baseline {bakeoff.get('baseline_model')}"
            )
            print(
                "Winner improvement vs baseline MAE: "
                f"{bakeoff.get('winner_improvement_pct_vs_baseline', 0.0):.2f}%"
            )
            print(
                "Winner MAE delta vs baseline: "
                f"{bakeoff.get('winner_mae_delta_vs_baseline', 0.0):.6f} "
                f"(tie={bakeoff.get('effective_tie_with_baseline')})"
            )
            if bakeoff.get("uplift_diagnosis"):
                print(f"Uplift diagnosis: {bakeoff['uplift_diagnosis']}")
            unavailable = bakeoff.get("unavailable_optional_models") or []
            if unavailable:
                print(f"Unavailable optional models: {', '.join(unavailable)}")
        else:
            print(f"Bakeoff skipped: {bakeoff.get('reason')}")

    if command == "real-eval":
        real_eval = payload["real_eval"]

        if real_eval.get("dataset_name") == "suite":
            print(
                f"Real dataset suite count={real_eval.get('dataset_count')} "
                f"total_samples={real_eval.get('total_samples_used')} "
                f"best_dataset={real_eval.get('best_dataset_by_spread_uplift')}"
            )
            dataset_results = real_eval.get("dataset_results") or {}
            for dataset_name, dataset_payload in dataset_results.items():
                spread = dataset_payload.get("spread_benchmark") or {}
                bakeoff = dataset_payload.get("bakeoff") or {}
                dataset_scorecard = (
                    dataset_payload.get("scorecard")
                    if isinstance(dataset_payload.get("scorecard"), dict)
                    else None
                )
                print(
                    f"[{dataset_name}] rows={dataset_payload.get('dataset_rows_loaded')} "
                    f"samples={dataset_payload.get('samples_used')} "
                    f"spread_best={spread.get('best_model')} "
                    f"spread_improvement={spread.get('best_improvement_pct', 0.0):.2f}% "
                    f"bakeoff_winner={bakeoff.get('winner_model')}"
                    f" grade={dataset_scorecard.get('grade') if dataset_scorecard else 'unknown'}"
                )
            scorecard = (
                real_eval.get("scorecard") if isinstance(real_eval.get("scorecard"), dict) else None
            )
            if scorecard:
                print(
                    "Scorecard: "
                    f"grade={scorecard.get('grade')} "
                    f"production_ready={scorecard.get('production_ready')} "
                    f"state_of_the_art={scorecard.get('state_of_the_art')}"
                )
                print(
                    "Is it state-of-the-art / cutting edge? "
                    f"{'Yes' if scorecard.get('state_of_the_art') else 'No'}"
                )
                if scorecard.get("assessment"):
                    print(f"Assessment: {scorecard['assessment']}")
            return

        spread = real_eval.get("spread_benchmark") or {}
        bakeoff = real_eval.get("bakeoff") or {}
        print(
            f"Real dataset name={real_eval.get('dataset_name')} "
            f"rows={real_eval.get('dataset_rows_loaded')} "
            f"samples={real_eval.get('samples_used')} "
            f"include_zero_area={real_eval.get('include_zero_area')}"
        )
        print(
            f"Spread benchmark winner={spread.get('best_model')} vs "
            f"baseline={spread.get('baseline_model')} "
            f"improvement={spread.get('best_improvement_pct', 0.0):.2f}%"
        )
        print(
            f"Spread MAE delta={spread.get('best_mae_delta_vs_baseline', 0.0):.6f} "
            f"(tie={spread.get('effective_tie_with_baseline')})"
        )
        if spread.get("uplift_diagnosis"):
            print(f"Spread uplift diagnosis: {spread['uplift_diagnosis']}")

        print(
            f"Algorithm bakeoff ran={bakeoff.get('bakeoff_ran')} "
            f"winner={bakeoff.get('winner_model')} "
            f"improvement={bakeoff.get('winner_improvement_pct_vs_baseline', 0.0):.2f}%"
        )
        if bakeoff.get("uplift_diagnosis"):
            print(f"Bakeoff uplift diagnosis: {bakeoff['uplift_diagnosis']}")

        tuning = real_eval.get("tuning")
        if tuning:
            print(
                f"Tuning winner={tuning.get('tuned_winner_model')} "
                f"improvement_vs_default_baseline="
                f"{tuning.get('tuned_improvement_pct_vs_default_baseline', 0.0):.2f}%"
            )

        scorecard = (
            real_eval.get("scorecard") if isinstance(real_eval.get("scorecard"), dict) else None
        )
        if scorecard:
            print(
                "Scorecard: "
                f"grade={scorecard.get('grade')} "
                f"production_ready={scorecard.get('production_ready')} "
                f"state_of_the_art={scorecard.get('state_of_the_art')}"
            )
            print(
                "Is it state-of-the-art / cutting edge? "
                f"{'Yes' if scorecard.get('state_of_the_art') else 'No'}"
            )
            if scorecard.get("assessment"):
                print(f"Assessment: {scorecard['assessment']}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    holdout_fraction = float(getattr(args, "holdout_fraction", 0.25))

    init_db()
    db = SessionLocal()
    try:
        payload: dict[str, Any] = {
            "command": args.command,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if args.command == "report":
            payload["report"] = run_benchmark_report(
                db,
                min_history_points=args.min_history_points,
                cv_splits=args.cv_splits,
                random_state=args.random_state,
                holdout_fraction=holdout_fraction,
            )
        elif args.command == "capture":
            payload["capture"] = asyncio.run(
                capture_live_history(
                    db,
                    iterations=args.iterations,
                    interval_seconds=args.interval_seconds,
                )
            )
            if args.run_benchmark:
                payload["report"] = run_benchmark_report(
                    db,
                    min_history_points=args.min_history_points,
                    cv_splits=args.cv_splits,
                    random_state=args.random_state,
                    holdout_fraction=holdout_fraction,
                )
        elif args.command == "seed":
            payload["seed"] = seed_synthetic_history(
                db,
                pair_limit=args.pair_limit,
                points_per_pair=args.points_per_pair,
                step_minutes=args.step_minutes,
                random_seed=args.random_seed,
                replace_existing=args.replace_existing,
            )
            if args.run_benchmark:
                payload["report"] = run_benchmark_report(
                    db,
                    min_history_points=args.min_history_points,
                    cv_splits=args.cv_splits,
                    random_state=args.random_state,
                    holdout_fraction=holdout_fraction,
                )
        elif args.command == "bakeoff":
            samples = build_historical_spread_samples(
                db, min_history_points=args.min_history_points
            )
            payload["bakeoff"] = benchmark_algorithm_bakeoff(
                samples,
                cv_splits=args.cv_splits,
                random_state=args.random_state,
                require_informative=not args.allow_non_informative,
            )
        elif args.command == "real-eval":
            if args.dataset_name == "all":
                if args.dataset_path:
                    raise ValueError("--dataset-path cannot be used when --dataset-name is all")
                payload["real_eval"] = benchmark_real_world_wildfire_dataset_suite(
                    list(REAL_WORLD_WILDFIRE_DATASETS),
                    download_if_missing=args.download_if_missing,
                    include_zero_area=args.include_zero_area,
                    cv_splits=args.cv_splits,
                    random_state=args.random_state,
                    require_informative=not args.allow_non_informative,
                    max_rows=args.max_rows,
                    run_tuning=args.run_tuning,
                )
            else:
                payload["real_eval"] = benchmark_real_world_wildfire_dataset(
                    args.dataset_path,
                    dataset_name=args.dataset_name,
                    download_if_missing=args.download_if_missing,
                    include_zero_area=args.include_zero_area,
                    cv_splits=args.cv_splits,
                    random_state=args.random_state,
                    require_informative=not args.allow_non_informative,
                    max_rows=args.max_rows,
                    run_tuning=args.run_tuning,
                )
                spread = payload["real_eval"].get("spread_benchmark")
                if isinstance(spread, dict):
                    payload["real_eval"]["scorecard"] = build_production_readiness_scorecard(
                        {
                            "can_benchmark": True,
                            **spread,
                        }
                    )

        if args.output == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_text(payload)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
