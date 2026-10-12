"""Combines the spread, risk and anomaly models with weather and external feeds per location."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.anomaly_detection import AnomalyDetector
from app.ml.benchmarking import (
    benchmark_spread_models,
    benchmark_spread_models_time_split,
    build_historical_spread_samples,
    build_production_readiness_scorecard,
    evaluate_model_promotion_gate,
    fine_tune_spread_models,
)
from app.ml.fire_prediction import FireSpreadPredictor
from app.ml.fire_weather import danger_level, fosberg_index
from app.ml.risk_scoring import RiskScorer
from app.ml.spread_model_registry import BASELINE_MODEL_NAME
from app.models.database import DistanceHistory, Fire, Location
from app.services.advanced_weather import AdvancedWeatherService
from app.services.external_intelligence import ExternalIntelligenceService
from app.services.fire_data import FireDataService

logger = logging.getLogger(__name__)
settings = get_settings()


def distance_history(
    db: Session, fire: Fire, location: Location | None = None
) -> list[DistanceHistory]:
    """A fire's distance history, oldest first, optionally for one location.

    A failed query is logged and treated as no history.
    """
    try:
        query = db.query(DistanceHistory).filter(DistanceHistory.fire_id == fire.id)
        if location is not None and getattr(location, "id", None) is not None:
            query = query.filter(DistanceHistory.location_id == location.id)
        return list(query.order_by(DistanceHistory.timestamp.asc()).all())
    except Exception as exc:
        logger.warning(
            "Failed to load distance history for %s: %s", getattr(fire, "fire_id", "?"), exc
        )
        return []


class MLFireAnalysisService:
    """Runs the analyses behind the /predictions endpoints."""

    def __init__(self) -> None:
        """Initialize ML analysis service with models."""
        self.predictor = FireSpreadPredictor()
        self.risk_scorer = RiskScorer()
        self.anomaly_detector = AnomalyDetector()
        self.weather_service = AdvancedWeatherService()
        self.external_intelligence_service = ExternalIntelligenceService()
        self._historical_model_signature: tuple[int, str | None] | None = None
        self._promoted_model_path = settings.SPREAD_MODEL_PROMOTED_PATH
        self._historical_model_summary: dict[str, object] = {
            "benchmark_ran": False,
            "selected_model": BASELINE_MODEL_NAME,
            "promoted": False,
            "training_samples": 0,
        }
        try:
            loaded = self.predictor.load_model_artifact(self._promoted_model_path)
            if loaded:
                self._historical_model_summary = {
                    "benchmark_ran": False,
                    "selected_model": self.predictor.spread_model_name,
                    "promoted": True,
                    "training_samples": 0,
                    "reason": "Loaded promoted model artifact at startup",
                }
        except Exception as exc:
            logger.warning("Failed to load promoted model artifact at startup: %s", exc)

    def _ensure_historical_spread_model(self, db: Session) -> None:
        """Benchmark and train the predictor on stored outcomes, once there are enough."""
        try:
            history_count, latest_timestamp = db.query(
                func.count(DistanceHistory.id),
                func.max(DistanceHistory.timestamp),
            ).one()
        except Exception as exc:
            logger.debug(
                "Skipping historical spread-model refresh; DB signature unavailable: %s", exc
            )
            return
        signature = (
            int(history_count or 0),
            latest_timestamp.isoformat() if latest_timestamp else None,
        )
        if signature == self._historical_model_signature:
            return

        try:
            samples = build_historical_spread_samples(db)
        except Exception as exc:
            logger.warning("Failed to derive historical spread samples: %s", exc)
            self._historical_model_signature = signature
            self._historical_model_summary = {
                "benchmark_ran": False,
                "selected_model": BASELINE_MODEL_NAME,
                "promoted": False,
                "training_samples": 0,
                "reason": "Historical sample derivation failed",
            }
            self.predictor.model_selection_summary = self._historical_model_summary
            return
        summary: dict[str, object] = {
            "benchmark_ran": False,
            "selected_model": BASELINE_MODEL_NAME,
            "promoted": False,
            "training_samples": len(samples),
        }

        if len(samples) >= settings.HISTORICAL_BENCHMARK_MIN_SAMPLES:
            benchmark = benchmark_spread_models(samples)
            try:
                benchmark["time_split"] = benchmark_spread_models_time_split(
                    samples,
                    holdout_fraction=settings.HISTORICAL_BENCHMARK_TIME_SPLIT_HOLDOUT_FRACTION,
                )
            except ValueError as exc:
                benchmark["time_split"] = {
                    "time_split_ran": False,
                    "reason": str(exc),
                    "holdout_fraction": settings.HISTORICAL_BENCHMARK_TIME_SPLIT_HOLDOUT_FRACTION,
                }

            if settings.HISTORICAL_BENCHMARK_RUN_TUNING:
                benchmark["tuning"] = fine_tune_spread_models(samples)

            benchmark["scorecard"] = build_production_readiness_scorecard(benchmark)

            promotion_gate = evaluate_model_promotion_gate(
                benchmark,
                min_improvement_pct=settings.SPREAD_MODEL_PROMOTION_MIN_IMPROVEMENT_PCT,
                require_informative=settings.SPREAD_MODEL_PROMOTION_REQUIRE_INFORMATIVE,
                require_non_tie_uplift=settings.SPREAD_MODEL_PROMOTION_REQUIRE_NON_TIE_UPLIFT,
                require_time_split=settings.SPREAD_MODEL_PROMOTION_REQUIRE_TIME_SPLIT,
                require_production_ready=settings.SPREAD_MODEL_PROMOTION_REQUIRE_PRODUCTION_READY,
                min_time_split_improvement_pct=(
                    settings.SPREAD_MODEL_PROMOTION_MIN_TIME_SPLIT_IMPROVEMENT_PCT
                ),
                max_time_split_degradation_pct=(
                    settings.SPREAD_MODEL_PROMOTION_MAX_TIME_SPLIT_DEGRADATION_PCT
                ),
                min_group_count=settings.SPREAD_MODEL_PROMOTION_MIN_GROUPS,
                min_r2=settings.SPREAD_MODEL_PROMOTION_MIN_R2,
                max_mae_std_ratio=settings.SPREAD_MODEL_PROMOTION_MAX_MAE_STD_RATIO,
                min_stability_score=settings.SPREAD_MODEL_PROMOTION_MIN_STABILITY_SCORE,
            )
            promotion_gate = self._apply_bootstrap_promotion_override(promotion_gate)

            selected_model = benchmark["baseline_model"]
            promoted = False
            selected_model_source = "baseline_fallback"

            if promotion_gate.get("promote"):
                selected_model, selected_model_source = self._select_serving_model(benchmark)
                promoted = True

            if promoted:
                self.predictor.train(samples, model_name=selected_model)
                try:
                    self.predictor.model_selection_summary = {
                        **benchmark,
                        "benchmark_ran": True,
                        "selected_model": selected_model,
                        "promoted": True,
                        "training_samples": len(samples),
                        "promotion_gate": promotion_gate,
                    }
                    self.predictor.save_model_artifact(
                        self._promoted_model_path,
                        metadata={
                            "promotion_gate": promotion_gate,
                            "benchmark": benchmark,
                            "training_samples": len(samples),
                        },
                    )
                except Exception as exc:
                    promoted = False
                    selected_model_source = "promotion_artifact_save_failed"
                    logger.warning("Promotion approved but artifact save failed: %s", exc)
            else:
                loaded_existing = False
                try:
                    loaded_existing = self.predictor.load_model_artifact(self._promoted_model_path)
                except Exception as exc:
                    logger.warning("Failed to load existing promoted model artifact: %s", exc)

                if loaded_existing:
                    selected_model = self.predictor.spread_model_name
                    selected_model_source = "existing_promoted_artifact"
                else:
                    self.predictor.train(samples, model_name=selected_model)
                    selected_model_source = "benchmark_baseline_unpromoted"

            summary = {
                **benchmark,
                "benchmark_ran": True,
                "selected_model": selected_model,
                "promoted": promoted,
                "selected_model_source": selected_model_source,
                "promotion_gate": promotion_gate,
                "training_samples": len(samples),
            }
            self.predictor.model_selection_summary = summary
            logger.info(
                "Historical spread-model benchmark selected %s using %s samples (promoted=%s)",
                selected_model,
                len(samples),
                promoted,
            )
        else:
            summary["reason"] = (
                f"Need at least {settings.HISTORICAL_BENCHMARK_MIN_SAMPLES} historical samples "
                "to benchmark spread models"
            )
            self.predictor.model_selection_summary = summary

        self._historical_model_signature = signature
        self._historical_model_summary = summary

    def _apply_bootstrap_promotion_override(self, promotion_gate: dict[str, Any]) -> dict[str, Any]:
        """Allow the first promotion when the strict gate misses only quality-margin checks."""
        if promotion_gate.get("promote"):
            return promotion_gate

        if not settings.SPREAD_MODEL_BOOTSTRAP_PROMOTION_ENABLED:
            return promotion_gate

        if Path(self._promoted_model_path).exists():
            return {
                **promotion_gate,
                "bootstrap_override_applied": False,
                "bootstrap_reason": (
                    "Bootstrap promotion is only allowed before the first artifact exists"
                ),
            }

        checks = {
            check.get("name"): check
            for check in (promotion_gate.get("checks") or [])
            if isinstance(check, dict)
        }
        required_for_bootstrap = [
            "dataset_informative",
            "non_tie_uplift",
            "minimum_group_count",
            "maximum_mae_std_ratio",
            "minimum_stability_score",
            "production_ready_scorecard",
        ]

        failures: list[str] = []
        for name in required_for_bootstrap:
            check = checks.get(name)
            if not check or not check.get("passed"):
                failures.append(
                    str(check.get("failure_reason") if check else f"Missing gate check: {name}")
                )

        if failures:
            return {
                **promotion_gate,
                "bootstrap_override_applied": False,
                "bootstrap_reason": "; ".join(failures),
            }

        return {
            **promotion_gate,
            "promote": True,
            "reason": "Promotion approved (bootstrap override for first artifact)",
            "bootstrap_override_applied": True,
            "bootstrap_reason": (
                "Strict gate missed quality-margin checks, but bootstrap override allowed a first "
                "artifact "
                "because dataset informativeness, group count, and stability checks passed."
            ),
        }

    @staticmethod
    def _select_serving_model(benchmark: dict[str, Any]) -> tuple[str, str]:
        """Pick the serving model, preferring the runtime ensemble when it is near-best."""
        best_model = str(benchmark.get("best_model") or BASELINE_MODEL_NAME)
        if not settings.SPREAD_MODEL_RUNTIME_ENSEMBLE_ENABLED:
            return best_model, "newly_promoted"

        results = benchmark.get("results") or []
        by_name = {
            row.get("name"): row
            for row in results
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }

        ensemble = by_name.get("voting_ensemble")
        best = by_name.get(best_model)
        if not ensemble or not best:
            return best_model, "newly_promoted"

        best_mae = float(best.get("mae") or 0.0)
        ensemble_mae = float(ensemble.get("mae") or 0.0)
        if best_mae <= 0:
            return best_model, "newly_promoted"

        max_delta_ratio = float(settings.SPREAD_MODEL_RUNTIME_ENSEMBLE_MAX_MAE_DELTA_PCT) / 100.0
        if ensemble_mae <= best_mae * (1.0 + max_delta_ratio):
            return "voting_ensemble", "newly_promoted_runtime_ensemble"

        return best_model, "newly_promoted"

    @staticmethod
    def _calculate_distance_miles(fire: Fire, location: Location) -> float:
        """Calculate current fire-to-location distance when coordinates are available."""
        coords = location.coordinates
        if coords is None:
            return 0.0
        return FireDataService.haversine_distance(fire.latitude, fire.longitude, *coords)

    @staticmethod
    def _build_effective_fire(
        fire: Fire, external_intelligence: dict[str, object] | None
    ) -> SimpleNamespace:
        """Copy perimeter-backed incident context onto the fire without mutating the ORM row."""
        fire_context = {
            "name": getattr(fire, "name", None),
            "acres_burned": getattr(fire, "acres_burned", None),
            "percent_contained": getattr(fire, "percent_contained", None),
            "discovered_at": getattr(fire, "discovered_at", None),
            "last_updated": getattr(fire, "last_updated", None),
            "bearing_degrees": getattr(fire, "bearing_degrees", None),
        }
        enriched = ExternalIntelligenceService.enrich_fire_context(
            fire_context, external_intelligence
        )
        return SimpleNamespace(
            fire_id=getattr(fire, "fire_id", None),
            name=enriched.get("name") or getattr(fire, "name", None),
            latitude=getattr(fire, "latitude", None),
            longitude=getattr(fire, "longitude", None),
            acres_burned=float(enriched.get("acres_burned") or 0.0),
            percent_contained=enriched.get("percent_contained"),
            discovered_at=enriched.get("discovered_at") or getattr(fire, "discovered_at", None),
            last_updated=enriched.get("last_updated") or getattr(fire, "last_updated", None),
            bearing_degrees=enriched.get("bearing_degrees")
            or getattr(fire, "bearing_degrees", None),
            growth_acres_24h=enriched.get("growth_acres_24h"),
            extra_metadata={
                **(getattr(fire, "extra_metadata", None) or {}),
                **(
                    {"growth_acres_24h": enriched.get("growth_acres_24h")}
                    if enriched.get("growth_acres_24h") is not None
                    else {}
                ),
            },
        )

    async def _collect_external_intelligence(
        self,
        fire: Fire,
        location: Location,
    ) -> dict[str, object]:
        """Collect backend external intelligence around the incident and watch location."""
        coords = location.coordinates
        if coords is None:
            return {}
        return await self.external_intelligence_service.collect_for_fire_and_location(
            fire.latitude, fire.longitude, *coords
        )

    async def analyze_fire_threat(
        self,
        fire: Fire,
        location: Location,
        distance_miles: float,
        db: Session,
    ) -> dict:
        """
        Comprehensive fire threat analysis for a location.

        Combines:
        - Fire risk assessment
        - Location vulnerability
        - Weather impact
        - Movement prediction
        - Anomaly detection
        - Threat score and recommendations

        Args:
            fire: Fire object
            location: Location object
            distance_miles: Distance from location to fire
            db: Database session
        """
        coords = location.coordinates
        if coords is None:
            raise ValueError(f"location {location.location_id} is not geocoded")
        latitude, longitude = coords
        self._ensure_historical_spread_model(db)
        history = distance_history(db, fire, location)

        weather, fwi_data, external_intelligence = await asyncio.gather(
            self.weather_service.get_current_weather(
                latitude,
                longitude,
            ),
            self.weather_service.get_fire_weather_indices(
                latitude,
                longitude,
            ),
            self._collect_external_intelligence(fire, location),
        )
        effective_fire = self._build_effective_fire(fire, external_intelligence)
        intelligence_adjustments = self.external_intelligence_service.summarize_threat_modifiers(
            external_intelligence,
        )

        # Calculate fire risk
        fire_risk = self.risk_scorer.calculate_fire_risk_score(effective_fire)

        # Calculate location vulnerability
        location_vuln = self.risk_scorer.calculate_location_vulnerability_score(
            location,
            effective_fire,
            distance_miles,
        )

        # Predict spread rate
        spread_rate = self.predictor.predict_spread_rate(effective_fire, history)

        # Calculate combined threat
        threat_score = self.risk_scorer.calculate_combined_threat_score(
            location,
            effective_fire,
            distance_miles,
            spread_rate,
        )
        threat_score = min(
            100.0,
            float(threat_score) + float(intelligence_adjustments.get("score_boost") or 0.0),
        )

        # Predict impact timeline
        impact_timeline = self.predictor.predict_impact_timeline(
            effective_fire,
            history,
            target_distance=0.5,  # Predict when fire reaches 0.5 miles
        )

        # Detect anomalies
        anomaly_result = self.anomaly_detector.detect_anomaly(effective_fire, history)

        # Get recommendations
        threat_description = self.risk_scorer.get_threat_description(threat_score)
        recommendations = self.risk_scorer.get_recommendations(threat_score)
        for recommendation in intelligence_adjustments.get("recommendations") or []:
            if recommendation not in recommendations:
                recommendations.insert(0, recommendation)

        # Calculate weather impact multiplier
        weather_multiplier = 1.0
        if weather:
            weather_multiplier = self.weather_service.calculate_fire_spread_multiplier(
                weather.get("temperature_f", 70),
                weather.get("humidity_pct", 50),
                weather.get("wind_speed_mph", 0),
            )

        # Adjusted spread rate with weather
        adjusted_spread_rate = spread_rate * weather_multiplier

        return {
            "threat_score": float(threat_score),
            "threat_level": threat_description,
            "fire_risk": float(fire_risk),
            "location_vulnerability": float(location_vuln),
            "current_distance_miles": float(distance_miles),
            "model_provenance": self.predictor.get_model_provenance(),
            "predictions": {
                "spread_model": self.predictor.spread_model_name,
                "spread_rate_mph": float(spread_rate),
                "adjusted_spread_rate_mph": float(adjusted_spread_rate),
                "external_risk_score_boost": float(
                    intelligence_adjustments.get("score_boost") or 0.0
                ),
                "impact_timeline": impact_timeline,
                "containment_forecast": self.predictor.predict_containment_time(
                    effective_fire,
                    history,
                ),
                "direction_probabilities": self.predictor.predict_direction_probabilities(
                    effective_fire,
                    history,
                ),
            },
            "weather": {
                "current": weather,
                "fire_weather_index": fwi_data.get("fwi_index") if fwi_data else None,
                "fire_danger_level": fwi_data.get("danger_level") if fwi_data else None,
                "spread_multiplier": float(weather_multiplier),
            },
            "anomalies": {
                "is_anomalous": anomaly_result.get("is_anomaly"),
                "reasons": anomaly_result.get("reasons", []),
                "insight": self.anomaly_detector.get_anomaly_insight(effective_fire, history),
            },
            "model_selection": self._historical_model_summary,
            "external_intelligence": external_intelligence,
            "external_risk_modifiers": intelligence_adjustments.get("modifiers") or [],
            "recommendations": recommendations,
        }

    async def get_evacuation_optimization(
        self,
        location: Location,
        fires: list[Fire],
        db: Session,
    ) -> dict:
        """
        Optimize evacuation routing considering multiple fires.

        Analyzes:
        - Threat from each fire
        - Safe route identification
        - Timing recommendations
        - Alternative destinations
        """
        fire_threats = []
        distance_estimates = []

        for fire in fires:
            distance = self._calculate_distance_miles(fire, location)
            distance_estimates.append(distance)

            threat = await self.analyze_fire_threat(
                fire,
                location,
                distance,
                db,
            )
            fire_threats.append(
                {
                    "fire_id": fire.fire_id,
                    "threat_score": threat["threat_score"],
                    "spread_rate": threat["predictions"]["adjusted_spread_rate_mph"],
                }
            )

        # Sort by threat
        fire_threats.sort(key=lambda x: x["threat_score"], reverse=True)

        # Determine evacuation urgency
        max_threat = fire_threats[0]["threat_score"] if fire_threats else 0

        if max_threat < 30:
            urgency = "MONITOR"
            timing = "Monitor situation, be ready to leave"
        elif max_threat < 50:
            urgency = "PREPARE"
            timing = "Prepare to evacuate - could need to leave within hours"
        elif max_threat < 70:
            urgency = "WARNING"
            timing = "Be ready to leave immediately - minutes to hours"
        else:
            urgency = "EVACUATE"
            timing = "Leave now"

        distance_to_safety = None
        if distance_estimates:
            distance_to_safety = round(max(distance_estimates) + 10.0, 1)

        return {
            "urgency_level": urgency,
            "timing": timing,
            "primary_threat": fire_threats[0] if fire_threats else None,
            "threat_ranking": fire_threats,
            "suggested_direction": "Away from fire(s)",
            "distance_to_safety": distance_to_safety,
        }

    async def get_predictive_analytics(
        self,
        location: Location,
        forecast_hours: int = 24,
    ) -> dict:
        """
        Get predictive analytics for location over forecast period.

        Args:
            location: Location to forecast
            forecast_hours: Hours into future to predict
        """
        coords = location.coordinates
        if coords is None:
            raise ValueError(f"location {location.location_id} is not geocoded")
        forecast = await self.weather_service.get_forecast(
            *coords,
            days=(forecast_hours // 24) + 1,
        )

        if not forecast:
            return {"error": "Unable to retrieve forecast"}

        predictions = []

        for day in forecast.get("forecast", []):
            day_date = day.get("date")
            max_temp = day.get("temp_max_f", 70)
            humidity = day.get("humidity_pct", 50)
            wind_speed = day.get("wind_speed_mph", 0)

            fwi = fosberg_index(max_temp, humidity, wind_speed)

            # Estimate spread multiplier
            multiplier = self.weather_service.calculate_fire_spread_multiplier(
                max_temp,
                humidity,
                wind_speed,
            )

            predictions.append(
                {
                    "date": day_date,
                    "temp_high_f": max_temp,
                    "humidity_pct": humidity,
                    "wind_speed_mph": wind_speed,
                    "fire_weather_index": fwi,
                    "spread_multiplier": multiplier,
                    "fire_danger": danger_level(fwi),
                }
            )

        return {
            "location_id": location.location_id,
            "forecast_period_hours": forecast_hours,
            "predictions": predictions,
        }
