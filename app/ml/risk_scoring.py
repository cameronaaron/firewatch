"""Threat scores (0-100) from fire size, containment, age, distance and trajectory."""

import math

import numpy as np

from app.models.database import FireLike, Location


class RiskScorer:
    """Weighted, bounded scores; no trained model is involved."""

    def calculate_fire_risk_score(self, fire: FireLike) -> float:
        """
        Calculate fire risk score (0-100).

        Factors:
        - Acreage (larger = higher risk)
        - Containment (lower = higher risk)
        - Age (newer = higher risk, higher growth potential)

        Returns:
            Risk score 0-100
        """
        components = []

        # Logarithmic in size, reaching 100 at 10,000 acres: 100 acres is 50,
        # 1,000 is 75.
        acres = max(0.0, fire.acres_burned or 0.0)
        acreage_risk = min(100.0, 100 * math.log10(1 + acres) / math.log10(1 + 10_000))
        components.append(acreage_risk * 0.35)

        # Containment risk (inverse - lower containment = higher risk)
        containment_risk = 100 - (fire.percent_contained or 0)
        components.append(containment_risk * 0.40)

        # Recency risk (fires discovered today are more volatile)
        if fire.discovered_at:
            hours_old = (fire.last_updated - fire.discovered_at).total_seconds() / 3600
            # Recency factor decreases with time
            recency_factor = max(0, 1 - (hours_old / 72)) * 100  # Decay over 3 days
            components.append(recency_factor * 0.25)

        return float(min(100, sum(components)))

    def calculate_location_vulnerability_score(
        self,
        location: Location,
        fire: FireLike,
        distance_miles: float,
    ) -> float:
        """
        Calculate location vulnerability score (0-100).

        Factors:
        - Distance to fire (closer = higher vulnerability)
        - Fire containment level (uncontained = higher vulnerability)
        - Fire size (larger = higher impact area)

        Args:
            location: Location object
            fire: FireLike object
            distance_miles: Distance from location to fire

        Returns:
            Vulnerability score 0-100
        """
        components = []

        # Quadratic falloff from 100 at the fire to 0 at 50 miles.
        max_distance = 50
        distance_factor = max(0, 1 - (distance_miles / max_distance))
        distance_risk = (distance_factor**2) * 100
        components.append(distance_risk * 0.40)

        # Containment vulnerability
        containment_factor = fire.percent_contained or 0
        uncontained_risk = (100 - containment_factor) * 0.8
        components.append(uncontained_risk * 0.35)

        # Size vulnerability (acres -> impact radius)
        # Rough approximation: impact_radius ≈ sqrt(acres) / 10
        impact_radius = np.sqrt(fire.acres_burned) / 10
        size_distance_ratio = min(1.0, distance_miles / max(1, impact_radius))
        size_risk = (1 - size_distance_ratio) * 100
        components.append(size_risk * 0.25)

        return min(100, sum(components))

    def calculate_combined_threat_score(
        self,
        location: Location,
        fire: FireLike,
        distance_miles: float,
        spread_rate: float | None = None,
    ) -> float:
        """
        Calculate combined threat score integrating all factors.

        Formula:
        threat = (0.3 * fire_risk + 0.4 * location_vulnerability + 0.3 * trajectory_risk)

        Args:
            location: Location object
            fire: FireLike object
            distance_miles: Distance from location to fire
            spread_rate: Predicted fire spread rate (miles/hour)

        Returns:
            Combined threat score 0-100
        """
        fire_risk = self.calculate_fire_risk_score(fire)
        location_vuln = self.calculate_location_vulnerability_score(
            location,
            fire,
            distance_miles,
        )

        # Trajectory risk: is fire moving toward location?
        trajectory_risk = self._calculate_trajectory_risk(
            location,
            fire,
            distance_miles,
            spread_rate or 0,
        )

        # Weighted combination
        combined = 0.30 * fire_risk + 0.40 * location_vuln + 0.30 * trajectory_risk

        return min(100, combined)

    def _calculate_trajectory_risk(
        self,
        location: Location,
        fire: FireLike,
        distance_miles: float,
        spread_rate: float,
    ) -> float:
        """
        Calculate risk based on fire trajectory toward location.

        Args:
            location: Location object
            fire: FireLike object
            distance_miles: Distance to fire
            spread_rate: Fire spread rate in miles/hour

        Returns:
            Trajectory risk score 0-100
        """
        if not hasattr(fire, "latitude") or not hasattr(fire, "longitude"):
            return 0.0

        # Calculate angle from fire to location
        fire_to_location_bearing = self._calculate_bearing(
            fire.latitude,
            fire.longitude,
            location.latitude,
            location.longitude,
        )

        # Bearing deviation from fire's current bearing
        current_bearing = getattr(fire, "bearing_degrees", 0) or 0
        bearing_diff = abs(fire_to_location_bearing - current_bearing)
        bearing_diff = min(bearing_diff, 360 - bearing_diff)

        # Probability fire is moving toward location
        # Gaussian distribution centered at 0 difference
        direction_alignment = np.exp(-0.5 * (bearing_diff / 45) ** 2)

        # Time-based risk: how soon could fire reach?
        if spread_rate > 0:
            hours_to_reach = distance_miles / spread_rate
            # Risk decreases for distant future impacts
            time_factor = max(0, 1 - (hours_to_reach / 72)) if hours_to_reach > 0 else 1
        else:
            time_factor = 0.5

        trajectory_risk = (direction_alignment * time_factor) * 100

        return float(min(100, trajectory_risk))

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate compass bearing from point 1 to point 2.

        Uses math.atan2 for accurate bearing calculation.

        Args:
            lat1, lon1: Starting coordinates (degrees)
            lat2, lon2: Ending coordinates (degrees)

        Returns:
            Bearing in degrees (0-360, 0=North)
        """
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlon = lon2_rad - lon1_rad

        x = np.sin(dlon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

        bearing_rad = np.arctan2(x, y)
        bearing_deg = np.degrees(bearing_rad)

        return float((bearing_deg + 360) % 360)

    def get_threat_description(self, score: float) -> str:
        """
        Get human-readable threat description based on score.

        Args:
            score: Threat score 0-100
        """
        if score < 10:
            return "Minimal"
        elif score < 30:
            return "Low"
        elif score < 50:
            return "Moderate"
        elif score < 70:
            return "High"
        elif score < 85:
            return "Very High"
        else:
            return "Critical"

    def get_recommendations(self, score: float) -> list[str]:
        """
        Get action recommendations based on threat score.

        Args:
            score: Threat score 0-100
        """
        recommendations = []

        if score < 10:
            recommendations = [
                "Monitor fire updates",
                "Keep emergency kit ready",
            ]
        elif score < 30:
            recommendations = [
                "Prepare to evacuate if conditions worsen",
                "Monitor fire progress regularly",
                "Keep vehicle fueled and ready",
            ]
        elif score < 50:
            recommendations = [
                "Be ready to leave on short notice",
                "Prepare evacuation bag and documents",
                "Monitor emergency alerts closely",
            ]
        elif score < 70:
            recommendations = [
                "Prepare for immediate evacuation",
                "Keep evacuation routes clear",
                "Be in contact with family members",
                "Monitor emergency broadcasts",
            ]
        elif score < 85:
            recommendations = [
                "Evacuate or prepare for immediate evacuation",
                "Gather essential items",
                "Drive to safe location if not already",
            ]
        else:
            recommendations = [
                "EVACUATE IMMEDIATELY",
                "Do not delay - leave now",
                "Drive to designated safe zone",
                "Bring essential documents and medications",
            ]

        return recommendations
