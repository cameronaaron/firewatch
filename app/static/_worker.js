const EDGE_API_PREFIX = "/edge/v1";
const EDGE_API_ALIASES = [EDGE_API_PREFIX, "/api/v1"];
const DEFAULT_FIRMS_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"];
const ALERT_SEVERITY = {
  UNKNOWN: 0,
  SAFE: 1,
  MONITOR: 2,
  PREPARE: 3,
  WARNING: 4,
  EVACUATE: 5,
};
const EXTERNAL_FETCH_TIMEOUT_MS = 8000;
const EXTERNAL_FETCH_TIMEOUTS = {
  geocoding: 6000,
  aqi: 7000,
  eonet: 7000,
  firms: 9000,
  nws: 7000,
  perimeter: 7000,
  smoke: 7000,
  purpleair: 7000,
  closures: 7000,
  terrain: 6000,
  utility: 7000,
  lightning: 7000,
};

function shouldHandleEdgeApi(pathname) {
  return EDGE_API_ALIASES.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

function shouldHandleDocs(pathname) {
  return pathname === "/docs" || pathname === "/openapi.json" || pathname === "/redoc";
}

function edgePath(pathname) {
  const matchedPrefix = EDGE_API_ALIASES.find((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
  if (!matchedPrefix) {
    return pathname;
  }
  const trimmed = pathname.slice(matchedPrefix.length);
  return trimmed || "/";
}

function corsHeaders(origin = "*") {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
  };
}

function jsonResponse(payload, { status = 200, origin = "*" } = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(origin),
    },
  });
}

function noContentResponse(origin = "*") {
  return new Response(null, {
    status: 204,
    headers: corsHeaders(origin),
  });
}

function edgeOpenApiDocument(requestUrl) {
  const serverUrl = `${requestUrl.protocol}//${requestUrl.host}`;
  return {
    openapi: "3.0.3",
    info: {
      title: "Firewatch Edge API",
      version: "2.0.0",
      description: "Worker-native Firewatch endpoints running on Cloudflare Pages/Workers.",
    },
    servers: [{ url: serverUrl }],
    paths: {
      "/edge/v1/health": {
        get: {
          summary: "Health check",
          responses: { 200: { description: "Healthy edge worker" } },
        },
      },
      "/edge/v1/health/diagnostics": {
        get: {
          summary: "Structured dependency diagnostics for D1, EONET, FIRMS, and AQI",
          responses: {
            200: { description: "Diagnostics returned" },
          },
        },
      },
      "/edge/v1/locations/resolve": {
        post: {
          summary: "Resolve an address to coordinates",
          responses: {
            200: { description: "Address resolved" },
            400: { description: "Address missing" },
            422: { description: "Address not resolvable" },
          },
        },
      },
      "/edge/v1/locations": {
        get: {
          summary: "List active watched locations",
          responses: {
            200: { description: "Locations returned" },
          },
        },
        post: {
          summary: "Create or update a watched location",
          responses: {
            200: { description: "Location saved" },
            400: { description: "Invalid payload" },
          },
        },
      },
      "/edge/v1/locations/{location_id}": {
        get: {
          summary: "Get a watched location",
          responses: {
            200: { description: "Location found" },
            404: { description: "Location not found" },
          },
        },
        delete: {
          summary: "Delete (deactivate) a watched location",
          responses: {
            200: { description: "Location deleted" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/alerts/assessment/{location_id}": {
        get: {
          summary: "Get threat assessment for a watched location",
          responses: {
            200: { description: "Assessment returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/predictions/threat/{location_id}": {
        get: {
          summary: "Get spread and threat prediction for a watched location",
          responses: {
            200: { description: "Threat prediction returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/predictions/forecast/{location_id}": {
        get: {
          summary: "Get horizon forecast for spread and risk trend",
          responses: {
            200: { description: "Forecast returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/predictions/evacuation/{location_id}": {
        get: {
          summary: "Get evacuation guidance for a watched location",
          responses: {
            200: { description: "Evacuation guidance returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/predictions/weather-impact/{location_id}": {
        get: {
          summary: "Get weather and air-quality impact on fire spread",
          responses: {
            200: { description: "Weather impact returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/predictions/anomalies/{location_id}": {
        get: {
          summary: "Detect unusual spread behavior for a watched location",
          responses: {
            200: { description: "Anomaly analysis returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/intelligence/{location_id}": {
        get: {
          summary: "Get external intelligence from weather, smoke, incidents, closures, and utilities",
          responses: {
            200: { description: "External intelligence returned" },
            404: { description: "Location not found" },
          },
        },
      },
      "/edge/v1/audit/provenance/{location_id}": {
        get: {
          summary: "One-call audit for edge and backend model provenance plus feed reliability",
          responses: {
            200: { description: "Provenance audit returned" },
            404: { description: "Location not found" },
          },
        },
      },
    },
  };
}

function docsHtml() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Firewatch Edge API Docs</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; line-height: 1.5; color: #132128; }
    h1 { margin-bottom: 0.25rem; }
    code { background: #eef3f5; padding: 0.1rem 0.3rem; border-radius: 4px; }
    ul { padding-left: 1.2rem; }
    a { color: #005ea2; }
  </style>
</head>
<body>
  <h1>Firewatch Edge API</h1>
  <p>These endpoints are served directly from Cloudflare Worker code (no backend tunnel required).</p>
  <p><a href="/openapi.json">OpenAPI JSON</a></p>
  <ul>
    <li><code>GET /edge/v1/health</code></li>
    <li><code>GET /edge/v1/health/diagnostics</code></li>
    <li><code>POST /edge/v1/locations/resolve</code></li>
    <li><code>POST /edge/v1/locations</code></li>
    <li><code>GET /edge/v1/locations</code></li>
    <li><code>GET /edge/v1/locations/{location_id}</code></li>
    <li><code>DELETE /edge/v1/locations/{location_id}</code></li>
    <li><code>GET /edge/v1/alerts/assessment/{location_id}</code></li>
    <li><code>GET /edge/v1/predictions/threat/{location_id}</code></li>
    <li><code>GET /edge/v1/predictions/forecast/{location_id}?hours=24</code></li>
    <li><code>GET /edge/v1/predictions/evacuation/{location_id}</code></li>
    <li><code>GET /edge/v1/predictions/weather-impact/{location_id}</code></li>
    <li><code>GET /edge/v1/predictions/anomalies/{location_id}</code></li>
    <li><code>GET /edge/v1/intelligence/{location_id}</code></li>
    <li><code>GET /edge/v1/audit/provenance/{location_id}</code></li>
  </ul>
</body>
</html>`;
}

function docsResponse(origin = "*") {
  return new Response(docsHtml(), {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      ...corsHeaders(origin),
    },
  });
}

function handleDocsRequest(requestUrl, origin = "*") {
  if (requestUrl.pathname === "/redoc") {
    return Response.redirect(new URL("/docs", requestUrl).toString(), 302);
  }
  if (requestUrl.pathname === "/openapi.json") {
    return jsonResponse(edgeOpenApiDocument(requestUrl), { origin });
  }
  return docsResponse(origin);
}

async function parseJsonBody(request) {
  try {
    return await request.json();
  } catch {
    return null;
  }
}

function externalTimeoutFor(kind) {
  return EXTERNAL_FETCH_TIMEOUTS[kind] || EXTERNAL_FETCH_TIMEOUT_MS;
}

async function fetchExternal(url, {
  kind = "external",
  method = "GET",
  headers,
  body,
  timeoutMs,
  parseAs = "json",
} = {}) {
  const timeout = timeoutMs || externalTimeoutFor(kind);
  const controller = new AbortController();
  const startedAt = Date.now();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      method,
      headers,
      body,
      signal: controller.signal,
    });
    const durationMs = Date.now() - startedAt;
    if (!response.ok) {
      return {
        ok: false,
        reason: "http_error",
        status: response.status,
        duration_ms: durationMs,
      };
    }

    let data = null;
    if (parseAs === "json") {
      try {
        data = await response.json();
      } catch {
        return {
          ok: false,
          reason: "invalid_json",
          status: response.status,
          duration_ms: durationMs,
        };
      }
    } else if (parseAs === "text") {
      try {
        data = await response.text();
      } catch {
        return {
          ok: false,
          reason: "invalid_text",
          status: response.status,
          duration_ms: durationMs,
        };
      }
    }

    return {
      ok: true,
      data,
      status: response.status,
      duration_ms: durationMs,
    };
  } catch (error) {
    const durationMs = Date.now() - startedAt;
    return {
      ok: false,
      reason: error?.name === "AbortError" ? "timeout" : "network_error",
      status: null,
      duration_ms: durationMs,
      error: String(error?.message || error),
    };
  } finally {
    clearTimeout(timer);
  }
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeAddress(address) {
  return String(address || "")
    .trim()
    .replace(/\s*,\s*/g, ", ")
    .replace(/\s+/g, " ");
}

function isZipCode(address) {
  return /^\d{5}(?:-\d{4})?$/.test(address);
}

function deriveDisplayName(displayName, address) {
  const explicit = String(displayName || "").trim();
  if (explicit) {
    return explicit;
  }

  if (isZipCode(address)) {
    return `ZIP ${address.slice(0, 5)}`;
  }

  const firstSegment = normalizeAddress(address).split(",")[0]?.trim();
  return firstSegment || "Watched location";
}

function extractParam(pathname, pattern) {
  const match = pathname.match(pattern);
  if (!match) {
    return null;
  }
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return null;
  }
}

function haversineMiles(lat1, lon1, lat2, lon2) {
  const toRad = (value) => (value * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return 3958.8 * c;
}

function calculateBearing(lat1, lon1, lat2, lon2) {
  const toRad = (value) => (value * Math.PI) / 180;
  const toDeg = (value) => (value * 180) / Math.PI;
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const lambda = toRad(lon2 - lon1);
  const y = Math.sin(lambda) * Math.cos(phi2);
  const x = Math.cos(phi1) * Math.sin(phi2)
    - Math.sin(phi1) * Math.cos(phi2) * Math.cos(lambda);
  const theta = toDeg(Math.atan2(y, x));
  return (theta + 360) % 360;
}

function cardinalDirection(bearing) {
  const points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return points[Math.round(bearing / 45) % 8];
}

function maxDistanceMiles(env) {
  return Number(env.MAX_FIRE_DISTANCE_MILES || 300);
}

function fireAlertLevel(distanceMiles, acresBurned = 0, percentContained = 0) {
  if (distanceMiles == null) {
    return "UNKNOWN";
  }

  let adjusted = distanceMiles;
  if (acresBurned > 5000) {
    adjusted *= 0.5;
  } else if (acresBurned > 1000) {
    adjusted *= 0.7;
  }
  if (percentContained < 30) {
    adjusted *= 0.7;
  }
  if (percentContained < 10) {
    adjusted *= 0.5;
  }

  if (adjusted <= 1) {
    return "EVACUATE";
  }
  if (adjusted <= 2) {
    return "WARNING";
  }
  if (adjusted <= 5) {
    return "PREPARE";
  }
  if (adjusted <= 15) {
    return "MONITOR";
  }
  return "SAFE";
}

function airQualityAlertLevel(airQuality) {
  if (!airQuality || airQuality.air_quality_index == null) {
    return "SAFE";
  }
  const aqi = Number(airQuality.air_quality_index);
  if (aqi >= 151) {
    return "WARNING";
  }
  if (aqi >= 101) {
    return "PREPARE";
  }
  if (aqi >= 51) {
    return "MONITOR";
  }
  return "SAFE";
}

function combineAlertLevels(...levels) {
  return levels.reduce((best, current) => (
    (ALERT_SEVERITY[current] || 0) > (ALERT_SEVERITY[best] || 0) ? current : best
  ), "SAFE");
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function isValidCoordinatePair(latitude, longitude) {
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return false;
  }
  return latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180;
}

function threatScoreFromSignals({ distanceMiles, acresBurned, percentContained, airQualityIndex }) {
  const distanceScore = distanceMiles == null
    ? 0
    : clampNumber((1 - (Math.min(distanceMiles, 60) / 60)) * 65, 0, 65);
  const sizeScore = clampNumber(Math.log10((Number(acresBurned) || 0) + 1) * 6, 0, 16);
  const containmentScore = clampNumber((100 - (Number(percentContained) || 0)) * 0.14, 0, 14);
  const airScore = airQualityIndex == null
    ? 0
    : clampNumber((Number(airQualityIndex) - 50) * 0.12, 0, 12);
  return Math.round(clampNumber(distanceScore + sizeScore + containmentScore + airScore, 0, 100));
}

function riskLevelFromScore(score) {
  if (score >= 80) {
    return "CRITICAL";
  }
  if (score >= 60) {
    return "HIGH";
  }
  if (score >= 35) {
    return "ELEVATED";
  }
  return "LOW";
}

function estimateSpreadRateMph(nearestFire, airQuality) {
  if (!nearestFire) {
    return 0;
  }

  let spreadRate = 0.25;
  if (/wildfire/i.test(String(nearestFire.name || ""))) {
    spreadRate += 0.35;
  }

  const frp = toNumber(nearestFire.frp);
  if (frp != null) {
    spreadRate += clampNumber(frp / 25, 0, 1.4);
  }

  const confidence = toNumber(nearestFire.confidence);
  if (confidence != null) {
    spreadRate += confidence >= 85 ? 0.22 : 0.1;
  }

  const aqi = toNumber(airQuality?.air_quality_index);
  if (aqi != null && aqi >= 100) {
    spreadRate += 0.1;
  }

  return Number(clampNumber(spreadRate, 0.05, 3.5).toFixed(2));
}

function etaHours(distanceMiles, targetDistanceMiles, spreadRateMph) {
  if (distanceMiles == null || !Number.isFinite(spreadRateMph) || spreadRateMph <= 0) {
    return null;
  }
  if (distanceMiles <= targetDistanceMiles) {
    return 0;
  }
  return Number(((distanceMiles - targetDistanceMiles) / spreadRateMph).toFixed(1));
}

function buildForecastPoints(distanceMiles, spreadRateMph, horizonHours, baseThreatScore) {
  const checkpoints = [1, 3, 6, 12, 24, 48, 72, 96, 120, 168]
    .filter((hours) => hours <= horizonHours);
  const approachRate = spreadRateMph * 0.35;

  return checkpoints.map((hours) => {
    if (distanceMiles == null) {
      return {
        hour: hours,
        projected_distance_miles: null,
        projected_threat_score: clampNumber(baseThreatScore - Math.round(hours * 0.08), 0, 100),
      };
    }

    const projectedDistance = Math.max(0, distanceMiles - (approachRate * hours));
    const distanceGain = distanceMiles - projectedDistance;
    const projectedThreat = clampNumber(
      Math.round(baseThreatScore + (distanceGain * 2.4) - (hours * 0.06)),
      0,
      100,
    );

    return {
      hour: hours,
      projected_distance_miles: Number(projectedDistance.toFixed(1)),
      projected_threat_score: projectedThreat,
    };
  });
}

function deriveIncidentContext(nearestFire, externalIntelligence) {
  const perimeterIncident = externalIntelligence?.incident_perimeters?.nearest_incident || null;
  const perimeterAcres = toNumber(perimeterIncident?.acres);
  const perimeterContainment = toNumber(perimeterIncident?.percent_contained);
  const nearestAcres = toNumber(nearestFire?.acres_burned);
  const nearestContainment = toNumber(nearestFire?.percent_contained);

  const acresBurned = perimeterAcres ?? nearestAcres ?? 0;
  const containmentPct = perimeterContainment ?? nearestContainment ?? 0;

  return {
    acres_burned: Number(clampNumber(acresBurned, 0, 10000000).toFixed(1)),
    percent_contained: Number(clampNumber(containmentPct, 0, 100).toFixed(1)),
    acres_source: perimeterAcres != null ? "perimeter_feed" : nearestAcres != null ? "nearest_incident" : "default",
    containment_source: perimeterContainment != null ? "perimeter_feed" : nearestContainment != null ? "nearest_incident" : "default",
  };
}

function smokeDensityScore(density) {
  const normalized = String(density || "").trim().toUpperCase();
  if (normalized === "HEAVY") {
    return 1;
  }
  if (normalized === "MODERATE") {
    return 0.6;
  }
  if (normalized === "LIGHT") {
    return 0.3;
  }
  return 0;
}

function buildImpactAssessment({
  nearestFire,
  distanceMiles,
  direction,
  alertLevel,
  threatScore,
  etaTo5,
  etaTo1,
  spreadRateMph,
  weatherImpact,
  externalIntelligence,
  incidentContext,
}) {
  if (!nearestFire) {
    return {
      risk_to_watch_level: "LOW",
      likely_impact_within_24h: false,
      likely_impact_within_72h: false,
      impact_probability_24h_pct: 0,
      impact_probability_72h_pct: 0,
      impact_driver_breakdown: [],
      top_impact_drivers: [],
      containment_pct_used: null,
      acres_burned_used: null,
      containment_source: null,
      acres_source: null,
      distance_to_watch_miles: null,
      direction_to_watch: null,
      fire_location_from_watch: null,
      summary: "No active nearby wildfire detected within the configured monitoring radius.",
      note: "Threat score reflects risk to this watched address, not statewide fire severity.",
    };
  }

  const incident = incidentContext || deriveIncidentContext(nearestFire, externalIntelligence);
  const aqi = toNumber(weatherImpact?.air_quality_index);
  const purpleAirPm25 = toNumber(externalIntelligence?.purpleair?.nearest_sensor?.pm25_ug_m3);
  const representativePm25 = purpleAirPm25 ?? toNumber(weatherImpact?.pm25_ug_m3);
  const windMax = toNumber(externalIntelligence?.hrrr_wind_proxy?.max_wind_mph);
  const terrainSlope = toNumber(externalIntelligence?.terrain_context?.slope_degrees);
  const alertCount = toNumber(externalIntelligence?.nws_alerts?.alert_count);
  const hasRedFlag = Boolean(externalIntelligence?.nws_alerts?.has_red_flag_warning);
  const lightningRisk = Boolean(externalIntelligence?.lightning_nowcast?.elevated_ignition_risk);
  const smokeDensity = smokeDensityScore(externalIntelligence?.hms_smoke?.nearest_plume?.density);
  const weatherSpreadMultiplier = toNumber(weatherImpact?.spread_multiplier) ?? 1;

  const distanceNorm = distanceMiles == null ? 0 : clampNumber(1 - (Math.min(distanceMiles, 80) / 80), 0, 1);
  const eta24Norm = etaTo5 != null ? clampNumber(1 - (etaTo5 / 24), 0, 1) : clampNumber(distanceNorm * 0.45, 0, 1);
  const eta72Norm = etaTo5 != null ? clampNumber(1 - (etaTo5 / 72), 0, 1) : clampNumber(distanceNorm * 0.65, 0, 1);
  const threatNorm = clampNumber((Number(threatScore) || 0) / 100, 0, 1);
  const spreadNorm = clampNumber((toNumber(spreadRateMph) || 0) / 4, 0, 1);
  const containmentNorm = clampNumber((100 - (incident.percent_contained || 0)) / 100, 0, 1);
  const sizeNorm = clampNumber(Math.log10((incident.acres_burned || 0) + 1) / 5, 0, 1);
  const weatherNorm = clampNumber((weatherSpreadMultiplier - 1) / 0.6, 0, 1);
  const windNorm = clampNumber(((windMax ?? 8) - 8) / 32, 0, 1);
  const terrainNorm = clampNumber((terrainSlope ?? 0) / 35, 0, 1);
  const smokeNorm = clampNumber((representativePm25 != null ? (representativePm25 - 12) / 95 : 0) + smokeDensity, 0, 1);
  const alertsNorm = hasRedFlag ? 1 : clampNumber((alertCount ?? 0) / 6, 0, 1);
  const lightningNorm = lightningRisk ? 1 : 0;

  const driverInputs = [
    {
      key: "eta",
      label: "Time-to-impact",
      value24: eta24Norm,
      value72: eta72Norm,
      weight24: 0.25,
      weight72: 0.19,
      detail: etaTo5 != null ? `ETA to 5 miles is ${etaTo5.toFixed(1)}h` : "ETA estimated from distance and spread trend",
    },
    {
      key: "distance",
      label: "Distance to watch",
      value24: distanceNorm,
      value72: distanceNorm,
      weight24: 0.16,
      weight72: 0.14,
      detail: distanceMiles != null ? `${distanceMiles.toFixed(1)} miles from watch` : "Distance unavailable",
    },
    {
      key: "threat",
      label: "Threat score",
      value24: threatNorm,
      value72: threatNorm,
      weight24: 0.14,
      weight72: 0.12,
      detail: `${Math.round(threatScore)}/100 risk score`,
    },
    {
      key: "containment",
      label: "Containment",
      value24: containmentNorm,
      value72: containmentNorm,
      weight24: 0.13,
      weight72: 0.15,
      detail: `${incident.percent_contained.toFixed(1)}% contained (${incident.containment_source})`,
    },
    {
      key: "spread",
      label: "Spread speed",
      value24: spreadNorm,
      value72: spreadNorm,
      weight24: 0.09,
      weight72: 0.09,
      detail: `${(toNumber(spreadRateMph) || 0).toFixed(2)} mph estimated spread`,
    },
    {
      key: "size",
      label: "Fire size",
      value24: sizeNorm,
      value72: sizeNorm,
      weight24: 0.03,
      weight72: 0.09,
      detail: `${incident.acres_burned.toFixed(1)} acres (${incident.acres_source})`,
    },
    {
      key: "weather",
      label: "Weather spread multiplier",
      value24: weatherNorm,
      value72: weatherNorm,
      weight24: 0.06,
      weight72: 0.08,
      detail: `${weatherSpreadMultiplier.toFixed(2)}x weather-adjusted spread`,
    },
    {
      key: "wind",
      label: "Forecast wind",
      value24: windNorm,
      value72: windNorm,
      weight24: 0.05,
      weight72: 0.06,
      detail: windMax != null ? `${windMax.toFixed(1)} mph max wind` : "No wind proxy available",
    },
    {
      key: "terrain",
      label: "Terrain slope",
      value24: terrainNorm,
      value72: terrainNorm,
      weight24: 0.025,
      weight72: 0.03,
      detail: terrainSlope != null ? `${terrainSlope.toFixed(1)} deg slope` : "No terrain slope available",
    },
    {
      key: "alerts",
      label: "NWS alerts",
      value24: alertsNorm,
      value72: alertsNorm,
      weight24: 0.03,
      weight72: 0.025,
      detail: hasRedFlag ? "Red Flag Warning active" : `${Math.round(alertCount || 0)} active alert(s)`,
    },
    {
      key: "smoke",
      label: "Smoke and PM2.5",
      value24: smokeNorm,
      value72: smokeNorm,
      weight24: 0.02,
      weight72: 0.02,
      detail: representativePm25 != null
        ? `PM2.5 ${representativePm25.toFixed(1)} ug/m3${aqi != null ? ` (AQI ${Math.round(aqi)})` : ""}`
        : "No PM2.5 signal available",
    },
    {
      key: "lightning",
      label: "Lightning ignition risk",
      value24: lightningNorm,
      value72: lightningNorm,
      weight24: 0.005,
      weight72: 0.005,
      detail: lightningRisk ? "Elevated nearby lightning activity" : "No elevated lightning signal",
    },
  ];
  const probability24Base = driverInputs.reduce((sum, driver) => sum + (driver.value24 * driver.weight24), 0);
  const probability72Base = driverInputs.reduce((sum, driver) => sum + (driver.value72 * driver.weight72), 0);

  let probability24 = Math.round(clampNumber(probability24Base * 100, 0, 100));
  let probability72 = Math.round(clampNumber(probability72Base * 100, 0, 100));

  if (alertLevel === "WARNING") {
    probability24 = Math.max(probability24, 72);
    probability72 = Math.max(probability72, 84);
  } else if (alertLevel === "EVACUATE") {
    probability24 = Math.max(probability24, 90);
    probability72 = Math.max(probability72, 96);
  }

  if (etaTo1 != null && etaTo1 <= 6) {
    probability24 = Math.max(probability24, 78);
  }

  const nearbyElevated = distanceMiles != null && distanceMiles <= 25 && threatScore >= 50;
  if (nearbyElevated) {
    probability24 = Math.max(probability24, 60);
  }

  if (probability72 < probability24) {
    probability72 = probability24;
  }

  const likely24h = probability24 >= 55;
  const likely72h = probability72 >= 50 || likely24h;

  let riskToWatchLevel = "LOW";
  if (probability24 >= 75 || probability72 >= 85 || alertLevel === "WARNING" || alertLevel === "EVACUATE") {
    riskToWatchLevel = "HIGH";
  } else if (probability24 >= 45 || probability72 >= 60 || alertLevel === "PREPARE" || likely72h) {
    riskToWatchLevel = "MODERATE";
  }

  if (nearbyElevated && riskToWatchLevel === "MODERATE") {
    riskToWatchLevel = "HIGH";
  }

  const rankedDriverLabels = driverInputs
    .map((driver) => ({
      ...driver,
      contribution24: Number((driver.value24 * driver.weight24 * 100).toFixed(1)),
      contribution72: Number((driver.value72 * driver.weight72 * 100).toFixed(1)),
    }))
    .sort((a, b) => (b.contribution72 + b.contribution24) - (a.contribution72 + a.contribution24));

  const topDrivers = rankedDriverLabels
    .filter((driver) => driver.contribution72 >= 3 || driver.contribution24 >= 3)
    .slice(0, 3)
    .map((driver) => `${driver.label}: ${driver.detail}`);

  const summary = probability24 >= 70
    ? `Modeled impact probability is ${probability24}% in 24h for this watched address.`
    : probability72 >= 55
      ? `Modeled impact probability is ${probability72}% within 72h for this watched address.`
      : "Current modeled conditions suggest lower near-term impact to this watched address.";

  return {
    risk_to_watch_level: riskToWatchLevel,
    likely_impact_within_24h: likely24h,
    likely_impact_within_72h: likely72h,
    impact_probability_24h_pct: probability24,
    impact_probability_72h_pct: probability72,
    impact_driver_breakdown: rankedDriverLabels.map((driver) => ({
      key: driver.key,
      label: driver.label,
      contribution_24h_pct: driver.contribution24,
      contribution_72h_pct: driver.contribution72,
      detail: driver.detail,
    })),
    top_impact_drivers: topDrivers,
    containment_pct_used: incident.percent_contained,
    acres_burned_used: incident.acres_burned,
    containment_source: incident.containment_source,
    acres_source: incident.acres_source,
    distance_to_watch_miles: distanceMiles,
    direction_to_watch: direction,
    fire_location_from_watch: nearestFire.fire_location_from_watch || null,
    summary,
    note: "Impact probability combines containment, distance, spread ETA, weather/wind, terrain, smoke, alerts, and lightning signals for this watched address.",
    eta_to_5_miles_hours: etaTo5,
    eta_to_1_mile_hours: etaTo1,
  };
}

function decisionPriority(state) {
  const rank = {
    SAFE: 0,
    WATCH_CLOSELY: 1,
    PREPARE: 2,
    DANGER_NOW: 3,
  };
  return rank[state] ?? 0;
}

function buildDecisionState({
  activeFireModeled,
  alertLevel,
  threatScore,
  etaTo5,
  etaTo1,
  forecastTrend,
  impactAssessment,
}) {
  if (!activeFireModeled) {
    return {
      state: "SAFE",
      summary: "No active nearby wildfire. Continue routine monitoring for this watched address.",
      factors: ["No active nearby wildfire detected"],
    };
  }

  const likely24h = Boolean(impactAssessment?.likely_impact_within_24h);
  const likely72h = Boolean(impactAssessment?.likely_impact_within_72h);

  if (alertLevel === "EVACUATE" || (etaTo1 != null && etaTo1 <= 2) || threatScore >= 90) {
    return {
      state: "DANGER_NOW",
      summary: "Potential near-immediate impact conditions. Treat this watched address as danger now.",
      factors: ["Very high threat or very short ETA to 1 mile"],
    };
  }

  if (alertLevel === "WARNING" || (etaTo5 != null && etaTo5 <= 8) || (likely24h && threatScore >= 75)) {
    return {
      state: "PREPARE",
      summary: "Elevated near-term impact risk. Prepare evacuation steps for this watched address now.",
      factors: ["High threat with short ETA or warning-level conditions"],
    };
  }

  if (likely24h || likely72h || threatScore >= 45 || (forecastTrend === "WORSENING" && threatScore >= 30)) {
    return {
      state: "WATCH_CLOSELY",
      summary: "Potential impact exists. Watch conditions closely and keep preparations ready.",
      factors: ["Modeled impact window and/or worsening trend"],
    };
  }

  return {
    state: "SAFE",
    summary: "Current modeled conditions suggest lower near-term impact risk for this watched address.",
    factors: ["Low threat and no short-term impact signals"],
  };
}

function weatherDangerLevel(index) {
  if (index >= 80) {
    return "EXTREME";
  }
  if (index >= 60) {
    return "HIGH";
  }
  if (index >= 35) {
    return "MODERATE";
  }
  return "LOW";
}

function buildWeatherImpact(airQuality, spreadRateMph, threatScore, activeFireModeled) {
  const aqi = toNumber(airQuality?.air_quality_index);
  let spreadMultiplier = activeFireModeled ? 1 + clampNumber(spreadRateMph / 12, 0, 0.25) : 1;
  if (activeFireModeled && aqi != null) {
    spreadMultiplier += clampNumber((aqi - 50) / 250, 0, 0.35);
  }

  const fireWeatherIndex = Math.round(clampNumber(
    (threatScore * 0.75) + ((spreadMultiplier - 1) * 55),
    0,
    100,
  ));

  return {
    active_fire_modeled: activeFireModeled,
    air_quality_index: aqi != null ? Math.round(aqi) : null,
    air_quality_category: airQuality?.air_quality_category || null,
    pm25_ug_m3: airQuality?.pm25_ug_m3 ?? null,
    spread_multiplier: Number(spreadMultiplier.toFixed(2)),
    fire_weather_index: fireWeatherIndex,
    fire_danger_level: weatherDangerLevel(fireWeatherIndex),
    observed_at: airQuality?.air_quality_observed_at || null,
  };
}

function buildEvacuationPlan(prediction) {
  const etaTo1 = prediction?.spread_prediction?.eta_to_1_mile_hours;
  const etaTo5 = prediction?.spread_prediction?.eta_to_5_miles_hours;
  let urgency = "MONITOR";
  let actionWindowHours = etaTo5;

  if (prediction.alert_level === "EVACUATE" || (etaTo1 != null && etaTo1 <= 2)) {
    urgency = "EVACUATE_NOW";
    actionWindowHours = etaTo1;
  } else if (prediction.alert_level === "WARNING" || (etaTo1 != null && etaTo1 <= 8)) {
    urgency = "PREPARE_IMMEDIATE";
    actionWindowHours = etaTo1;
  } else if (prediction.alert_level === "PREPARE" || (etaTo5 != null && etaTo5 <= 18)) {
    urgency = "PREPARE_SOON";
    actionWindowHours = etaTo5;
  }

  return {
    location_id: prediction.location_id,
    generated_at: prediction.generated_at,
    urgency,
    recommended_action_window_hours: actionWindowHours,
    actions: prediction.recommendations.slice(0, 5),
  };
}

function buildAnomalyDetection(prediction) {
  if (prediction?.model_status === "no_active_fire_nearby") {
    return {
      location_id: prediction.location_id,
      generated_at: prediction.generated_at,
      is_anomalous: false,
      anomaly_score: 0,
      reasons: ["No active nearby wildfire available for anomaly modeling"],
      insight: "No active nearby wildfire to evaluate for anomalous spread behavior.",
    };
  }

  const spreadRate = prediction?.spread_prediction?.estimated_spread_rate_mph || 0;
  const etaTo5 = prediction?.spread_prediction?.eta_to_5_miles_hours;
  const trend = prediction?.forecast_summary?.trend;
  const reasons = [];
  let score = 0;

  if (spreadRate >= 1.5) {
    reasons.push("Rapid estimated spread rate");
    score += 30;
  }
  if (trend === "WORSENING") {
    reasons.push("Forecast trend indicates worsening threat");
    score += 28;
  }
  if (etaTo5 != null && etaTo5 <= 10) {
    reasons.push("ETA to 5-mile threshold is short");
    score += 30;
  }
  if (prediction.threat_score >= 75) {
    reasons.push("Threat score is in high-risk band");
    score += 22;
  }

  const anomalyScore = clampNumber(score, 0, 100);
  const isAnomalous = anomalyScore >= 60;

  return {
    location_id: prediction.location_id,
    generated_at: prediction.generated_at,
    is_anomalous: isAnomalous,
    anomaly_score: anomalyScore,
    reasons,
    insight: isAnomalous
      ? "Unusual spread conditions detected. Increase monitoring cadence and readiness."
      : "No unusual spread behavior detected right now.",
  };
}

function applyExternalIntelligenceAdjustments(prediction, externalIntelligence) {
  if (!externalIntelligence) {
    return prediction;
  }

  prediction.external_intelligence = externalIntelligence;
  prediction.external_feed_reliability = externalIntelligence?.reliability || null;
  const modifiers = [];
  let scoreBoost = 0;
  const feedStatus = externalIntelligence?.feed_status || {};
  const reliabilityRatio = clampNumber(
    toNumber(externalIntelligence?.reliability?.success_ratio) ?? 1,
    0,
    1,
  );

  if (externalIntelligence?.reliability?.status === "degraded" || externalIntelligence?.reliability?.status === "down") {
    modifiers.push("External feed reliability is degraded; confidence is reduced until upstream signals recover");
  }

  const windMax = toNumber(externalIntelligence?.hrrr_wind_proxy?.max_wind_mph);
  if (
    prediction.model_status === "active_fire_modeled"
    && feedStatus.hrrr_wind_proxy === "ok"
    && windMax != null
    && windMax >= 20
  ) {
    const windBoost = Math.min(12, Math.round((windMax - 18) * 0.8));
    const spreadFactor = 1 + Math.min(0.35, (windMax - 20) / 50);
    scoreBoost += windBoost;
    modifiers.push(`Strong forecast wind (${windMax.toFixed(1)} mph) increases spread potential`);

    const currentDistance = toNumber(prediction?.spread_prediction?.current_distance_miles);
    const spreadRate = toNumber(prediction?.spread_prediction?.estimated_spread_rate_mph);
    if (currentDistance != null && spreadRate != null) {
      const adjustedSpreadRate = Number((spreadRate * spreadFactor).toFixed(2));
      prediction.spread_prediction.estimated_spread_rate_mph = adjustedSpreadRate;
      prediction.spread_prediction.eta_to_5_miles_hours = etaHours(currentDistance, 5, adjustedSpreadRate);
      prediction.spread_prediction.eta_to_2_miles_hours = etaHours(currentDistance, 2, adjustedSpreadRate);
      prediction.spread_prediction.eta_to_1_mile_hours = etaHours(currentDistance, 1, adjustedSpreadRate);
    }
  }

  if (feedStatus.nws_alerts === "ok" && externalIntelligence?.nws_alerts?.has_red_flag_warning) {
    scoreBoost += 8;
    modifiers.push("NWS Red Flag Warning active near this watch");
    if (prediction.decision_state === "SAFE") {
      prediction.decision_state = "WATCH_CLOSELY";
      prediction.decision_summary = "A Red Flag Warning is active nearby. Watch conditions closely and keep preparations ready.";
      prediction.decision_factors = ["NWS Red Flag Warning nearby"];
    }
  }

  const purpleAirPm25 = toNumber(externalIntelligence?.purpleair?.nearest_sensor?.pm25_ug_m3);
  if (feedStatus.purpleair === "ok" && purpleAirPm25 != null && purpleAirPm25 >= 55) {
    scoreBoost += 6;
    modifiers.push(`PurpleAir nearby PM2.5 is elevated (${purpleAirPm25.toFixed(1)} ug/m3)`);
  }

  if (feedStatus.utility_status === "ok" && externalIntelligence?.utility_status?.has_psps_nearby) {
    modifiers.push("Utility PSPS conditions reported nearby");
  }

  if (
    feedStatus.lightning_nowcast === "ok"
    && externalIntelligence?.lightning_nowcast?.elevated_ignition_risk
    && prediction.model_status === "no_active_fire_nearby"
  ) {
    modifiers.push("Elevated nearby lightning activity may increase ignition risk");
    if (prediction.decision_state === "SAFE") {
      prediction.decision_state = "WATCH_CLOSELY";
      prediction.decision_summary = "No active nearby fire, but lightning activity increases near-term ignition risk. Watch closely.";
      prediction.decision_factors = ["Elevated nearby lightning activity"];
    }
  }

  if (scoreBoost > 0 && reliabilityRatio < 1) {
    scoreBoost = Math.round(scoreBoost * reliabilityRatio);
  }

  if (scoreBoost > 0) {
    prediction.threat_score = clampNumber(prediction.threat_score + scoreBoost, 0, 100);
    prediction.risk_level = riskLevelFromScore(prediction.threat_score);
  }

  if (prediction.impact_assessment) {
    if (prediction.threat_score >= 75) {
      prediction.impact_assessment.risk_to_watch_level = "HIGH";
    } else if (prediction.threat_score >= 45 && prediction.impact_assessment.risk_to_watch_level === "LOW") {
      prediction.impact_assessment.risk_to_watch_level = "MODERATE";
    }
  }

  prediction.external_risk_modifiers = modifiers;
  prediction.model_confidence = {
    baseline: prediction.spread_prediction?.confidence ?? 0,
    external_signal_count: modifiers.length,
    confidence_label: externalIntelligence?.reliability?.status === "ok"
      ? (modifiers.length >= 3 ? "ENHANCED" : modifiers.length > 0 ? "BASELINE_PLUS" : "BASELINE")
      : "DEGRADED_EXTERNAL_FEEDS",
  };

  return prediction;
}

async function collectLocationSignals(location, env) {
  const [nearestFire, airQuality] = await Promise.all([
    findNearestFire(location.latitude, location.longitude, env),
    fetchAirQuality(location.latitude, location.longitude),
  ]);

  const fireLevel = nearestFire
    ? fireAlertLevel(
      nearestFire.distance_miles,
      nearestFire.acres_burned || 0,
      nearestFire.percent_contained || 0,
    )
    : "SAFE";
  const airLevel = airQualityAlertLevel(airQuality);
  const alertLevel = combineAlertLevels(fireLevel, airLevel);

  return {
    nearestFire,
    airQuality,
    alertLevel,
    distanceMiles: nearestFire ? Number(nearestFire.distance_miles.toFixed(1)) : null,
    direction: nearestFire ? nearestFire.direction : null,
  };
}

function buildThreatPrediction(location, signals, requestTimestamp, externalIntelligence = null) {
  const { nearestFire, airQuality, distanceMiles, direction, alertLevel } = signals;
  const activeFireModeled = Boolean(nearestFire);
  const incidentContext = deriveIncidentContext(nearestFire, externalIntelligence);
  const spreadRateMph = estimateSpreadRateMph(nearestFire, airQuality);
  const threatScore = activeFireModeled
    ? threatScoreFromSignals({
      distanceMiles,
      acresBurned: incidentContext.acres_burned,
      percentContained: incidentContext.percent_contained,
      airQualityIndex: airQuality?.air_quality_index,
    })
    : 0;
  const riskLevel = riskLevelFromScore(threatScore);
  const etaTo5 = activeFireModeled ? etaHours(distanceMiles, 5, spreadRateMph) : null;
  const etaTo2 = activeFireModeled ? etaHours(distanceMiles, 2, spreadRateMph) : null;
  const etaTo1 = activeFireModeled ? etaHours(distanceMiles, 1, spreadRateMph) : null;
  const forecastPoints = activeFireModeled ? buildForecastPoints(distanceMiles, spreadRateMph, 24, threatScore) : [];
  const maxForecastThreat = forecastPoints.reduce((max, item) => Math.max(max, item.projected_threat_score), threatScore);
  const forecastTrend = activeFireModeled ? (maxForecastThreat > threatScore + 8 ? "WORSENING" : "STABLE") : "NO_ACTIVE_FIRE";
  const weatherImpact = buildWeatherImpact(airQuality, spreadRateMph, threatScore, activeFireModeled);
  const recommendations = nearestFire
    ? recommendationsWithFire(
      alertLevel,
      {
        distance_miles: distanceMiles || nearestFire.distance_miles,
        direction,
        fire_location_from_watch: nearestFire.fire_location_from_watch || null,
      },
      airQuality,
    )
    : recommendationsWithoutFire(airQuality);
  const impactAssessment = buildImpactAssessment({
    nearestFire,
    distanceMiles,
    direction,
    alertLevel,
    threatScore,
    etaTo5,
    etaTo1,
    spreadRateMph,
    weatherImpact,
    externalIntelligence,
    incidentContext,
  });
  const decision = buildDecisionState({
    activeFireModeled,
    alertLevel,
    threatScore,
    etaTo5,
    etaTo1,
    forecastTrend,
    impactAssessment,
  });

  const prediction = {
    location_id: location.location_id,
    location_display_name: location.display_name,
    generated_at: requestTimestamp,
    model_status: activeFireModeled ? "active_fire_modeled" : "no_active_fire_nearby",
    inference_mode: "heuristic_only",
    prediction_engine: {
      kind: "edge_heuristic",
      model_name: "heuristic_edge_v1",
      promoted_model_path: null,
      notes: "Edge output is heuristic-only. Use backend promoted model endpoints for production ML scoring.",
    },
    alert_level: alertLevel,
    threat_score: threatScore,
    risk_level: riskLevel,
    nearest_fire_name: nearestFire?.name || null,
    satellite_signal: nearestFire
      ? {
        source_id: nearestFire.source || null,
        source_label: nearestFire.source_label || normalizeFirmsSourceLabel(nearestFire.source || ""),
        source_priority: toNumber(nearestFire.source_priority),
        confidence_pct: toNumber(nearestFire.confidence),
        frp: toNumber(nearestFire.frp),
      }
      : null,
    incident_context: activeFireModeled ? incidentContext : null,
    nearest_fire_location: nearestFire
      ? {
        latitude: Number(nearestFire.latitude),
        longitude: Number(nearestFire.longitude),
        source: nearestFire.source || null,
      }
      : null,
    spread_prediction: {
      estimated_spread_rate_mph: activeFireModeled ? spreadRateMph : null,
      estimated_direction: activeFireModeled ? direction : null,
      current_distance_miles: activeFireModeled ? distanceMiles : null,
      confidence: activeFireModeled ? 0.72 : 0,
      eta_to_5_miles_hours: etaTo5,
      eta_to_2_miles_hours: etaTo2,
      eta_to_1_mile_hours: etaTo1,
    },
    forecast_summary: {
      hours: 24,
      trend: forecastTrend,
      max_projected_threat_score: maxForecastThreat,
    },
    decision_state: decision.state,
    decision_summary: decision.summary,
    decision_factors: decision.factors,
    impact_assessment: impactAssessment,
    weather_impact: weatherImpact,
    recommendations,
  };

  applyExternalIntelligenceAdjustments(prediction, externalIntelligence);
  prediction.evacuation_plan = buildEvacuationPlan(prediction);
  const roadClosuresNearby = toNumber(externalIntelligence?.road_closures?.closure_count_nearby) || 0;
  if (roadClosuresNearby > 0 && prediction.evacuation_plan?.urgency === "MONITOR") {
    prediction.evacuation_plan.urgency = "PREPARE_SOON";
    prediction.evacuation_plan.actions = [
      "Road closures reported nearby; pre-plan alternate evacuation routes",
      ...prediction.evacuation_plan.actions,
    ].slice(0, 5);
  }
  prediction.anomaly_detection = buildAnomalyDetection(prediction);

  prediction.impact_assessment = buildImpactAssessment({
    nearestFire,
    distanceMiles,
    direction: prediction.spread_prediction?.estimated_direction || direction,
    alertLevel: prediction.alert_level,
    threatScore: prediction.threat_score,
    etaTo5: prediction.spread_prediction?.eta_to_5_miles_hours,
    etaTo1: prediction.spread_prediction?.eta_to_1_mile_hours,
    spreadRateMph: prediction.spread_prediction?.estimated_spread_rate_mph,
    weatherImpact: prediction.weather_impact,
    externalIntelligence,
    incidentContext,
  });

  const refreshedDecision = buildDecisionState({
    activeFireModeled,
    alertLevel: prediction.alert_level,
    threatScore: prediction.threat_score,
    etaTo5: prediction.spread_prediction?.eta_to_5_miles_hours,
    etaTo1: prediction.spread_prediction?.eta_to_1_mile_hours,
    forecastTrend: prediction.forecast_summary?.trend,
    impactAssessment: prediction.impact_assessment,
  });

  if (decisionPriority(refreshedDecision.state) > decisionPriority(prediction.decision_state)) {
    prediction.decision_state = refreshedDecision.state;
    prediction.decision_summary = refreshedDecision.summary;
    prediction.decision_factors = refreshedDecision.factors;
  }

  return prediction;
}

function formatAirQualitySummary(airQuality) {
  if (!airQuality) {
    return null;
  }
  const details = [];
  if (airQuality.air_quality_index != null) {
    details.push(`US AQI ${airQuality.air_quality_index}`);
  }
  if (airQuality.pm25_ug_m3 != null) {
    details.push(`PM2.5 ${Number(airQuality.pm25_ug_m3).toFixed(1)} ug/m3`);
  }
  const detailText = details.length ? ` (${details.join(", ")})` : "";
  return `Current air quality is ${airQuality.air_quality_category || "Unknown"}${detailText}`;
}

function recommendationsWithFire(level, nearestFire, airQuality) {
  const map = {
    SAFE: [
      "Stay informed of fire conditions",
      "Review your evacuation plan",
      "Monitor local news and alerts",
    ],
    MONITOR: [
      "Stay informed of fire conditions",
      "Review your evacuation plan",
      "Monitor local news and alerts",
    ],
    PREPARE: [
      "Alert: fire activity is approaching your area",
      `Fire is about ${nearestFire.distance_miles.toFixed(1)} miles away (${nearestFire.fire_location_from_watch || "location direction unavailable"} of you)`,
      "Prepare essential items and review evacuation routes",
    ],
    WARNING: [
      "Urgent: prepare to evacuate",
      `Fire is about ${nearestFire.distance_miles.toFixed(1)} miles away (${nearestFire.fire_location_from_watch || "location direction unavailable"} of you)`,
      "Follow local emergency updates closely",
    ],
    EVACUATE: [
      "Evacuate immediately",
      `Fire is about ${nearestFire.distance_miles.toFixed(1)} miles away (${nearestFire.fire_location_from_watch || "location direction unavailable"} of you)`,
      "Follow official evacuation instructions",
    ],
  };
  const selected = [...(map[level] || map.SAFE)];
  const airSummary = formatAirQualitySummary(airQuality);
  if (airSummary) {
    selected.push(airSummary);
    selected.push("Use indoor air filtration if AQI is elevated and avoid strenuous outdoor activity");
  }
  return selected;
}

function recommendationsWithoutFire(airQuality) {
  const selected = [
    "No active fires detected nearby",
    "Stay informed of weather conditions",
    "Review evacuation plan regularly",
  ];
  const airSummary = formatAirQualitySummary(airQuality);
  if (airSummary) {
    selected.push(airSummary);
    selected.push("Limit outdoor exposure if smoke or pollution is affecting your area");
  }
  return selected;
}

async function checkD1Dependency(db) {
  const startedAt = Date.now();
  if (!db) {
    return {
      status: "down",
      latency_ms: Date.now() - startedAt,
      error: "FIREWATCH_DB binding is not configured",
    };
  }
  try {
    const row = await db.prepare("SELECT COUNT(*) AS active_count FROM locations WHERE is_active = 1").first();
    return {
      status: "ok",
      latency_ms: Date.now() - startedAt,
      active_location_count: toNumber(row?.active_count) ?? 0,
    };
  } catch (error) {
    return {
      status: "down",
      latency_ms: Date.now() - startedAt,
      error: String(error?.message || error),
    };
  }
}

async function checkEonetDependency() {
  const probes = [
    {
      name: "events_open_wildfires",
      url: "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&category=wildfires&limit=1",
      includeEventCount: true,
    },
    {
      name: "events_30d_wildfires",
      url: "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&category=wildfires&days=30&limit=5",
      includeEventCount: true,
    },
    {
      name: "category_wildfires",
      url: "https://eonet.gsfc.nasa.gov/api/v3/categories/wildfires",
      includeEventCount: false,
    },
  ];

  const failures = [];
  for (const probe of probes) {
    const result = await fetchExternal(probe.url, {
      kind: "eonet",
      parseAs: "json",
      headers: {
        Accept: "application/json",
        "User-Agent": "firewatch-edge/1.0",
      },
    });

    if (!result.ok) {
      failures.push({
        probe: probe.name,
        reason: result.reason,
        upstream_status: result.status,
      });
      continue;
    }

    const events = Array.isArray(result.data?.events) ? result.data.events : [];
    return {
      status: "ok",
      upstream_status: result.status,
      latency_ms: result.duration_ms,
      probe: probe.name,
      sample_event_count: probe.includeEventCount ? events.length : null,
      recovered_via_fallback: failures.length > 0,
      failed_probes: failures,
    };
  }

  const lastFailure = failures[failures.length - 1] || {};
  return {
    status: "degraded",
    reason: lastFailure.reason || "upstream_unavailable",
    upstream_status: lastFailure.upstream_status || null,
    latency_ms: null,
    probe: lastFailure.probe || null,
    failed_probes: failures,
  };
}

async function checkFirmsDependency(env) {
  if (!env.FIRMS_MAP_KEY) {
    return {
      status: "not_configured",
      reason: "FIRMS_MAP_KEY missing",
    };
  }

  const source = String(env.FIRMS_SOURCE_IDS || DEFAULT_FIRMS_SOURCES.join(","))
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)[0] || DEFAULT_FIRMS_SOURCES[0];
  const bbox = buildLocationBbox(34.1434, -118.3906, 25);
  const dayRange = 1;
  const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${env.FIRMS_MAP_KEY}/${source}/${bbox}/${dayRange}`;
  const result = await fetchExternal(url, { kind: "firms", parseAs: "text" });
  if (!result.ok) {
    return {
      status: "degraded",
      reason: result.reason,
      upstream_status: result.status,
      latency_ms: result.duration_ms,
      source,
    };
  }

  const rows = parseCsv(result.data);
  return {
    status: "ok",
    upstream_status: result.status,
    latency_ms: result.duration_ms,
    source,
    sample_hotspot_rows: rows.length,
  };
}

async function checkAqiDependency() {
  const url = new URL("https://air-quality-api.open-meteo.com/v1/air-quality");
  url.searchParams.set("latitude", "34.1434");
  url.searchParams.set("longitude", "-118.3906");
  url.searchParams.set("current", "us_aqi,pm2_5");
  url.searchParams.set("timezone", "auto");
  const result = await fetchExternal(url.toString(), { kind: "aqi", parseAs: "json" });
  if (!result.ok) {
    return {
      status: "degraded",
      reason: result.reason,
      upstream_status: result.status,
      latency_ms: result.duration_ms,
    };
  }

  const current = result.data?.current || {};
  return {
    status: "ok",
    upstream_status: result.status,
    latency_ms: result.duration_ms,
    sample_aqi: toNumber(current.us_aqi),
    sample_pm25: toNumber(current.pm2_5),
  };
}

function summarizeDiagnosticsStatus(dependencies) {
  if (dependencies.d1?.status !== "ok") {
    return "unhealthy";
  }
  const hasDegraded = Object.values(dependencies).some((value) => value?.status === "degraded");
  const hasNotConfigured = Object.values(dependencies).some((value) => value?.status === "not_configured");
  if (hasDegraded || hasNotConfigured) {
    return "degraded";
  }
  return "healthy";
}

async function runHealthDiagnostics(db, env) {
  const [d1, eonet, firms, aqi] = await Promise.all([
    checkD1Dependency(db),
    checkEonetDependency(),
    checkFirmsDependency(env),
    checkAqiDependency(),
  ]);
  const dependencies = { d1, eonet, firms, aqi };
  const summary = {
    ok_count: Object.values(dependencies).filter((item) => item.status === "ok").length,
    degraded_count: Object.values(dependencies).filter((item) => item.status === "degraded").length,
    not_configured_count: Object.values(dependencies).filter((item) => item.status === "not_configured").length,
    down_count: Object.values(dependencies).filter((item) => item.status === "down").length,
  };

  return {
    status: summarizeDiagnosticsStatus(dependencies),
    edge: true,
    timestamp: new Date().toISOString(),
    dependencies,
    summary,
  };
}

function extractEdgeProvenance(prediction) {
  return {
    generated_at: prediction?.generated_at || null,
    model_status: prediction?.model_status || null,
    inference_mode: prediction?.inference_mode || null,
    prediction_engine: prediction?.prediction_engine || null,
    model_confidence: prediction?.model_confidence || null,
    external_feed_reliability: prediction?.external_feed_reliability || null,
  };
}

async function probeBackendModelProvenance(locationId, env) {
  const baseOrigin = String(env.PROVENANCE_BACKEND_BASE_URL || "").trim().replace(/\/+$/, "");
  if (!baseOrigin) {
    return {
      status: "not_configured",
      reason: "PROVENANCE_BACKEND_BASE_URL is not configured",
      model_provenance: null,
    };
  }

  const url = `${baseOrigin}/api/v1/predictions/threat/${encodeURIComponent(locationId)}`;
  const result = await fetchExternal(url, {
    kind: "external",
    parseAs: "json",
    headers: {
      accept: "application/json",
    },
  });

  if (!result.ok) {
    return {
      status: "degraded",
      reason: result.reason,
      upstream_status: result.status,
      latency_ms: result.duration_ms,
      model_provenance: null,
      request_url: url,
    };
  }

  const payload = result.data || {};
  const modelProvenance = payload.model_provenance || payload?.predictions?.model_provenance || null;
  return {
    status: "ok",
    upstream_status: result.status,
    latency_ms: result.duration_ms,
    model_provenance: modelProvenance,
    request_url: url,
  };
}

async function resolveAddress(address) {
  const normalized = normalizeAddress(address);
  if (!normalized) {
    return null;
  }

  const queries = isZipCode(normalized)
    ? [`${normalized} USA`, normalized]
    : [normalized];

  for (const query of queries) {
    const url = new URL("https://nominatim.openstreetmap.org/search");
    url.searchParams.set("format", "jsonv2");
    url.searchParams.set("limit", "1");
    url.searchParams.set("q", query);

    const response = await fetchExternal(url.toString(), {
      kind: "geocoding",
      headers: {
        accept: "application/json",
        "user-agent": "firewatch-edge/1.0",
      },
      parseAs: "json",
    });
    if (!response.ok) {
      continue;
    }

    const payload = response.data;
    if (!Array.isArray(payload) || payload.length === 0) {
      continue;
    }

    const latitude = toNumber(payload[0]?.lat);
    const longitude = toNumber(payload[0]?.lon);
    if (latitude == null || longitude == null) {
      continue;
    }
    return {
      latitude,
      longitude,
      geocoded_at: new Date().toISOString(),
    };
  }

  return null;
}

async function fetchAirQuality(latitude, longitude) {
  const url = new URL("https://air-quality-api.open-meteo.com/v1/air-quality");
  url.searchParams.set("latitude", String(latitude));
  url.searchParams.set("longitude", String(longitude));
  url.searchParams.set("current", "us_aqi,pm2_5");
  url.searchParams.set("timezone", "auto");

  try {
    const response = await fetchExternal(url.toString(), {
      kind: "aqi",
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }
    const payload = response.data;
    const current = payload?.current || {};
    const aqi = toNumber(current.us_aqi);
    const pm25 = toNumber(current.pm2_5);
    if (aqi == null && pm25 == null) {
      return null;
    }

    let category = "Good";
    if (aqi != null) {
      if (aqi >= 301) {
        category = "Hazardous";
      } else if (aqi >= 201) {
        category = "Very Unhealthy";
      } else if (aqi >= 151) {
        category = "Unhealthy";
      } else if (aqi >= 101) {
        category = "Unhealthy for Sensitive Groups";
      } else if (aqi >= 51) {
        category = "Moderate";
      }
    }

    return {
      air_quality_index: aqi != null ? Math.round(aqi) : null,
      pm25_ug_m3: pm25,
      air_quality_category: category,
      air_quality_source: "Open-Meteo Air Quality",
      air_quality_observed_at: current.time ? `${current.time}Z` : null,
    };
  } catch {
    return null;
  }
}

function bboxFromRadiusMiles(latitude, radiusMiles) {
  const radiusKm = radiusMiles * 1.60934;
  const latDelta = radiusKm / 111.32;
  const lonScale = Math.max(0.1, Math.cos((latitude * Math.PI) / 180));
  const lonDelta = radiusKm / (111.32 * lonScale);
  return {
    west: Number((0 - lonDelta).toFixed(5)),
    east: Number((0 + lonDelta).toFixed(5)),
    latDelta,
    lonDelta,
  };
}

function buildLocationBbox(latitude, longitude, radiusMiles) {
  const radiusKm = radiusMiles * 1.60934;
  const latDelta = radiusKm / 111.32;
  const lonScale = Math.max(0.1, Math.cos((latitude * Math.PI) / 180));
  const lonDelta = radiusKm / (111.32 * lonScale);
  const west = longitude - lonDelta;
  const east = longitude + lonDelta;
  const south = latitude - latDelta;
  const north = latitude + latDelta;
  return `${west.toFixed(5)},${south.toFixed(5)},${east.toFixed(5)},${north.toFixed(5)}`;
}

function parseCsv(text) {
  const lines = String(text || "").trim().split(/\r?\n/);
  if (lines.length < 2) {
    return [];
  }
  const headers = lines[0].split(",").map((value) => value.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = (values[index] || "").trim();
    });
    return row;
  });
}

function cardinalToBearing(cardinal) {
  const normalized = String(cardinal || "").trim().toUpperCase();
  const map = {
    N: 0,
    NNE: 22.5,
    NE: 45,
    ENE: 67.5,
    E: 90,
    ESE: 112.5,
    SE: 135,
    SSE: 157.5,
    S: 180,
    SSW: 202.5,
    SW: 225,
    WSW: 247.5,
    W: 270,
    WNW: 292.5,
    NW: 315,
    NNW: 337.5,
  };
  return map[normalized] ?? null;
}

function parseWindSpeedMph(rawValue) {
  const raw = String(rawValue || "").trim().toLowerCase();
  if (!raw) {
    return null;
  }

  const values = [...raw.matchAll(/\d+(?:\.\d+)?/g)].map((match) => Number(match[0]));
  if (!values.length) {
    return null;
  }

  const unit = raw.includes("km") ? "kmh" : raw.includes("kt") || raw.includes("knot") ? "knots" : "mph";
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;

  if (unit === "kmh") {
    return Number((average * 0.621371).toFixed(1));
  }
  if (unit === "knots") {
    return Number((average * 1.15078).toFixed(1));
  }
  return Number(average.toFixed(1));
}

function severityRank(value) {
  const normalized = String(value || "").toUpperCase();
  const rank = {
    UNKNOWN: 0,
    MINOR: 1,
    MODERATE: 2,
    SEVERE: 3,
    EXTREME: 4,
  };
  return rank[normalized] ?? 0;
}

function parseFeaturePoint(geometry) {
  if (!geometry || !geometry.type) {
    return null;
  }

  if (geometry.type === "Point" && Array.isArray(geometry.coordinates) && geometry.coordinates.length >= 2) {
    const lon = toNumber(geometry.coordinates[0]);
    const lat = toNumber(geometry.coordinates[1]);
    if (lat == null || lon == null) {
      return null;
    }
    return { latitude: lat, longitude: lon };
  }

  if (geometry.type === "LineString" && Array.isArray(geometry.coordinates) && geometry.coordinates.length > 0) {
    const first = geometry.coordinates[0];
    const lon = toNumber(first?.[0]);
    const lat = toNumber(first?.[1]);
    if (lat == null || lon == null) {
      return null;
    }
    return { latitude: lat, longitude: lon };
  }

  if (geometry.type === "Polygon" && Array.isArray(geometry.coordinates) && geometry.coordinates.length > 0) {
    const ring = geometry.coordinates[0];
    if (!Array.isArray(ring) || !ring.length) {
      return null;
    }
    let latSum = 0;
    let lonSum = 0;
    let count = 0;
    for (const point of ring) {
      const lon = toNumber(point?.[0]);
      const lat = toNumber(point?.[1]);
      if (lat == null || lon == null) {
        continue;
      }
      latSum += lat;
      lonSum += lon;
      count += 1;
    }
    if (!count) {
      return null;
    }
    return {
      latitude: Number((latSum / count).toFixed(5)),
      longitude: Number((lonSum / count).toFixed(5)),
    };
  }

  return null;
}

function nearestFeature(features, latitude, longitude) {
  let nearest = null;
  for (const feature of features) {
    const point = parseFeaturePoint(feature?.geometry);
    if (!point) {
      continue;
    }
    const distance = haversineMiles(latitude, longitude, point.latitude, point.longitude);
    if (!nearest || distance < nearest.distance_miles) {
      nearest = {
        feature,
        distance_miles: Number(distance.toFixed(1)),
        point,
      };
    }
  }
  return nearest;
}

async function fetchNwsAlerts(latitude, longitude) {
  const url = `https://api.weather.gov/alerts/active?point=${latitude},${longitude}`;
  try {
    const response = await fetchExternal(url, {
      kind: "nws",
      headers: {
        Accept: "application/geo+json",
        "User-Agent": "firewatch-edge/1.0",
      },
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }

    const payload = response.data;
    const features = Array.isArray(payload?.features) ? payload.features : [];
    const alerts = features
      .map((feature) => {
        const properties = feature?.properties || {};
        return {
          event: properties.event || null,
          severity: properties.severity || "Unknown",
          urgency: properties.urgency || "Unknown",
          certainty: properties.certainty || "Unknown",
          headline: properties.headline || null,
          effective: properties.effective || null,
          expires: properties.expires || null,
        };
      })
      .filter((alert) => alert.event || alert.headline);

    if (!alerts.length) {
      return {
        source: "NWS Alerts",
        alert_count: 0,
        highest_severity: "UNKNOWN",
        has_red_flag_warning: false,
        has_evacuation_language: false,
        alerts: [],
      };
    }

    const highest = alerts.reduce((best, current) => (
      severityRank(current.severity) > severityRank(best) ? current.severity : best
    ), "UNKNOWN");
    const hasRedFlagWarning = alerts.some((alert) => /red\s*flag/i.test(`${alert.event || ""} ${alert.headline || ""}`));
    const hasEvacuationLanguage = alerts.some((alert) => /evacuat|shelter\s+in\s+place/i.test(`${alert.event || ""} ${alert.headline || ""}`));

    return {
      source: "NWS Alerts",
      alert_count: alerts.length,
      highest_severity: String(highest || "UNKNOWN").toUpperCase(),
      has_red_flag_warning: hasRedFlagWarning,
      has_evacuation_language: hasEvacuationLanguage,
      alerts: alerts.slice(0, 5),
    };
  } catch {
    return null;
  }
}

async function fetchNwsHourlyWind(latitude, longitude) {
  const pointUrl = `https://api.weather.gov/points/${latitude},${longitude}`;
  try {
    const pointResponse = await fetchExternal(pointUrl, {
      kind: "nws",
      headers: {
        Accept: "application/geo+json",
        "User-Agent": "firewatch-edge/1.0",
      },
      parseAs: "json",
    });
    if (!pointResponse.ok) {
      return null;
    }

    const pointPayload = pointResponse.data;
    const hourlyUrl = pointPayload?.properties?.forecastHourly;
    if (!hourlyUrl) {
      return null;
    }

    const hourlyResponse = await fetchExternal(hourlyUrl, {
      kind: "nws",
      headers: {
        Accept: "application/geo+json",
        "User-Agent": "firewatch-edge/1.0",
      },
      parseAs: "json",
    });
    if (!hourlyResponse.ok) {
      return null;
    }

    const hourlyPayload = hourlyResponse.data;
    const periods = Array.isArray(hourlyPayload?.properties?.periods)
      ? hourlyPayload.properties.periods.slice(0, 8)
      : [];
    if (!periods.length) {
      return null;
    }

    const windSamples = periods
      .map((period) => ({
        start_time: period?.startTime || null,
        wind_speed_mph: parseWindSpeedMph(period?.windSpeed),
        wind_direction_cardinal: period?.windDirection || null,
      }))
      .filter((sample) => sample.wind_speed_mph != null);
    if (!windSamples.length) {
      return null;
    }

    const maxWind = windSamples.reduce((max, sample) => Math.max(max, sample.wind_speed_mph), 0);
    const avgWind = windSamples.reduce((sum, sample) => sum + sample.wind_speed_mph, 0) / windSamples.length;
    const directionCounts = new Map();
    for (const sample of windSamples) {
      const direction = sample.wind_direction_cardinal || "VAR";
      directionCounts.set(direction, (directionCounts.get(direction) || 0) + 1);
    }
    const prevailingDirection = [...directionCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null;

    return {
      source: "NWS Hourly Forecast",
      model_hint: "NWS blended hourly guidance",
      sample_count: windSamples.length,
      max_wind_mph: Number(maxWind.toFixed(1)),
      avg_wind_mph: Number(avgWind.toFixed(1)),
      prevailing_direction_cardinal: prevailingDirection,
      prevailing_direction_degrees: cardinalToBearing(prevailingDirection),
      periods: windSamples,
    };
  } catch {
    return null;
  }
}

async function fetchGeoJson(url, headers = {}) {
  try {
    const response = await fetchExternal(url, {
      kind: "perimeter",
      headers,
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }
    const payload = response.data;
    return Array.isArray(payload?.features) ? payload.features : [];
  } catch {
    return null;
  }
}

async function fetchIncidentPerimeters(latitude, longitude, env) {
  if (!env.FIRE_PERIMETER_GEOJSON_URL) {
    return null;
  }
  const features = await fetchGeoJson(env.FIRE_PERIMETER_GEOJSON_URL);
  if (!features) {
    return null;
  }
  const nearest = nearestFeature(features, latitude, longitude);
  if (!nearest) {
    return {
      source: "Configured perimeter feed",
      incident_count: features.length,
      nearest_incident: null,
    };
  }
  const properties = nearest.feature?.properties || {};
  return {
    source: "Configured perimeter feed",
    incident_count: features.length,
    nearest_incident: {
      name: properties.IncidentName || properties.incident_name || properties.name || properties.IRWINID || "Incident",
      acres: toNumber(properties.DailyAcres || properties.acres || properties.Acres),
      percent_contained: toNumber(properties.PercentContained || properties.percent_contained || properties.Containment),
      growth_acres_24h: toNumber(properties.DailyGrowth || properties.growth_acres_24h),
      distance_miles: nearest.distance_miles,
    },
  };
}

async function fetchHmsSmokePlume(latitude, longitude, env) {
  if (!env.HMS_SMOKE_GEOJSON_URL) {
    return null;
  }
  const features = await fetchGeoJson(env.HMS_SMOKE_GEOJSON_URL);
  if (!features) {
    return null;
  }
  const nearest = nearestFeature(features, latitude, longitude);
  if (!nearest) {
    return {
      source: "Configured HMS smoke feed",
      plume_count: features.length,
      nearest_plume: null,
    };
  }
  const properties = nearest.feature?.properties || {};
  return {
    source: "Configured HMS smoke feed",
    plume_count: features.length,
    nearest_plume: {
      density: properties.DENSITY || properties.density || properties.SMOKE || "UNKNOWN",
      category: properties.Category || properties.category || null,
      distance_miles: nearest.distance_miles,
      observed_at: properties.OBS_TIME || properties.observed_at || properties.time || null,
    },
  };
}

async function fetchPurpleAir(latitude, longitude, env) {
  if (!env.PURPLEAIR_API_KEY) {
    return null;
  }

  const latDelta = 0.2;
  const lonDelta = 0.2;
  const url = new URL("https://api.purpleair.com/v1/sensors");
  url.searchParams.set("fields", "name,latitude,longitude,pm2.5_atm,confidence,last_seen");
  url.searchParams.set("location_type", "0");
  url.searchParams.set("nwlng", String(longitude - lonDelta));
  url.searchParams.set("nwlat", String(latitude + latDelta));
  url.searchParams.set("selng", String(longitude + lonDelta));
  url.searchParams.set("selat", String(latitude - latDelta));

  try {
    const response = await fetchExternal(url.toString(), {
      kind: "purpleair",
      headers: {
        "X-API-Key": env.PURPLEAIR_API_KEY,
      },
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }

    const payload = response.data;
    const fields = Array.isArray(payload?.fields) ? payload.fields : [];
    const data = Array.isArray(payload?.data) ? payload.data : [];
    if (!fields.length || !data.length) {
      return {
        source: "PurpleAir",
        sensor_count: 0,
        nearest_sensor: null,
      };
    }

    const getField = (row, fieldName) => {
      const index = fields.indexOf(fieldName);
      if (index === -1) {
        return null;
      }
      return row[index];
    };

    let nearest = null;
    for (const row of data) {
      const sensorLat = toNumber(getField(row, "latitude"));
      const sensorLon = toNumber(getField(row, "longitude"));
      if (sensorLat == null || sensorLon == null) {
        continue;
      }
      const distance = haversineMiles(latitude, longitude, sensorLat, sensorLon);
      if (!nearest || distance < nearest.distance_miles) {
        nearest = {
          name: getField(row, "name") || "PurpleAir sensor",
          pm25_ug_m3: toNumber(getField(row, "pm2.5_atm")),
          confidence_pct: toNumber(getField(row, "confidence")),
          last_seen_epoch: toNumber(getField(row, "last_seen")),
          distance_miles: Number(distance.toFixed(1)),
        };
      }
    }

    return {
      source: "PurpleAir",
      sensor_count: data.length,
      nearest_sensor: nearest,
    };
  } catch {
    return null;
  }
}

async function fetchRoadClosures(latitude, longitude, env) {
  if (!env.ROAD_CLOSURES_GEOJSON_URL) {
    return null;
  }
  const features = await fetchGeoJson(env.ROAD_CLOSURES_GEOJSON_URL);
  if (!features) {
    return null;
  }
  const nearby = features
    .map((feature) => {
      const point = parseFeaturePoint(feature?.geometry);
      if (!point) {
        return null;
      }
      const distance = haversineMiles(latitude, longitude, point.latitude, point.longitude);
      return { feature, distance_miles: Number(distance.toFixed(1)) };
    })
    .filter(Boolean)
    .filter((item) => item.distance_miles <= 30)
    .sort((a, b) => a.distance_miles - b.distance_miles);

  return {
    source: "Configured road closure feed",
    closure_count_nearby: nearby.length,
    nearest_closure: nearby[0]
      ? {
        distance_miles: nearby[0].distance_miles,
        road_name: nearby[0].feature?.properties?.road_name || nearby[0].feature?.properties?.name || null,
        status: nearby[0].feature?.properties?.status || nearby[0].feature?.properties?.closure_status || "CLOSED",
      }
      : null,
  };
}

async function fetchTerrainContext(latitude, longitude) {
  const offset = 0.02;
  const latitudes = [latitude, latitude + offset, latitude - offset, latitude, latitude];
  const longitudes = [longitude, longitude, longitude, longitude + offset, longitude - offset];
  const url = new URL("https://api.open-meteo.com/v1/elevation");
  url.searchParams.set("latitude", latitudes.join(","));
  url.searchParams.set("longitude", longitudes.join(","));

  try {
    const response = await fetchExternal(url.toString(), {
      kind: "terrain",
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }
    const payload = response.data;
    const elevations = Array.isArray(payload?.elevation) ? payload.elevation.map((value) => toNumber(value)) : [];
    if (elevations.length < 5 || elevations.some((value) => value == null)) {
      return null;
    }

    const center = elevations[0];
    const north = elevations[1];
    const south = elevations[2];
    const east = elevations[3];
    const west = elevations[4];
    const metersPerDegLat = 111320;
    const metersPerDegLon = Math.max(1, Math.cos((latitude * Math.PI) / 180) * 111320);
    const dzDy = (north - south) / (2 * offset * metersPerDegLat);
    const dzDx = (east - west) / (2 * offset * metersPerDegLon);
    const slopeRadians = Math.atan(Math.sqrt((dzDx ** 2) + (dzDy ** 2)));
    const slopeDegrees = Number(((slopeRadians * 180) / Math.PI).toFixed(1));
    const aspectRadians = Math.atan2(dzDy, -dzDx);
    const aspectDegrees = Number((((aspectRadians * 180) / Math.PI) + 360).toFixed(1)) % 360;

    return {
      source: "Open-Meteo elevation",
      elevation_m: Number(center.toFixed(1)),
      slope_degrees: slopeDegrees,
      aspect_degrees: aspectDegrees,
    };
  } catch {
    return null;
  }
}

async function fetchUtilityStatus(latitude, longitude, env) {
  if (!env.UTILITY_STATUS_URL) {
    return null;
  }
  try {
    const response = await fetchExternal(env.UTILITY_STATUS_URL, {
      kind: "utility",
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }
    const payload = response.data;
    const entries = Array.isArray(payload?.entries) ? payload.entries : Array.isArray(payload) ? payload : [];
    const nearby = entries
      .map((entry) => {
        const lat = toNumber(entry?.latitude);
        const lon = toNumber(entry?.longitude);
        if (lat == null || lon == null) {
          return null;
        }
        const distance = haversineMiles(latitude, longitude, lat, lon);
        return {
          status: entry?.status || "unknown",
          utility: entry?.utility || null,
          distance_miles: Number(distance.toFixed(1)),
          psps: Boolean(entry?.psps),
        };
      })
      .filter(Boolean)
      .filter((entry) => entry.distance_miles <= 30)
      .sort((a, b) => a.distance_miles - b.distance_miles);

    return {
      source: "Configured utility status feed",
      nearby_count: nearby.length,
      nearest: nearby[0] || null,
      has_psps_nearby: nearby.some((entry) => entry.psps),
    };
  } catch {
    return null;
  }
}

async function fetchLightningNowcast(latitude, longitude, env) {
  if (!env.LIGHTNING_NOWCAST_URL) {
    return null;
  }
  try {
    const response = await fetchExternal(env.LIGHTNING_NOWCAST_URL, {
      kind: "lightning",
      parseAs: "json",
    });
    if (!response.ok) {
      return null;
    }
    const payload = response.data;
    const strikes = Array.isArray(payload?.strikes) ? payload.strikes : Array.isArray(payload) ? payload : [];
    const nearby = strikes
      .map((strike) => {
        const lat = toNumber(strike?.latitude);
        const lon = toNumber(strike?.longitude);
        if (lat == null || lon == null) {
          return null;
        }
        const distance = haversineMiles(latitude, longitude, lat, lon);
        return {
          distance_miles: Number(distance.toFixed(1)),
          observed_at: strike?.observed_at || strike?.time || null,
        };
      })
      .filter(Boolean)
      .filter((entry) => entry.distance_miles <= 40)
      .sort((a, b) => a.distance_miles - b.distance_miles);

    return {
      source: "Configured lightning nowcast feed",
      strikes_nearby_count: nearby.length,
      nearest_strike: nearby[0] || null,
      elevated_ignition_risk: nearby.length >= 3,
    };
  } catch {
    return null;
  }
}

function configuredFeedStatus(isConfigured, payload) {
  if (!isConfigured) {
    return "not_configured";
  }
  return payload ? "ok" : "degraded";
}

function summarizeExternalFeedReliability(feedStatus) {
  const values = Object.values(feedStatus || {});
  const expected = values.filter((status) => status !== "not_configured");
  const okCount = expected.filter((status) => status === "ok").length;
  const degradedCount = expected.filter((status) => status === "degraded").length;
  const expectedCount = expected.length;
  const successRatio = expectedCount > 0 ? okCount / expectedCount : 1;

  let status = "ok";
  if (expectedCount === 0) {
    status = "not_configured";
  } else if (successRatio < 0.6) {
    status = "down";
  } else if (successRatio < 0.9) {
    status = "degraded";
  }

  return {
    status,
    ok_count: okCount,
    degraded_count: degradedCount,
    not_configured_count: values.filter((value) => value === "not_configured").length,
    expected_count: expectedCount,
    success_ratio: Number(successRatio.toFixed(3)),
    slo_target_success_ratio: 0.9,
    slo_met: expectedCount > 0 ? successRatio >= 0.9 : null,
  };
}

async function collectExternalIntelligence(location, env) {
  const latitude = Number(location.latitude);
  const longitude = Number(location.longitude);
  const [
    nwsAlerts,
    nwsHourlyWind,
    incidentPerimeters,
    hmsSmoke,
    purpleAir,
    roadClosures,
    terrain,
    utilityStatus,
    lightningNowcast,
  ] = await Promise.all([
    fetchNwsAlerts(latitude, longitude),
    fetchNwsHourlyWind(latitude, longitude),
    fetchIncidentPerimeters(latitude, longitude, env),
    fetchHmsSmokePlume(latitude, longitude, env),
    fetchPurpleAir(latitude, longitude, env),
    fetchRoadClosures(latitude, longitude, env),
    fetchTerrainContext(latitude, longitude),
    fetchUtilityStatus(latitude, longitude, env),
    fetchLightningNowcast(latitude, longitude, env),
  ]);

  const feedStatus = {
    nws_alerts: configuredFeedStatus(true, nwsAlerts),
    hrrr_wind_proxy: configuredFeedStatus(true, nwsHourlyWind),
    incident_perimeters: configuredFeedStatus(Boolean(env.FIRE_PERIMETER_GEOJSON_URL), incidentPerimeters),
    hms_smoke: configuredFeedStatus(Boolean(env.HMS_SMOKE_GEOJSON_URL), hmsSmoke),
    purpleair: configuredFeedStatus(Boolean(env.PURPLEAIR_API_KEY), purpleAir),
    road_closures: configuredFeedStatus(Boolean(env.ROAD_CLOSURES_GEOJSON_URL), roadClosures),
    terrain_context: configuredFeedStatus(true, terrain),
    utility_status: configuredFeedStatus(Boolean(env.UTILITY_STATUS_URL), utilityStatus),
    lightning_nowcast: configuredFeedStatus(Boolean(env.LIGHTNING_NOWCAST_URL), lightningNowcast),
  };
  const reliability = summarizeExternalFeedReliability(feedStatus);

  return {
    nws_alerts: nwsAlerts,
    hrrr_wind_proxy: nwsHourlyWind,
    incident_perimeters: incidentPerimeters,
    hms_smoke: hmsSmoke,
    purpleair: purpleAir,
    road_closures: roadClosures,
    terrain_context: terrain,
    utility_status: utilityStatus,
    lightning_nowcast: lightningNowcast,
    feed_status: feedStatus,
    reliability,
  };
}

function normalizeFirmsSourceLabel(source) {
  const upper = String(source || "").toUpperCase();
  if (upper === "VIIRS_SNPP_NRT") {
    return "VIIRS SNPP thermal anomaly";
  }
  if (upper === "VIIRS_NOAA21_NRT") {
    return "VIIRS NOAA-21 thermal anomaly";
  }
  if (upper === "VIIRS_NOAA20_NRT") {
    return "VIIRS NOAA-20 thermal anomaly";
  }
  if (upper === "MODIS_NRT") {
    return "MODIS thermal anomaly";
  }
  if (upper.startsWith("VIIRS")) {
    return "VIIRS thermal anomaly";
  }
  if (upper.startsWith("MODIS")) {
    return "MODIS thermal anomaly";
  }
  return `${source} thermal anomaly`;
}

function firmsSourcePriority(source) {
  const upper = String(source || "").toUpperCase();
  if (upper === "VIIRS_SNPP_NRT") {
    return 1.0;
  }
  if (upper === "VIIRS_NOAA21_NRT") {
    return 0.96;
  }
  if (upper === "VIIRS_NOAA20_NRT") {
    return 0.94;
  }
  if (upper === "MODIS_NRT") {
    return 0.75;
  }
  if (upper.startsWith("VIIRS")) {
    return 0.9;
  }
  return 0;
}

function parseFirmsConfidence(row) {
  const raw = String(row?.confidence || "").trim().toLowerCase();
  if (!raw) {
    return null;
  }

  const numeric = Number(raw);
  if (Number.isFinite(numeric)) {
    return numeric;
  }

  if (raw === "l" || raw === "low") {
    return 30;
  }
  if (raw === "n" || raw === "nominal") {
    return 60;
  }
  if (raw === "h" || raw === "high") {
    return 85;
  }
  return null;
}

function parseFirmsType(row) {
  const raw = String(row?.type || "").trim();
  if (!raw) {
    return null;
  }

  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : null;
}

function parseFirmsFrp(row) {
  const raw = String(row?.frp || "").trim();
  if (!raw) {
    return null;
  }

  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : null;
}

async function fetchFirmsHotspots(latitude, longitude, env) {
  if (!env.FIRMS_MAP_KEY) {
    return [];
  }

  const sources = String(env.FIRMS_SOURCE_IDS || DEFAULT_FIRMS_SOURCES.join(","))
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const radiusMiles = Number(env.FIRMS_RADIUS_MILES || 75);
  const dayRange = Math.min(5, Math.max(1, Number(env.FIRMS_DAY_RANGE || 1)));
  const minConfidence = Math.min(100, Math.max(0, Number(env.FIRMS_MIN_CONFIDENCE || 60)));
  const minFrp = Math.max(0, Number(env.FIRMS_MIN_FRP || 6));
  const vegetationOnly = String(env.FIRMS_VEGETATION_ONLY || "true").toLowerCase() !== "false";
  const bbox = buildLocationBbox(latitude, longitude, radiusMiles);

  const requests = sources.map(async (source) => {
    const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${env.FIRMS_MAP_KEY}/${source}/${bbox}/${dayRange}`;
    const response = await fetchExternal(url, {
      kind: "firms",
      parseAs: "text",
    });
    if (!response.ok) {
      return [];
    }
    const csvText = response.data;
    const rows = parseCsv(csvText);
    return rows
      .map((row) => {
        const lat = toNumber(row.latitude);
        const lon = toNumber(row.longitude);
        const confidence = parseFirmsConfidence(row);
        const fireType = parseFirmsType(row);
        const frp = parseFirmsFrp(row);
        if (lat == null || lon == null) {
          return null;
        }
        if (confidence != null && confidence < minConfidence) {
          return null;
        }
        if (vegetationOnly && fireType != null && fireType !== 0) {
          return null;
        }
        if (frp != null && frp < minFrp) {
          return null;
        }
        return {
          name: normalizeFirmsSourceLabel(source),
          latitude: lat,
          longitude: lon,
          acres_burned: 0,
          percent_contained: 0,
          source: source,
          source_label: normalizeFirmsSourceLabel(source),
          source_priority: firmsSourcePriority(source),
          confidence,
          frp,
          firms_type: fireType,
        };
      })
      .filter(Boolean);
  });

  const results = await Promise.allSettled(requests);
  return results.flatMap((result) => (result.status === "fulfilled" ? result.value : []));
}

async function fetchEonetWildfires() {
  try {
    const url = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&category=wildfires&limit=200";
    const response = await fetchExternal(url, {
      kind: "eonet",
      parseAs: "json",
    });
    if (!response.ok) {
      return [];
    }
    const payload = response.data;
    const events = Array.isArray(payload?.events) ? payload.events : [];
    return events
      .map((event) => {
        const geometries = Array.isArray(event.geometry) ? event.geometry : [];
        const latest = geometries[geometries.length - 1];
        const coordinates = latest?.coordinates;
        if (!Array.isArray(coordinates) || coordinates.length < 2) {
          return null;
        }
        const longitude = toNumber(coordinates[0]);
        const latitude = toNumber(coordinates[1]);
        if (latitude == null || longitude == null) {
          return null;
        }
        return {
          name: String(event.title || event.id || "Wildfire event"),
          latitude,
          longitude,
          acres_burned: 0,
          percent_contained: 0,
          source: "EONET",
          source_label: "NASA EONET wildfire event",
          source_priority: 0,
        };
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

async function findNearestFire(latitude, longitude, env) {
  const [eonetFires, firmsFires] = await Promise.all([
    fetchEonetWildfires(),
    fetchFirmsHotspots(latitude, longitude, env),
  ]);
  const allFires = [...firmsFires, ...eonetFires];
  if (!allFires.length) {
    return null;
  }

  const distanceLimit = maxDistanceMiles(env);
  let nearest = null;

  for (const fire of allFires) {
    const distance = haversineMiles(fire.latitude, fire.longitude, latitude, longitude);
    if (distance > distanceLimit) {
      continue;
    }
    const sourcePriority = clampNumber(toNumber(fire.source_priority) ?? firmsSourcePriority(fire.source), 0, 1);
    const confidenceBonus = clampNumber((toNumber(fire.confidence) ?? 0) / 100, 0, 1) * 0.5;
    const frpBonus = clampNumber((toNumber(fire.frp) ?? 0) / 120, 0, 1) * 0.7;
    const sourceBonusMiles = sourcePriority * 1.4;
    const signalBonusMiles = confidenceBonus + frpBonus;
    const adjustedDistance = Math.max(0, distance - sourceBonusMiles - signalBonusMiles);

    if (!nearest || adjustedDistance < nearest.adjusted_distance_miles || (Math.abs(adjustedDistance - nearest.adjusted_distance_miles) < 0.1 && distance < nearest.distance_miles)) {
      const bearingFireToWatch = calculateBearing(fire.latitude, fire.longitude, latitude, longitude);
      const bearingWatchToFire = (bearingFireToWatch + 180) % 360;
      nearest = {
        ...fire,
        source_label: fire.source_label || normalizeFirmsSourceLabel(fire.source || ""),
        source_priority: sourcePriority,
        distance_miles: distance,
        adjusted_distance_miles: Number(adjustedDistance.toFixed(2)),
        direction: `${cardinalDirection(bearingFireToWatch)} (${Math.round(bearingFireToWatch)} deg)`,
        direction_to_watch: `${cardinalDirection(bearingFireToWatch)} (${Math.round(bearingFireToWatch)} deg)`,
        fire_location_from_watch: `${cardinalDirection(bearingWatchToFire)} (${Math.round(bearingWatchToFire)} deg)`,
      };
    }
  }

  return nearest;
}

function dbLocationShape(row) {
  if (!row) {
    return null;
  }
  return {
    latitude: Number(row.latitude),
    longitude: Number(row.longitude),
    location_id: row.location_id,
    display_name: row.display_name,
    icon: row.icon,
    address: row.address,
    id: row.id,
    geocoded_at: row.geocoded_at,
    is_active: Boolean(row.is_active),
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

async function getLocation(db, locationId) {
  const row = await db
    .prepare(
      `SELECT id, location_id, display_name, icon, address, latitude, longitude, geocoded_at, is_active, created_at, updated_at
       FROM locations
       WHERE location_id = ?1 AND is_active = 1
       LIMIT 1`
    )
    .bind(locationId)
    .first();
  return dbLocationShape(row);
}

async function listLocations(db) {
  const result = await db
    .prepare(
      `SELECT id, location_id, display_name, icon, address, latitude, longitude, geocoded_at, is_active, created_at, updated_at
       FROM locations
       WHERE is_active = 1
       ORDER BY updated_at DESC, id DESC`
    )
    .all();
  const rows = Array.isArray(result?.results) ? result.results : [];
  return rows.map((row) => dbLocationShape(row)).filter(Boolean);
}

async function upsertLocation(db, payload) {
  const displayName = deriveDisplayName(payload.display_name, payload.address || payload.location_id);
  await db
    .prepare(
      `INSERT INTO locations (
         location_id, display_name, icon, address, latitude, longitude, geocoded_at, is_active, created_at, updated_at
       ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
       ON CONFLICT(location_id) DO UPDATE SET
         display_name = excluded.display_name,
         icon = excluded.icon,
         address = excluded.address,
         latitude = excluded.latitude,
         longitude = excluded.longitude,
         geocoded_at = excluded.geocoded_at,
         is_active = 1,
         updated_at = CURRENT_TIMESTAMP`
    )
    .bind(
      payload.location_id,
      displayName,
      String(payload.icon || "📍"),
      payload.persist_address ? normalizeAddress(payload.address || "") : null,
      Number(payload.latitude),
      Number(payload.longitude),
      payload.geocoded_at || new Date().toISOString()
    )
    .run();

  return getLocation(db, payload.location_id);
}

async function deleteLocation(db, locationId) {
  const result = await db
    .prepare("UPDATE locations SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE location_id = ?1 AND is_active = 1")
    .bind(locationId)
    .run();
  return (result.meta?.changes || 0) > 0;
}

async function handleEdgeApi(request, env, requestUrl) {
  const origin = request.headers.get("Origin") || "*";
  const db = env.FIREWATCH_DB;
  const path = edgePath(requestUrl.pathname);
  const method = request.method.toUpperCase();

  if (method === "GET" && path === "/health") {
    return jsonResponse({ status: db ? "healthy" : "degraded", edge: true, timestamp: new Date().toISOString() }, { origin });
  }

  if (method === "GET" && path === "/health/diagnostics") {
    const diagnostics = await runHealthDiagnostics(db, env);
    return jsonResponse(diagnostics, { origin });
  }

  if (!db) {
    return jsonResponse({ detail: "FIREWATCH_DB binding is not configured" }, { status: 500, origin });
  }

  if (method === "POST" && path === "/locations/resolve") {
    const body = await parseJsonBody(request);
    const address = normalizeAddress(body?.address);
    if (!address) {
      return jsonResponse({ detail: "Address is required" }, { status: 400, origin });
    }

    const resolved = await resolveAddress(address);
    if (!resolved) {
      return jsonResponse({ detail: "We could not resolve that location." }, { status: 422, origin });
    }
    return jsonResponse(resolved, { origin });
  }

  if (method === "POST" && path === "/locations") {
    const body = await parseJsonBody(request);
    const locationId = String(body?.location_id || "").trim();
    const latitude = toNumber(body?.latitude);
    const longitude = toNumber(body?.longitude);

    if (!locationId || latitude == null || longitude == null) {
      return jsonResponse({ detail: "location_id, latitude, and longitude are required" }, { status: 400, origin });
    }

    if (!isValidCoordinatePair(latitude, longitude)) {
      return jsonResponse(
        { detail: "latitude must be between -90 and 90, and longitude must be between -180 and 180" },
        { status: 400, origin },
      );
    }

    const saved = await upsertLocation(db, {
      location_id: locationId,
      display_name: body?.display_name,
      icon: body?.icon,
      address: body?.address,
      persist_address: Boolean(body?.persist_address),
      latitude,
      longitude,
      geocoded_at: body?.geocoded_at,
    });

    return jsonResponse(saved, { origin });
  }

  if (method === "GET" && path === "/locations") {
    const locations = await listLocations(db);
    return jsonResponse(locations, { origin });
  }

  const locationIdFromPath = extractParam(path, /^\/locations\/([^/]+)$/);
  if (locationIdFromPath && method === "GET") {
    const row = await getLocation(db, locationIdFromPath);
    if (!row) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }
    return jsonResponse(row, { origin });
  }

  if (locationIdFromPath && method === "DELETE") {
    const deleted = await deleteLocation(db, locationIdFromPath);
    if (!deleted) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }
    return jsonResponse({ message: "Location deleted successfully" }, { origin });
  }

  const assessmentLocationId = extractParam(path, /^\/alerts\/assessment\/([^/]+)$/);
  if (assessmentLocationId && method === "GET") {
    const location = await getLocation(db, assessmentLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const signals = await collectLocationSignals(location, env);
    const { nearestFire, airQuality, alertLevel, distanceMiles, direction } = signals;
    const response = {
      location_id: location.location_id,
      location_display_name: location.display_name,
      alert_level: alertLevel,
      nearest_fire_name: nearestFire ? nearestFire.name : null,
      nearest_fire_location: nearestFire
        ? {
          latitude: Number(nearestFire.latitude),
          longitude: Number(nearestFire.longitude),
          source: nearestFire.source || null,
        }
        : null,
      distance_miles: distanceMiles,
      direction,
      acres_burned: nearestFire ? Number(nearestFire.acres_burned || 0) : null,
      percent_contained: nearestFire ? Number(nearestFire.percent_contained || 0) : null,
      moving_towards: false,
      speed_mph: 0.0,
      smoke_density: null,
      smoke_distance_miles: null,
      smoke_source: null,
      smoke_observed_at: null,
      air_quality_index: airQuality?.air_quality_index ?? null,
      pm25_ug_m3: airQuality?.pm25_ug_m3 ?? null,
      air_quality_category: airQuality?.air_quality_category ?? null,
      air_quality_source: airQuality?.air_quality_source ?? null,
      air_quality_observed_at: airQuality?.air_quality_observed_at ?? null,
      recommendations: [],
    };

    if (nearestFire) {
      response.recommendations = recommendationsWithFire(
        alertLevel,
        { distance_miles: distanceMiles, direction },
        airQuality,
      );
    } else {
      response.recommendations = recommendationsWithoutFire(airQuality);
    }
    return jsonResponse(response, { origin });
  }

  const threatLocationId = extractParam(path, /^\/predictions\/threat\/([^/]+)$/);
  if (threatLocationId && method === "GET") {
    const location = await getLocation(db, threatLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const signals = await collectLocationSignals(location, env);
    const externalIntelligence = await collectExternalIntelligence(location, env);
    const prediction = buildThreatPrediction(location, signals, new Date().toISOString(), externalIntelligence);
    return jsonResponse(prediction, { origin });
  }

  const forecastLocationId = extractParam(path, /^\/predictions\/forecast\/([^/]+)$/);
  if (forecastLocationId && method === "GET") {
    const location = await getLocation(db, forecastLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const requestedHours = toNumber(requestUrl.searchParams.get("hours"));
    const horizonHours = clampNumber(Math.round(requestedHours || 24), 1, 168);
    const signals = await collectLocationSignals(location, env);
    const externalIntelligence = await collectExternalIntelligence(location, env);
    const threat = buildThreatPrediction(location, signals, new Date().toISOString(), externalIntelligence);
    const forecastPoints = threat.model_status === "active_fire_modeled"
      ? buildForecastPoints(
        threat.spread_prediction.current_distance_miles,
        threat.spread_prediction.estimated_spread_rate_mph,
        horizonHours,
        threat.threat_score,
      )
      : [];

    return jsonResponse({
      location_id: location.location_id,
      location_display_name: location.display_name,
      generated_at: threat.generated_at,
      horizon_hours: horizonHours,
      nearest_fire_name: threat.nearest_fire_name,
      nearest_fire_location: threat.nearest_fire_location,
      spread_prediction: threat.spread_prediction,
      weather_impact: threat.weather_impact,
      external_intelligence: threat.external_intelligence,
      trend: threat.forecast_summary?.trend || null,
      forecast_points: forecastPoints,
    }, { origin });
  }

  const evacuationLocationId = extractParam(path, /^\/predictions\/evacuation\/([^/]+)$/);
  if (evacuationLocationId && method === "GET") {
    const location = await getLocation(db, evacuationLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const signals = await collectLocationSignals(location, env);
    const externalIntelligence = await collectExternalIntelligence(location, env);
    const threat = buildThreatPrediction(location, signals, new Date().toISOString(), externalIntelligence);
    return jsonResponse({
      location_id: location.location_id,
      location_display_name: location.display_name,
      ...threat.evacuation_plan,
      based_on_alert_level: threat.alert_level,
      based_on_threat_score: threat.threat_score,
      external_intelligence: threat.external_intelligence,
    }, { origin });
  }

  const weatherImpactLocationId = extractParam(path, /^\/predictions\/weather-impact\/([^/]+)$/);
  if (weatherImpactLocationId && method === "GET") {
    const location = await getLocation(db, weatherImpactLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const signals = await collectLocationSignals(location, env);
    const externalIntelligence = await collectExternalIntelligence(location, env);
    const threat = buildThreatPrediction(location, signals, new Date().toISOString(), externalIntelligence);
    return jsonResponse({
      location_id: location.location_id,
      location_display_name: location.display_name,
      generated_at: threat.generated_at,
      ...threat.weather_impact,
      nearest_fire_name: threat.nearest_fire_name,
      spread_rate_mph: threat.spread_prediction?.estimated_spread_rate_mph ?? null,
      external_intelligence: threat.external_intelligence,
    }, { origin });
  }

  const anomalyLocationId = extractParam(path, /^\/predictions\/anomalies\/([^/]+)$/);
  if (anomalyLocationId && method === "GET") {
    const location = await getLocation(db, anomalyLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const signals = await collectLocationSignals(location, env);
    const externalIntelligence = await collectExternalIntelligence(location, env);
    const threat = buildThreatPrediction(location, signals, new Date().toISOString(), externalIntelligence);
    return jsonResponse({
      location_id: location.location_id,
      location_display_name: location.display_name,
      ...threat.anomaly_detection,
      external_intelligence: threat.external_intelligence,
    }, { origin });
  }

  const intelligenceLocationId = extractParam(path, /^\/intelligence\/([^/]+)$/);
  if (intelligenceLocationId && method === "GET") {
    const location = await getLocation(db, intelligenceLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const externalIntelligence = await collectExternalIntelligence(location, env);
    return jsonResponse({
      location_id: location.location_id,
      location_display_name: location.display_name,
      generated_at: new Date().toISOString(),
      ...externalIntelligence,
    }, { origin });
  }

  const provenanceAuditLocationId = extractParam(path, /^\/audit\/provenance\/([^/]+)$/);
  if (provenanceAuditLocationId && method === "GET") {
    const location = await getLocation(db, provenanceAuditLocationId);
    if (!location) {
      return jsonResponse({ detail: "Location not found" }, { status: 404, origin });
    }

    const generatedAt = new Date().toISOString();
    const [signals, externalIntelligence, backendProvenance, diagnostics] = await Promise.all([
      collectLocationSignals(location, env),
      collectExternalIntelligence(location, env),
      probeBackendModelProvenance(location.location_id, env),
      runHealthDiagnostics(db, env),
    ]);
    const prediction = buildThreatPrediction(location, signals, generatedAt, externalIntelligence);

    return jsonResponse({
      location_id: location.location_id,
      location_display_name: location.display_name,
      generated_at: generatedAt,
      edge_provenance: extractEdgeProvenance(prediction),
      backend_provenance: backendProvenance,
      diagnostics_status: diagnostics?.status || "unknown",
      diagnostics_summary: diagnostics?.summary || null,
      recommendation: backendProvenance?.status === "ok"
        ? "Edge and backend provenance are available in one call."
        : "Backend provenance unavailable. Configure PROVENANCE_BACKEND_BASE_URL or investigate backend reachability.",
    }, { origin });
  }

  return jsonResponse({ detail: "Not found" }, { status: 404, origin });
}

export default {
  async fetch(request, env) {
    const requestUrl = new URL(request.url);
    const origin = request.headers.get("Origin") || "*";

    if (request.method === "OPTIONS" && (shouldHandleEdgeApi(requestUrl.pathname) || shouldHandleDocs(requestUrl.pathname))) {
      return noContentResponse(origin);
    }

    if (shouldHandleEdgeApi(requestUrl.pathname)) {
      return handleEdgeApi(request, env, requestUrl);
    }

    if (shouldHandleDocs(requestUrl.pathname)) {
      return handleDocsRequest(requestUrl, origin);
    }

    return env.ASSETS.fetch(request);
  },
};
