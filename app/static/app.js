const STORAGE_KEY = "firewatch.watchLocations";
const ADVANCED_VIEW_STORAGE_KEY = "firewatch.advancedView";
const API_PREFIX = "/edge/v1";
const AUTO_REFRESH_INTERVAL_MS = 60000;

const state = {
  locations: loadStoredLocations(),
  assessments: new Map(),
  predictions: new Map(),
  forecasts: new Map(),
  evacuations: new Map(),
  weatherImpacts: new Map(),
  anomalies: new Map(),
  intelligence: new Map(),
  locationSnapshots: new Map(),
  locationChanges: new Map(),
  diagnostics: null,
  diagnosticsError: null,
  lastRefreshAt: null,
  nextRefreshAt: null,
  refreshTimerId: null,
  refreshCountdownId: null,
  isRefreshing: false,
  addressPreview: null,
  previewTimer: null,
  previewRequestId: 0,
  mapView: null,
  mapMarkerLayer: null,
  mapIncidentLayer: null,
  showAdvanced: loadAdvancedViewPreference(),
};

const elements = {
  form: document.getElementById("watch-form"),
  list: document.getElementById("watch-list"),
  status: document.getElementById("status"),
  clearButton: document.getElementById("clear-storage"),
  watchCount: document.getElementById("watch-count"),
  watchStatusLabel: document.getElementById("watch-status-label"),
  addressInput: document.getElementById("address"),
  displayNameInput: document.getElementById("display-name"),
  iconInput: document.getElementById("icon"),
  saveButton: document.getElementById("save-watch"),
  addressPreview: document.getElementById("address-preview"),
  mapContainer: document.getElementById("watch-map"),
  mapSummary: document.getElementById("map-summary"),
  refreshCountdown: document.getElementById("refresh-countdown"),
  refreshNow: document.getElementById("refresh-now"),
  advancedToggle: document.getElementById("advanced-toggle"),
  diagnosticsPanel: document.getElementById("diagnostics-panel"),
  diagnosticsSummary: document.getElementById("diagnostics-summary"),
  diagnosticsGrid: document.getElementById("diagnostics-grid"),
  diagnosticsRawMeta: document.getElementById("diagnostics-raw-meta"),
  diagnosticsRaw: document.getElementById("diagnostics-raw"),
};

bootstrap().catch((error) => {
  console.error(error);
  setStatus("We couldn't load your watchlist.", true);
});

async function bootstrap() {
  elements.form.addEventListener("submit", handleSaveWatch);
  elements.clearButton.addEventListener("click", handleClearStorage);
  elements.refreshNow?.addEventListener("click", handleManualRefresh);
  elements.advancedToggle?.addEventListener("click", handleAdvancedToggle);
  elements.addressInput.addEventListener("input", handleAddressInput);
  elements.addressInput.addEventListener("blur", handleAddressBlur);
  renderAdvancedMode();
  renderAddressPreview();
  renderWatchList();
  renderDiagnosticsPanel();
  initializeMap();
  await syncStoredLocations();
  await refreshDashboardData();
  startAutoRefreshLoop();
}

function loadAdvancedViewPreference() {
  try {
    return window.localStorage.getItem(ADVANCED_VIEW_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function persistAdvancedViewPreference() {
  try {
    if (state.showAdvanced) {
      window.localStorage.setItem(ADVANCED_VIEW_STORAGE_KEY, "1");
    } else {
      window.localStorage.removeItem(ADVANCED_VIEW_STORAGE_KEY);
    }
  } catch {
    // Ignore preference persistence failures.
  }
}

function renderAdvancedMode() {
  document.body.classList.toggle("advanced-mode", state.showAdvanced);
  if (!elements.advancedToggle) {
    return;
  }
  elements.advancedToggle.classList.toggle("is-active", state.showAdvanced);
  elements.advancedToggle.setAttribute("aria-pressed", state.showAdvanced ? "true" : "false");
  elements.advancedToggle.textContent = state.showAdvanced ? "Advanced view: On" : "Advanced view: Off";
}

function handleAdvancedToggle() {
  state.showAdvanced = !state.showAdvanced;
  persistAdvancedViewPreference();
  renderAdvancedMode();

  if (state.showAdvanced) {
    refreshDiagnostics().catch(() => {});
  } else {
    state.diagnostics = null;
    state.diagnosticsError = null;
    renderDiagnosticsPanel();
  }

  renderWatchList();
}

function initializeMap() {
  if (!elements.mapContainer || typeof window.L === "undefined") {
    return;
  }

  state.mapView = window.L.map(elements.mapContainer, {
    zoomControl: true,
    minZoom: 3,
  });

  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.mapView);

  state.mapMarkerLayer = window.L.layerGroup().addTo(state.mapView);
  state.mapIncidentLayer = window.L.layerGroup().addTo(state.mapView);
  state.mapView.setView([39.8283, -98.5795], 4);
  updateMapView();
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
  const explicitName = String(displayName || "").trim();
  if (explicitName) {
    return explicitName;
  }

  if (isZipCode(address)) {
    return `ZIP ${address.slice(0, 5)}`;
  }

  const firstSegment = address.split(",")[0]?.trim();
  return firstSegment || "Watched location";
}

function setFormPending(isPending, label = "Add watch") {
  elements.saveButton.disabled = isPending;
  elements.saveButton.textContent = isPending ? label : "Add watch";
}

function renderAddressPreview() {
  const preview = state.addressPreview;
  const defaultMessage = "Type a street address, place name, city, or ZIP code.";
  const message = preview?.message || defaultMessage;

  elements.addressPreview.textContent = message;
  elements.addressPreview.classList.toggle("is-loading", preview?.status === "loading");
  elements.addressPreview.classList.toggle("is-success", preview?.status === "resolved");
  elements.addressPreview.classList.toggle("is-error", preview?.status === "error");
}

function handleAddressInput() {
  const address = normalizeAddress(elements.addressInput.value);
  window.clearTimeout(state.previewTimer);

  if (!address) {
    state.addressPreview = null;
    renderAddressPreview();
    return;
  }

  if (state.addressPreview?.status === "resolved" && state.addressPreview.address === address) {
    renderAddressPreview();
    return;
  }

  if (address.length < 5) {
    state.addressPreview = {
      address,
      status: "hint",
      message: "Keep typing and we'll look it up.",
    };
    renderAddressPreview();
    return;
  }

  state.addressPreview = {
    address,
    status: "loading",
    message: isZipCode(address) ? "Checking that ZIP code..." : "Looking that place up...",
  };
  renderAddressPreview();

  state.previewTimer = window.setTimeout(() => {
    resolveAddress(address).catch(() => {});
  }, 350);
}

async function handleAddressBlur() {
  const address = normalizeAddress(elements.addressInput.value);
  elements.addressInput.value = address;

  if (address.length < 5) {
    return;
  }

  try {
    await resolveAddress(address);
  } catch {
    // The preview already contains the actionable feedback.
  }
}

async function resolveAddress(address) {
  const normalizedAddress = normalizeAddress(address);
  if (!normalizedAddress) {
    return null;
  }

  if (
    state.addressPreview?.status === "resolved"
    && state.addressPreview.address === normalizedAddress
  ) {
    return state.addressPreview;
  }

  const requestId = ++state.previewRequestId;
  state.addressPreview = {
    address: normalizedAddress,
    status: "loading",
    message: isZipCode(normalizedAddress) ? "Checking that ZIP code..." : "Looking that place up...",
  };
  renderAddressPreview();

  try {
    const preview = await postJson(`${API_PREFIX}/locations/resolve`, { address: normalizedAddress });
    if (requestId !== state.previewRequestId) {
      return null;
    }

    state.addressPreview = {
      address: normalizedAddress,
      status: "resolved",
      latitude: preview.latitude,
      longitude: preview.longitude,
      message: `Found it near ${Number(preview.latitude).toFixed(4)}, ${Number(preview.longitude).toFixed(4)}.`,
    };
    renderAddressPreview();
    return state.addressPreview;
  } catch (error) {
    if (requestId !== state.previewRequestId) {
      return null;
    }

    state.addressPreview = {
      address: normalizedAddress,
      status: "error",
      message: error.message || "We couldn't find that yet. Try a street address, city and state, landmark, or ZIP code.",
    };
    renderAddressPreview();
    throw error;
  }
}

async function getResolvedAddress(address) {
  const normalizedAddress = normalizeAddress(address);
  if (
    state.addressPreview?.status === "resolved"
    && state.addressPreview.address === normalizedAddress
  ) {
    return state.addressPreview;
  }

  const preview = await resolveAddress(normalizedAddress);
  if (preview?.latitude == null || preview?.longitude == null) {
    throw new Error("We couldn't find that yet. Try a street address, city and state, landmark, or ZIP code.");
  }
  return preview;
}

function loadStoredLocations() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveStoredLocations() {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state.locations));
}

function buildLocationId(displayName, address) {
  const base = (displayName || address)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "watch-location";
  const suffix = typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID().slice(0, 8)
    : String(Date.now()).slice(-8);
  return `${base}-${suffix}`;
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("is-error", isError);
  elements.status.classList.toggle("is-success", Boolean(message) && !isError);
}

function updateDashboardSummary() {
  if (elements.watchCount) {
    elements.watchCount.textContent = String(state.locations.length);
  }

  if (!elements.watchStatusLabel) {
    return;
  }

  const decisionStates = state.locations
    .map((location) => {
      const prediction = state.predictions.get(location.location_id);
      if (prediction?.decision_state) {
        return prediction.decision_state;
      }
      const fallbackAlert = state.assessments.get(location.location_id)?.alert_level;
      return fallbackDecisionState(fallbackAlert);
    })
    .filter(Boolean);

  if (decisionStates.some((stateValue) => stateValue === "DANGER_NOW" || stateValue === "PREPARE")) {
    elements.watchStatusLabel.textContent = "Needs attention";
    return;
  }

  if (decisionStates.some((stateValue) => stateValue === "WATCH_CLOSELY")) {
    elements.watchStatusLabel.textContent = "Monitoring";
    return;
  }

  if (decisionStates.some((stateValue) => stateValue === "SYNCING")) {
    elements.watchStatusLabel.textContent = "Syncing data";
    return;
  }

  if (state.locations.length > 0) {
    elements.watchStatusLabel.textContent = "All set";
    return;
  }

  elements.watchStatusLabel.textContent = "Ready";
}

async function handleSaveWatch(event) {
  event.preventDefault();
  const formData = new FormData(elements.form);
  const address = normalizeAddress(formData.get("address"));
  const displayName = deriveDisplayName(formData.get("display_name"), address);
  const icon = String(formData.get("icon") || "📍").trim() || "📍";

  if (!address) {
    setStatus("Add a place before saving.", true);
    return;
  }

  if (state.locations.some((location) => normalizeAddress(location.address) === address)) {
    setStatus("That place is already in your watchlist.", true);
    return;
  }

  setFormPending(true, "Saving...");
  setStatus("Checking the address and saving your watch...");

  try {
    const preview = await getResolvedAddress(address);
    const location = {
      location_id: buildLocationId(displayName, address),
      display_name: displayName,
      icon,
      address,
      latitude: preview.latitude,
      longitude: preview.longitude,
    };

    await ensureLocationOnServer(location);
    state.locations.unshift(location);
    saveStoredLocations();
    elements.form.reset();
    elements.iconInput.value = "📍";
    state.addressPreview = null;
    renderAddressPreview();
    renderWatchList();
    await refreshAssessments();
    setStatus("Saved to this browser.");
  } catch (error) {
    setStatus(error.message || "We couldn't save that watch.", true);
  } finally {
    setFormPending(false);
  }
}

async function handleClearStorage() {
  const locations = [...state.locations];
  for (const location of locations) {
    await deleteServerLocation(location.location_id);
  }
  state.locations = [];
  state.assessments.clear();
  state.predictions.clear();
  state.forecasts.clear();
  state.evacuations.clear();
  state.weatherImpacts.clear();
  state.anomalies.clear();
  state.intelligence.clear();
  state.locationSnapshots.clear();
  state.locationChanges.clear();
  state.diagnostics = null;
  state.diagnosticsError = null;
  state.lastRefreshAt = null;
  state.nextRefreshAt = null;
  saveStoredLocations();
  renderWatchList();
  renderDiagnosticsPanel();
  updateRefreshCountdown();
  setStatus("Your local watchlist is cleared.");
}

async function handleManualRefresh() {
  try {
    await refreshDashboardData();
    setStatus("Live data refreshed.");
  } catch (error) {
    setStatus(error.message || "Refresh failed. Try again in a moment.", true);
  }
}

async function syncStoredLocations() {
  for (const location of state.locations) {
    await ensureLocationOnServer(location);
  }
}

async function ensureLocationOnServer(location) {
  const existing = await fetchJson(`${API_PREFIX}/locations/${encodeURIComponent(location.location_id)}`, { allow404: true });
  if (existing) {
    return existing;
  }
  return postJson(`${API_PREFIX}/locations`, {
    location_id: location.location_id,
    display_name: location.display_name,
    icon: location.icon,
    latitude: location.latitude,
    longitude: location.longitude,
    persist_address: false,
  });
}

async function refreshDashboardData() {
  if (state.isRefreshing) {
    scheduleNextAutoRefresh();
    updateRefreshCountdown();
    return;
  }
  state.isRefreshing = true;
  try {
    const tasks = [refreshAssessments()];
    if (state.showAdvanced) {
      tasks.push(refreshDiagnostics());
    }
    await Promise.all(tasks);
  } catch (error) {
    // Keep the loop alive even if one refresh cycle fails unexpectedly.
    console.error("Dashboard refresh failed", error);
    throw error;
  } finally {
    state.isRefreshing = false;
    scheduleNextAutoRefresh();
    updateRefreshCountdown();
  }
}

function startAutoRefreshLoop() {
  stopAutoRefreshLoop();
  scheduleNextAutoRefresh();
  state.refreshCountdownId = window.setInterval(() => {
    updateRefreshCountdown();
  }, 1000);
  updateRefreshCountdown();
}

function stopAutoRefreshLoop() {
  if (state.refreshTimerId) {
    window.clearTimeout(state.refreshTimerId);
    state.refreshTimerId = null;
  }
  if (state.refreshCountdownId) {
    window.clearInterval(state.refreshCountdownId);
    state.refreshCountdownId = null;
  }
}

function scheduleNextAutoRefresh() {
  const hasWatches = state.locations.length > 0;
  if (!hasWatches) {
    state.nextRefreshAt = null;
    updateRefreshCountdown();
    return;
  }

  state.nextRefreshAt = Date.now() + AUTO_REFRESH_INTERVAL_MS;
  if (state.refreshTimerId) {
    window.clearTimeout(state.refreshTimerId);
  }
  state.refreshTimerId = window.setTimeout(async () => {
    await refreshDashboardData();
  }, AUTO_REFRESH_INTERVAL_MS);
}

function updateRefreshCountdown() {
  if (!elements.refreshCountdown) {
    return;
  }
  if (!state.locations.length) {
    elements.refreshCountdown.textContent = "Auto-refresh paused until a watch is added.";
    for (const node of document.querySelectorAll(".next-refresh-value")) {
      node.textContent = "Paused";
    }
    return;
  }
  if (!state.nextRefreshAt) {
    state.nextRefreshAt = Date.now() + AUTO_REFRESH_INTERVAL_MS;
    if (!state.refreshTimerId) {
      state.refreshTimerId = window.setTimeout(async () => {
        await refreshDashboardData();
      }, AUTO_REFRESH_INTERVAL_MS);
    }
  }
  const remainingMs = Math.max(0, state.nextRefreshAt - Date.now());
  const nextLabel = formatCountdown(remainingMs);
  elements.refreshCountdown.textContent = `Next auto-refresh in ${nextLabel}.`;
  for (const node of document.querySelectorAll(".next-refresh-value")) {
    node.textContent = nextLabel;
  }
}

async function refreshAssessments() {
  const previousSnapshots = state.locationSnapshots;
  const results = await Promise.all(
    state.locations.map(async (location) => {
      const [assessment, prediction, forecast, evacuation, weatherImpact, anomaly, intelligence] = await Promise.all([
        fetchJsonOrNull(
          `${API_PREFIX}/alerts/assessment/${encodeURIComponent(location.location_id)}`,
          { allow404: true, allow400: true },
        ),
        fetchJsonOrNull(
          `${API_PREFIX}/predictions/threat/${encodeURIComponent(location.location_id)}`,
          { allow404: true, allow400: true },
        ),
        fetchJsonOrNull(
          `${API_PREFIX}/predictions/forecast/${encodeURIComponent(location.location_id)}?hours=72`,
          { allow404: true, allow400: true },
        ),
        fetchJsonOrNull(
          `${API_PREFIX}/predictions/evacuation/${encodeURIComponent(location.location_id)}`,
          { allow404: true, allow400: true },
        ),
        fetchJsonOrNull(
          `${API_PREFIX}/predictions/weather-impact/${encodeURIComponent(location.location_id)}`,
          { allow404: true, allow400: true },
        ),
        fetchJsonOrNull(
          `${API_PREFIX}/predictions/anomalies/${encodeURIComponent(location.location_id)}`,
          { allow404: true, allow400: true },
        ),
        fetchJsonOrNull(
          `${API_PREFIX}/intelligence/${encodeURIComponent(location.location_id)}`,
          { allow404: true, allow400: true },
        ),
      ]);
      return [location.location_id, assessment, prediction, forecast, evacuation, weatherImpact, anomaly, intelligence];
    }),
  );

  state.assessments = new Map(results.map(([locationId, assessment]) => [locationId, assessment]));
  state.predictions = new Map(results.map(([locationId, , prediction]) => [locationId, prediction]));
  state.forecasts = new Map(results.map(([locationId, , , forecast]) => [locationId, forecast]));
  state.evacuations = new Map(results.map(([locationId, , , , evacuation]) => [locationId, evacuation]));
  state.weatherImpacts = new Map(results.map(([locationId, , , , , weatherImpact]) => [locationId, weatherImpact]));
  state.anomalies = new Map(results.map(([locationId, , , , , , anomaly]) => [locationId, anomaly]));
  state.intelligence = new Map(results.map(([locationId, , , , , , , intelligence]) => [locationId, intelligence]));
  const nextSnapshots = new Map();
  const nextChanges = new Map();
  const fetchedAt = new Date().toISOString();
  for (const [locationId, assessment, prediction] of results) {
    const current = buildLocationSnapshot({ assessment, prediction, fetchedAt });
    const previous = previousSnapshots.get(locationId) || null;
    nextSnapshots.set(locationId, current);
    nextChanges.set(locationId, buildLocationChange(previous, current));
  }
  state.locationSnapshots = nextSnapshots;
  state.locationChanges = nextChanges;
  state.lastRefreshAt = fetchedAt;
  renderWatchList();
}

async function refreshDiagnostics() {
  if (!state.showAdvanced) {
    state.diagnostics = null;
    state.diagnosticsError = null;
    renderDiagnosticsPanel();
    return;
  }

  try {
    const diagnostics = await fetchJson(`${API_PREFIX}/health/diagnostics`);
    state.diagnostics = diagnostics;
    state.diagnosticsError = null;
  } catch (error) {
    state.diagnostics = null;
    state.diagnosticsError = error.message || "Diagnostics unavailable";
  }
  renderDiagnosticsPanel();
}

async function deleteServerLocation(locationId) {
  const response = await fetch(`${API_PREFIX}/locations/${encodeURIComponent(locationId)}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 404) {
    throw new Error("We couldn't remove that watch from the backend.");
  }
}

function renderWatchList() {
  updateDashboardSummary();
  updateMapView();

  if (state.locations.length === 0) {
    elements.list.innerHTML = `
      <article class="empty-state">
        <p class="panel-label">No saved places yet</p>
        <h3>Add a place to start keeping an eye on nearby fires.</h3>
        <p class="address">The full address stays in this browser, so it is still here when you come back.</p>
      </article>
    `;
    return;
  }

  elements.list.innerHTML = state.locations.map((location) => {
    const showAdvanced = state.showAdvanced;
    const assessment = state.assessments.get(location.location_id);
    const prediction = state.predictions.get(location.location_id);
    const forecast = state.forecasts.get(location.location_id);
    const evacuation = state.evacuations.get(location.location_id) || prediction?.evacuation_plan;
    const weatherImpact = state.weatherImpacts.get(location.location_id) || prediction?.weather_impact;
    const anomaly = state.anomalies.get(location.location_id) || prediction?.anomaly_detection;
    const intelligence = state.intelligence.get(location.location_id) || prediction?.external_intelligence;
    const alertLevel = assessment?.alert_level || "SYNCING";
    const nearestFire = assessment?.nearest_fire_name || "No nearby fires detected";
    const distance = assessment?.distance_miles != null ? `${assessment.distance_miles.toFixed(1)} miles` : "No active threat";
    const smoke = assessment?.smoke_density
      ? `${assessment.smoke_density} smoke${assessment?.smoke_distance_miles != null ? ` • ${assessment.smoke_distance_miles.toFixed(1)} miles` : ""}`
      : null;
    const airQuality = assessment?.air_quality_index != null
      ? `${assessment.air_quality_category || "Air quality"} • US AQI ${assessment.air_quality_index}${assessment?.pm25_ug_m3 != null ? ` • PM2.5 ${assessment.pm25_ug_m3.toFixed(1)}` : ""}`
      : null;
    const exposureGuidance = (assessment?.recommendations || [])
      .filter((item) => /smoke|air quality|outdoor exposure|pm2\.5|aqi/i.test(item))
      .slice(0, 3);
    const modelStatus = prediction?.model_status || null;
    const activeFireModeled = modelStatus === "active_fire_modeled";
    const threatScore = prediction?.threat_score;
    const riskLevel = prediction?.risk_level || null;
    const impactAssessment = prediction?.impact_assessment || null;
    const riskToWatchLevel = impactAssessment?.risk_to_watch_level || null;
    const likely24h = impactAssessment?.likely_impact_within_24h;
    const likely72h = impactAssessment?.likely_impact_within_72h;
    const impactProbability24 = toFiniteNumber(impactAssessment?.impact_probability_24h_pct);
    const impactProbability72 = toFiniteNumber(impactAssessment?.impact_probability_72h_pct);
    const impactDriverBreakdown = Array.isArray(impactAssessment?.impact_driver_breakdown)
      ? impactAssessment.impact_driver_breakdown.slice(0, 12)
      : [];
    const topImpactDrivers = Array.isArray(impactAssessment?.top_impact_drivers)
      ? impactAssessment.top_impact_drivers.slice(0, 3)
      : [];
    const containmentUsed = toFiniteNumber(impactAssessment?.containment_pct_used);
    const containmentSource = impactAssessment?.containment_source || null;
    const satelliteSignal = prediction?.satellite_signal || null;
    const satelliteSourceId = satelliteSignal?.source_id || prediction?.nearest_fire_location?.source || null;
    const satelliteSourceLabel = satelliteSignal?.source_label || null;
    const satelliteConfidence = toFiniteNumber(satelliteSignal?.confidence_pct);
    const satelliteFrp = toFiniteNumber(satelliteSignal?.frp);
    const directionToWatch = impactAssessment?.direction_to_watch || prediction?.spread_prediction?.estimated_direction || null;
    const fireLocationFromWatch = impactAssessment?.fire_location_from_watch || oppositeDirection(directionToWatch);
    const impactSummary = impactAssessment?.summary || null;
    const impactNote = impactAssessment?.note || null;
    const spreadRate = prediction?.spread_prediction?.estimated_spread_rate_mph;
    const etaToFive = prediction?.spread_prediction?.eta_to_5_miles_hours;
    const etaToOne = prediction?.spread_prediction?.eta_to_1_mile_hours;
    const forecastTrend = prediction?.forecast_summary?.trend || null;
    const forecastMax = prediction?.forecast_summary?.max_projected_threat_score;
    const forecastPoints = Array.isArray(forecast?.forecast_points) ? forecast.forecast_points.slice(0, 4) : [];
    const weatherDanger = weatherImpact?.fire_danger_level || null;
    const spreadMultiplier = weatherImpact?.spread_multiplier;
    const anomalyScore = anomaly?.anomaly_score;
    const anomalyReasons = Array.isArray(anomaly?.reasons) ? anomaly.reasons.slice(0, 2) : [];
    const evacuationUrgency = evacuation?.urgency || null;
    const evacuationWindow = evacuation?.recommended_action_window_hours;
    const evacuationActions = Array.isArray(evacuation?.actions) ? evacuation.actions.slice(0, 3) : [];
    const modifiers = Array.isArray(prediction?.external_risk_modifiers) ? prediction.external_risk_modifiers.slice(0, 2) : [];
    const nwsAlertCount = intelligence?.nws_alerts?.alert_count;
    const redFlag = Boolean(intelligence?.nws_alerts?.has_red_flag_warning);
    const windMax = intelligence?.hrrr_wind_proxy?.max_wind_mph;
    const incidentName = intelligence?.incident_perimeters?.nearest_incident?.name;
    const plumeDensity = intelligence?.hms_smoke?.nearest_plume?.density;
    const purpleAirPm = intelligence?.purpleair?.nearest_sensor?.pm25_ug_m3;
    const closureCount = intelligence?.road_closures?.closure_count_nearby;
    const hasPsps = Boolean(intelligence?.utility_status?.has_psps_nearby);
    const lightningRisk = Boolean(intelligence?.lightning_nowcast?.elevated_ignition_risk);
    const terrainSlope = intelligence?.terrain_context?.slope_degrees;
    const decisionState = prediction?.decision_state || fallbackDecisionState(alertLevel);
    const decisionSummary = prediction?.decision_summary || null;
    const decisionClass = decisionState ? decisionState.toLowerCase().replaceAll("_", "-") : "syncing";
    const inferenceMode = prediction?.inference_mode || null;
    const predictionEngineKind = prediction?.prediction_engine?.kind || null;
    const predictionEngineModel = prediction?.prediction_engine?.model_name || null;
    const predictionEngineNote = prediction?.prediction_engine?.notes || null;
    const coordinates = `${Number(location.latitude).toFixed(4)}, ${Number(location.longitude).toFixed(4)}`;
    const snapshot = state.locationSnapshots.get(location.location_id) || null;
    const change = state.locationChanges.get(location.location_id) || null;
    const freshness = freshnessStateFor(snapshot?.updated_at);

    return `
      <article class="watch-card" data-location-id="${location.location_id}">
        <div class="watch-card-topline">
          <div class="decision-pill ${decisionClass}">${escapeHtml(formatDecisionState(decisionState))}</div>
          <span class="watch-card-coordinates">${coordinates}</span>
        </div>
        <section class="watch-update-strip" aria-label="Data freshness">
          <p class="watch-update-copy"><strong>Last updated:</strong> ${escapeHtml(formatLastUpdated(snapshot?.updated_at))}</p>
          <p class="watch-freshness ${escapeHtml(freshness.className)}">${escapeHtml(freshness.message)}</p>
          <p class="watch-update-copy"><strong>Next auto-refresh:</strong> <span class="next-refresh-value">${escapeHtml(formatNextRefreshLabel())}</span></p>
        </section>
        <section class="watch-change-strip" aria-label="What changed since last refresh">
          <p class="watch-change-title">What changed since last refresh</p>
          <div class="watch-change-grid">
            <p class="watch-change-item"><span>Distance</span><strong>${escapeHtml(formatDistanceDelta(change))}</strong></p>
            <p class="watch-change-item"><span>Threat</span><strong>${escapeHtml(formatThreatDelta(change))}</strong></p>
            <p class="watch-change-item"><span>State</span><strong>${escapeHtml(formatStateDelta(change))}</strong></p>
          </div>
        </section>
        ${decisionSummary ? `<p class="decision-summary">${escapeHtml(decisionSummary)}</p>` : ""}
        <header class="watch-card-header">
          <div>
            <h3>${escapeHtml(location.icon)} ${escapeHtml(location.display_name)}</h3>
            <p class="address">${escapeHtml(location.address)}</p>
          </div>
          <button class="remove-button" type="button" data-action="remove">Remove</button>
        </header>
        <div class="watch-meta">
          <div class="meta-block">
            <span class="meta-label">Nearest fire</span>
            <strong>${escapeHtml(nearestFire)}</strong>
          </div>
          <div class="meta-block">
            <span class="meta-label">Distance</span>
            <strong>${escapeHtml(distance)}</strong>
          </div>
          ${satelliteSourceId ? `
          <div class="meta-block">
            <span class="meta-label">Satellite source</span>
            <strong>${escapeHtml(satelliteSourceId)}</strong>
            <small class="field-note">${escapeHtml(satelliteSourceLabel || "Thermal anomaly signal")}${satelliteConfidence != null ? ` • confidence ${Math.round(satelliteConfidence)}%` : ""}${satelliteFrp != null ? ` • FRP ${satelliteFrp.toFixed(1)}` : ""}</small>
          </div>
          ` : ""}
          ${smoke ? `
          <div class="meta-block">
            <span class="meta-label">Smoke</span>
            <strong>${escapeHtml(smoke)}</strong>
          </div>
          ` : ""}
          ${airQuality ? `
          <div class="meta-block">
            <span class="meta-label">Air quality</span>
            <strong>${escapeHtml(airQuality)}</strong>
          </div>
          ` : ""}
        </div>
        ${exposureGuidance.length ? `
        <section class="smoke-advisory" aria-label="Exposure guidance">
          <p class="smoke-advisory-title">Smoke and air guidance</p>
          <ul class="smoke-advisory-list">
            ${exposureGuidance.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
          </ul>
        </section>
        ` : ""}
        ${prediction ? `
        <section class="prediction-panel" aria-label="Fire spread prediction">
          <p class="prediction-title">Spread prediction</p>
          ${showAdvanced && (inferenceMode || predictionEngineKind) ? `
          <p class="prediction-note">
            <strong>Inference mode:</strong> ${escapeHtml(inferenceMode || "unknown")}${predictionEngineKind ? ` • ${escapeHtml(predictionEngineKind)}` : ""}${predictionEngineModel ? ` • ${escapeHtml(predictionEngineModel)}` : ""}
          </p>
          ` : ""}
          ${showAdvanced && predictionEngineNote ? `<p class="prediction-note">${escapeHtml(predictionEngineNote)}</p>` : ""}
          <div class="prediction-grid">
            <div class="prediction-item">
              <span class="prediction-label">Risk to this address</span>
              <strong>${threatScore != null ? `${threatScore}/100` : "Unknown"}${riskToWatchLevel ? ` • ${escapeHtml(riskToWatchLevel)}` : riskLevel ? ` • ${escapeHtml(riskLevel)}` : ""}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">Estimated spread speed</span>
              <strong>${activeFireModeled && spreadRate != null ? `${Number(spreadRate).toFixed(2)} mph` : "Not applicable"}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">ETA to 5 miles</span>
              <strong>${activeFireModeled ? formatEtaHours(etaToFive) : "Not applicable"}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">ETA to 1 mile</span>
              <strong>${activeFireModeled ? formatEtaHours(etaToOne) : "Not applicable"}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">Impact probability (24h)</span>
              <strong>${activeFireModeled && impactProbability24 != null ? `${Math.round(impactProbability24)}%` : "Not applicable"}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">Impact probability (72h)</span>
              <strong>${activeFireModeled && impactProbability72 != null ? `${Math.round(impactProbability72)}%` : "Not applicable"}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">Fire location relative to you</span>
              <strong>${activeFireModeled && fireLocationFromWatch ? escapeHtml(fireLocationFromWatch) : "Not applicable"}</strong>
            </div>
            ${showAdvanced ? `
            <div class="prediction-item">
              <span class="prediction-label">Watch location from fire</span>
              <strong>${activeFireModeled && directionToWatch ? escapeHtml(directionToWatch) : "Not applicable"}</strong>
            </div>
            <div class="prediction-item">
              <span class="prediction-label">Containment used</span>
              <strong>${activeFireModeled && containmentUsed != null ? `${containmentUsed.toFixed(1)}%${containmentSource ? ` • ${escapeHtml(containmentSource.replaceAll("_", " "))}` : ""}` : "Not applicable"}</strong>
            </div>
            ` : ""}
          </div>
          ${impactSummary ? `<p class="prediction-trend"><strong>Impact summary:</strong> ${escapeHtml(impactSummary)}</p>` : ""}
          ${(impactProbability24 != null || impactProbability72 != null)
    ? `<p class="prediction-trend">Modeled impact probability: <strong>24h ${impactProbability24 != null ? `${Math.round(impactProbability24)}%` : "n/a"}</strong> • <strong>72h ${impactProbability72 != null ? `${Math.round(impactProbability72)}%` : "n/a"}</strong></p>`
    : ""}
          ${(likely24h != null || likely72h != null) ? `<p class="prediction-trend">Impact likelihood: <strong>24h ${likely24h ? "POSSIBLE" : "LOW"}</strong> • <strong>72h ${likely72h ? "POSSIBLE" : "LOW"}</strong></p>` : ""}
          ${forecastTrend && forecastTrend !== "NO_ACTIVE_FIRE" ? `<p class="prediction-trend">24h trend: <strong>${escapeHtml(forecastTrend)}</strong>${forecastMax != null ? ` • peak threat ${forecastMax}/100` : ""}</p>` : ""}
          ${forecastTrend === "NO_ACTIVE_FIRE" ? `<p class="prediction-trend">24h trend: <strong>No active nearby wildfire</strong></p>` : ""}
          ${showAdvanced && topImpactDrivers.length ? `<p class="prediction-note"><strong>Main drivers:</strong> ${escapeHtml(topImpactDrivers.join(" • "))}</p>` : ""}
          ${impactDriverBreakdown.length ? `
          <details class="why-score-panel" ${showAdvanced ? "open" : ""}>
            <summary>Why this score</summary>
            <p class="why-score-caption">Driver contributions to modeled impact probability for this watch.</p>
            <ul class="why-score-list">
              ${impactDriverBreakdown.map((driver) => {
    const c24 = toFiniteNumber(driver?.contribution_24h_pct);
    const c72 = toFiniteNumber(driver?.contribution_72h_pct);
    return `<li class="why-score-item">
                  <div class="why-score-row">
                    <strong>${escapeHtml(driver?.label || "Driver")}</strong>
                    <span>24h ${c24 != null ? `${c24.toFixed(1)}%` : "n/a"} • 72h ${c72 != null ? `${c72.toFixed(1)}%` : "n/a"}</span>
                  </div>
                  <p>${escapeHtml(driver?.detail || "")}</p>
                </li>`;
  }).join("")}
            </ul>
          </details>
          ` : ""}
          ${showAdvanced && impactNote ? `<p class="prediction-note">${escapeHtml(impactNote)}</p>` : ""}
        </section>
        ` : ""}
        ${forecastPoints.length ? `
        <section class="insight-panel" aria-label="Forecast timeline">
          <p class="insight-title">Forecast timeline (72h)</p>
          <div class="timeline-grid">
            ${forecastPoints.map((point) => `
              <div class="timeline-item">
                <span class="timeline-hour">+${escapeHtml(point.hour)}h</span>
                <strong>${point.projected_distance_miles != null ? `${Number(point.projected_distance_miles).toFixed(1)} mi` : "n/a"}</strong>
                <span class="timeline-threat">Threat ${escapeHtml(point.projected_threat_score)}/100</span>
              </div>
            `).join("")}
          </div>
        </section>
        ` : ""}
        ${evacuationUrgency ? `
        <section class="insight-panel" aria-label="Evacuation guidance">
          <p class="insight-title">Evacuation guidance</p>
          <p class="insight-row"><strong>${escapeHtml(evacuationUrgency)}</strong>${evacuationWindow != null ? ` • action window ${formatEtaHours(evacuationWindow)}` : ""}</p>
          ${evacuationActions.length ? `
          <ul class="insight-list">
            ${evacuationActions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
          </ul>
          ` : ""}
        </section>
        ` : ""}
        ${(weatherImpact || anomaly) ? `
        <section class="insight-panel" aria-label="Weather and anomaly insights">
          <p class="insight-title">Conditions and trend alerts</p>
          ${weatherImpact ? `
          <p class="insight-row">Weather danger: <strong>${escapeHtml(weatherDanger || "Unknown")}</strong>${showAdvanced && spreadMultiplier != null ? ` • spread multiplier ${Number(spreadMultiplier).toFixed(2)}x` : ""}</p>
          ` : ""}
          ${anomaly ? `
          <p class="insight-row">Trend alert: <strong>${anomaly?.is_anomalous ? "FLAGGED" : "Normal"}</strong>${showAdvanced && anomalyScore != null ? ` • score ${anomalyScore}/100` : ""}</p>
          ${showAdvanced && anomalyReasons.length ? `<p class="insight-row">${escapeHtml(anomalyReasons.join(" • "))}</p>` : ""}
          ` : ""}
          ${showAdvanced && modifiers.length ? `<p class="insight-row">External modifiers: ${escapeHtml(modifiers.join(" • "))}</p>` : ""}
        </section>
        ` : ""}
        ${showAdvanced && intelligence ? `
        <section class="insight-panel" aria-label="External intelligence feeds">
          <p class="insight-title">External intelligence</p>
          <p class="insight-row">NWS alerts: <strong>${nwsAlertCount != null ? escapeHtml(nwsAlertCount) : "n/a"}</strong>${redFlag ? " • Red Flag Warning nearby" : ""}</p>
          <p class="insight-row">NWS wind (HRRR proxy): <strong>${windMax != null ? `${Number(windMax).toFixed(1)} mph max` : "n/a"}</strong></p>
          <p class="insight-row">Incident perimeter: <strong>${escapeHtml(incidentName || "n/a")}</strong></p>
          <p class="insight-row">HMS smoke plume: <strong>${escapeHtml(plumeDensity || "n/a")}</strong></p>
          <p class="insight-row">PurpleAir PM2.5: <strong>${purpleAirPm != null ? `${Number(purpleAirPm).toFixed(1)} ug/m3` : "n/a"}</strong></p>
          <p class="insight-row">Road closures nearby: <strong>${closureCount != null ? escapeHtml(closureCount) : "n/a"}</strong></p>
          <p class="insight-row">Utility PSPS: <strong>${hasPsps ? "Possible" : "None reported"}</strong></p>
          <p class="insight-row">Lightning nowcast risk: <strong>${lightningRisk ? "Elevated" : "Low/none"}</strong></p>
          <p class="insight-row">Terrain slope: <strong>${terrainSlope != null ? `${Number(terrainSlope).toFixed(1)} deg` : "n/a"}</strong></p>
        </section>
        ` : ""}
      </article>
    `;
  }).join("");

  for (const button of elements.list.querySelectorAll('[data-action="remove"]')) {
    button.addEventListener("click", async (event) => {
      const card = event.target.closest("[data-location-id]");
      const locationId = card?.dataset.locationId;
      if (!locationId) {
        return;
      }
      try {
        await deleteServerLocation(locationId);
        state.locations = state.locations.filter((location) => location.location_id !== locationId);
        state.assessments.delete(locationId);
        state.predictions.delete(locationId);
        state.forecasts.delete(locationId);
        state.evacuations.delete(locationId);
        state.weatherImpacts.delete(locationId);
        state.anomalies.delete(locationId);
        state.intelligence.delete(locationId);
        state.locationSnapshots.delete(locationId);
        state.locationChanges.delete(locationId);
        saveStoredLocations();
        renderWatchList();
        scheduleNextAutoRefresh();
        updateRefreshCountdown();
        setStatus("Watch removed.");
      } catch (error) {
        setStatus(error.message || "We couldn't remove that watch.", true);
      }
    });
  }

  if (state.showAdvanced) {
    renderDeveloperDiagnosticsPayload();
  }
}

function renderDiagnosticsPanel() {
  if (!elements.diagnosticsSummary || !elements.diagnosticsGrid) {
    return;
  }

  if (state.diagnosticsError) {
    elements.diagnosticsSummary.textContent = `Diagnostics unavailable: ${state.diagnosticsError}`;
    elements.diagnosticsGrid.innerHTML = "";
  } else if (!state.diagnostics) {
    elements.diagnosticsSummary.textContent = "Waiting for diagnostics...";
    elements.diagnosticsGrid.innerHTML = "";
  } else {
    const diagnosticsStatus = String(state.diagnostics.status || "unknown").toUpperCase();
    const summary = state.diagnostics.summary || {};
    const lastUpdated = formatLastUpdated(state.diagnostics.timestamp);
    elements.diagnosticsSummary.textContent = `Overall ${diagnosticsStatus}. Last updated ${lastUpdated}. Ok: ${summary.ok_count || 0}, degraded: ${summary.degraded_count || 0}, not configured: ${summary.not_configured_count || 0}, down: ${summary.down_count || 0}.`;

    const dependencies = state.diagnostics.dependencies || {};
    const dependencyOrder = ["d1", "eonet", "firms", "aqi"];
    elements.diagnosticsGrid.innerHTML = dependencyOrder.map((key) => {
      const item = dependencies[key] || {};
      const status = String(item.status || "unknown");
      const statusClass = status.toLowerCase().replaceAll("_", "-");
      const latency = toFiniteNumber(item.latency_ms);
      const reason = item.reason || item.error || "No additional details";
      return `
        <article class="diagnostic-card">
          <h3>${escapeHtml(key)}</h3>
          <p class="diagnostic-status ${escapeHtml(statusClass)}">${escapeHtml(status.toUpperCase())}</p>
          <p class="diagnostic-meta">Latency: ${escapeHtml(latency != null ? `${Math.round(latency)} ms` : "n/a")}</p>
          <p class="diagnostic-meta">${escapeHtml(reason)}</p>
        </article>
      `;
    }).join("");
  }

  renderDeveloperDiagnosticsPayload();
}

function renderDeveloperDiagnosticsPayload() {
  if (!elements.diagnosticsRaw) {
    return;
  }

  const locationPayloads = state.locations.map((location) => {
    const locationId = location.location_id;
    return {
      location,
      assessment: state.assessments.get(locationId) || null,
      prediction: state.predictions.get(locationId) || null,
      forecast: state.forecasts.get(locationId) || null,
      evacuation: state.evacuations.get(locationId) || null,
      weather_impact: state.weatherImpacts.get(locationId) || null,
      anomaly_detection: state.anomalies.get(locationId) || null,
      external_intelligence: state.intelligence.get(locationId) || null,
      snapshot: state.locationSnapshots.get(locationId) || null,
      change: state.locationChanges.get(locationId) || null,
    };
  });

  const payload = {
    generated_at: new Date().toISOString(),
    diagnostics: state.diagnostics,
    diagnostics_error: state.diagnosticsError,
    watch_count: state.locations.length,
    watches: locationPayloads,
  };

  if (elements.diagnosticsRawMeta) {
    const sizeBytes = new TextEncoder().encode(JSON.stringify(payload)).length;
    elements.diagnosticsRawMeta.textContent = `Generated ${formatLastUpdated(payload.generated_at)} • ${state.locations.length} watch${state.locations.length === 1 ? "" : "es"} • ${Math.round(sizeBytes / 1024)} KB`;
  }

  elements.diagnosticsRaw.textContent = JSON.stringify(payload, null, 2);
}

function createMapPinIcon(decisionClass) {
  if (typeof window.L === "undefined") {
    return null;
  }

  const allowed = new Set(["safe", "watch-closely", "prepare", "danger-now", "syncing"]);
  const normalized = allowed.has(decisionClass) ? decisionClass : "syncing";
  return window.L.divIcon({
    className: `map-pin map-pin-${normalized}`,
    html: "<span></span>",
    iconSize: [18, 18],
    iconAnchor: [9, 9],
    popupAnchor: [0, -8],
  });
}

function createFireIcon() {
  if (typeof window.L === "undefined") {
    return null;
  }

  return window.L.divIcon({
    className: "map-fire-icon",
    html: "<span class=\"map-fire-core\"></span><span class=\"map-fire-pulse\"></span>",
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -10],
  });
}

function parseBearingDegrees(directionLabel) {
  const text = String(directionLabel || "").trim();
  if (!text) {
    return null;
  }
  const match = text.match(/\(([-+]?\d+(?:\.\d+)?)\s*deg\)/i);
  if (!match) {
    return null;
  }
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function clampValue(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toRadians(value) {
  return (value * Math.PI) / 180;
}

function toDegrees(value) {
  return (value * 180) / Math.PI;
}

function haversineMilesBetween(startPoint, endPoint) {
  const earthRadiusMiles = 3958.7613;
  const lat1 = toRadians(startPoint.latitude);
  const lat2 = toRadians(endPoint.latitude);
  const latDelta = toRadians(endPoint.latitude - startPoint.latitude);
  const lonDelta = toRadians(endPoint.longitude - startPoint.longitude);
  const a = Math.sin(latDelta / 2) ** 2
    + Math.cos(lat1) * Math.cos(lat2) * Math.sin(lonDelta / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return earthRadiusMiles * c;
}

function destinationPointFromBearing(latitude, longitude, bearingDegrees, distanceMiles) {
  const earthRadiusMiles = 3958.7613;
  const angularDistance = distanceMiles / earthRadiusMiles;
  const bearing = toRadians(bearingDegrees);
  const lat1 = toRadians(latitude);
  const lon1 = toRadians(longitude);

  const sinLat1 = Math.sin(lat1);
  const cosLat1 = Math.cos(lat1);
  const sinAngular = Math.sin(angularDistance);
  const cosAngular = Math.cos(angularDistance);

  const lat2 = Math.asin(
    sinLat1 * cosAngular + cosLat1 * sinAngular * Math.cos(bearing),
  );

  const lon2 = lon1 + Math.atan2(
    Math.sin(bearing) * sinAngular * cosLat1,
    cosAngular - (sinLat1 * Math.sin(lat2)),
  );

  return {
    latitude: toDegrees(lat2),
    longitude: toDegrees(lon2),
  };
}

function interpolateLatLng(startPoint, endPoint, fraction) {
  const ratio = clampValue(Number(fraction) || 0, 0, 1);
  return {
    latitude: startPoint.latitude + ((endPoint.latitude - startPoint.latitude) * ratio),
    longitude: startPoint.longitude + ((endPoint.longitude - startPoint.longitude) * ratio),
  };
}

function deriveNearestFirePoint(location, assessment, prediction) {
  const byPrediction = prediction?.nearest_fire_location;
  if (Number.isFinite(byPrediction?.latitude) && Number.isFinite(byPrediction?.longitude)) {
    return {
      latitude: Number(byPrediction.latitude),
      longitude: Number(byPrediction.longitude),
      source: byPrediction.source || null,
    };
  }

  const byAssessment = assessment?.nearest_fire_location;
  if (Number.isFinite(byAssessment?.latitude) && Number.isFinite(byAssessment?.longitude)) {
    return {
      latitude: Number(byAssessment.latitude),
      longitude: Number(byAssessment.longitude),
      source: byAssessment.source || null,
    };
  }

  const watchLatitude = Number(location.latitude);
  const watchLongitude = Number(location.longitude);
  if (!Number.isFinite(watchLatitude) || !Number.isFinite(watchLongitude)) {
    return null;
  }

  const distanceMiles = toFiniteNumber(
    prediction?.spread_prediction?.current_distance_miles ?? assessment?.distance_miles,
  );
  if (distanceMiles == null) {
    return null;
  }

  const watchToFireDirection = prediction?.impact_assessment?.fire_location_from_watch
    || oppositeDirection(
      prediction?.impact_assessment?.direction_to_watch
      || prediction?.spread_prediction?.estimated_direction
      || assessment?.direction,
    );
  const bearing = parseBearingDegrees(watchToFireDirection);
  if (bearing == null) {
    return null;
  }

  return destinationPointFromBearing(watchLatitude, watchLongitude, bearing, distanceMiles);
}

function updateMapView() {
  if (!state.mapView || !state.mapMarkerLayer || !state.mapIncidentLayer || typeof window.L === "undefined") {
    return;
  }

  state.mapMarkerLayer.clearLayers();
  state.mapIncidentLayer.clearLayers();

  const bounds = [];
  const fireKeys = new Set();
  const counts = {
    SYNCING: 0,
    SAFE: 0,
    WATCH_CLOSELY: 0,
    PREPARE: 0,
    DANGER_NOW: 0,
  };
  let visualizedFires = 0;
  let spreadTracks = 0;
  let forecastNodes = 0;

  for (const location of state.locations) {
    const latitude = Number(location.latitude);
    const longitude = Number(location.longitude);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      continue;
    }

    const prediction = state.predictions.get(location.location_id);
    const assessment = state.assessments.get(location.location_id);
  const forecast = state.forecasts.get(location.location_id);
    const alertLevel = assessment?.alert_level;
    const decisionState = prediction?.decision_state || fallbackDecisionState(alertLevel);
    counts[decisionState] = (counts[decisionState] || 0) + 1;
    const decisionClass = decisionState.toLowerCase().replaceAll("_", "-");
    const nearestFireName = assessment?.nearest_fire_name || "No nearby fires detected";
    const distanceCopy = assessment?.distance_miles != null
      ? `${assessment.distance_miles.toFixed(1)} miles`
      : "No active threat";
    const impactProbability24 = toFiniteNumber(prediction?.impact_assessment?.impact_probability_24h_pct);
    const impactProbability72 = toFiniteNumber(prediction?.impact_assessment?.impact_probability_72h_pct);
    const snapshot = state.locationSnapshots.get(location.location_id) || null;
    const freshness = freshnessStateFor(snapshot?.updated_at);

    const marker = window.L.marker([latitude, longitude], {
      icon: createMapPinIcon(decisionClass),
    });

    marker.bindPopup(`
      <p class="map-popup-title"><strong>${escapeHtml(location.icon)} ${escapeHtml(location.display_name)}</strong></p>
      <p class="map-popup-copy">${escapeHtml(formatDecisionState(decisionState))}</p>
      <p class="map-popup-copy">Nearest fire: ${escapeHtml(nearestFireName)}</p>
      <p class="map-popup-copy">Distance: ${escapeHtml(distanceCopy)}</p>
      ${(impactProbability24 != null || impactProbability72 != null)
        ? `<p class="map-popup-copy">Impact probability: 24h ${escapeHtml(impactProbability24 != null ? `${Math.round(impactProbability24)}%` : "n/a")} • 72h ${escapeHtml(impactProbability72 != null ? `${Math.round(impactProbability72)}%` : "n/a")}</p>`
        : ""}
      <p class="map-popup-copy">Last updated: ${escapeHtml(formatLastUpdated(snapshot?.updated_at))}</p>
      <p class="map-popup-copy">${escapeHtml(freshness.message)}</p>
    `);
    marker.addTo(state.mapMarkerLayer);
    bounds.push([latitude, longitude]);

    const firePoint = deriveNearestFirePoint(location, assessment, prediction);
    if (!firePoint) {
      continue;
    }

    const fireLabel = prediction?.nearest_fire_name || assessment?.nearest_fire_name || "Nearby fire";
    const satelliteSignal = prediction?.satellite_signal || null;
    const satelliteSourceId = satelliteSignal?.source_id || firePoint.source || null;
    const satelliteSourceLabel = satelliteSignal?.source_label || null;
    const satelliteConfidence = toFiniteNumber(satelliteSignal?.confidence_pct);
    const satelliteFrp = toFiniteNumber(satelliteSignal?.frp);
    const fireKey = `${firePoint.latitude.toFixed(3)}:${firePoint.longitude.toFixed(3)}:${fireLabel}`;
    if (!fireKeys.has(fireKey)) {
      fireKeys.add(fireKey);

      const fireMarker = window.L.marker([firePoint.latitude, firePoint.longitude], {
        icon: createFireIcon(),
      });
      fireMarker.bindPopup(`
        <p class="map-popup-title"><strong>🔥 ${escapeHtml(fireLabel)}</strong></p>
        <p class="map-popup-copy">Source ID: ${escapeHtml(satelliteSourceId || "Derived incident location")}</p>
        ${satelliteSourceLabel ? `<p class="map-popup-copy">Signal: ${escapeHtml(satelliteSourceLabel)}</p>` : ""}
        ${(satelliteConfidence != null || satelliteFrp != null)
          ? `<p class="map-popup-copy">${satelliteConfidence != null ? `Confidence ${Math.round(satelliteConfidence)}%` : ""}${satelliteConfidence != null && satelliteFrp != null ? " • " : ""}${satelliteFrp != null ? `FRP ${satelliteFrp.toFixed(1)}` : ""}</p>`
          : ""}
      `);
      fireMarker.addTo(state.mapIncidentLayer);

      window.L.circle([firePoint.latitude, firePoint.longitude], {
        radius: 1800,
        color: "#d93025",
        weight: 1,
        fillColor: "#d93025",
        fillOpacity: 0.08,
        className: "spread-pulse-ring",
      }).addTo(state.mapIncidentLayer);

      window.L.circle([firePoint.latitude, firePoint.longitude], {
        radius: 3200,
        color: "#e65c00",
        weight: 1,
        fillColor: "#e65c00",
        fillOpacity: 0.06,
        className: "spread-pulse-ring delayed",
      }).addTo(state.mapIncidentLayer);

      bounds.push([firePoint.latitude, firePoint.longitude]);
      visualizedFires += 1;
    }

    const impact24 = toFiniteNumber(prediction?.impact_assessment?.impact_probability_24h_pct) ?? 0;
    const pathColor = impact24 >= 75 ? "#c62828" : impact24 >= 55 ? "#e65c00" : impact24 >= 35 ? "#f9ab00" : "#1a73e8";
    const pathWeight = impact24 >= 75 ? 3.6 : impact24 >= 55 ? 3.1 : impact24 >= 35 ? 2.7 : 2.3;
    const pathClass = impact24 >= 75 ? "spread-path-line high" : impact24 >= 55 ? "spread-path-line medium" : "spread-path-line low";

    window.L.polyline(
      [
        [firePoint.latitude, firePoint.longitude],
        [latitude, longitude],
      ],
      {
        color: pathColor,
        weight: pathWeight,
        opacity: 0.88,
        dashArray: "8 8",
        className: pathClass,
      },
    ).addTo(state.mapIncidentLayer);
    spreadTracks += 1;

    const currentDistanceMiles = toFiniteNumber(
      prediction?.spread_prediction?.current_distance_miles ?? assessment?.distance_miles,
    ) ?? haversineMilesBetween(
      { latitude, longitude },
      { latitude: firePoint.latitude, longitude: firePoint.longitude },
    );

    if (!Number.isFinite(currentDistanceMiles) || currentDistanceMiles <= 0) {
      continue;
    }

    const forecastPoints = Array.isArray(forecast?.forecast_points)
      ? forecast.forecast_points.filter((point) => toFiniteNumber(point?.projected_distance_miles) != null).slice(0, 6)
      : [];

    for (const point of forecastPoints) {
      const projectedDistance = toFiniteNumber(point.projected_distance_miles);
      if (projectedDistance == null) {
        continue;
      }

      const fractionFromWatch = clampValue(projectedDistance / currentDistanceMiles, 0, 1);
      const forecastPoint = interpolateLatLng(
        { latitude, longitude },
        { latitude: firePoint.latitude, longitude: firePoint.longitude },
        fractionFromWatch,
      );

      const threat = toFiniteNumber(point.projected_threat_score) ?? 0;
      const fillColor = threat >= 75 ? "#d93025" : threat >= 45 ? "#e65c00" : "#f9ab00";
      const nodeTier = threat >= 75 ? "high" : threat >= 45 ? "medium" : "low";
      const nodeRadius = threat >= 75 ? 6.2 : threat >= 45 ? 5.4 : 4.8;
      const node = window.L.circleMarker([forecastPoint.latitude, forecastPoint.longitude], {
        radius: nodeRadius,
        color: "rgba(255,255,255,0.95)",
        weight: 1.4,
        fillColor,
        fillOpacity: 0.9,
        className: `spread-forecast-point ${nodeTier}`,
      });

      node.bindTooltip(
        `+${point.hour}h • ${projectedDistance.toFixed(1)} mi to watch • threat ${Math.round(threat)}/100`,
        { direction: "top", opacity: 0.92 },
      );
      node.addTo(state.mapIncidentLayer);
      bounds.push([forecastPoint.latitude, forecastPoint.longitude]);
      forecastNodes += 1;
    }
  }

  if (bounds.length === 1) {
    state.mapView.setView(bounds[0], 8);
  } else if (bounds.length > 1) {
    state.mapView.fitBounds(bounds, { padding: [24, 24], maxZoom: 9 });
  } else {
    state.mapView.setView([39.8283, -98.5795], 4);
  }

  if (elements.mapSummary) {
    const total = state.locations.length;
    if (!total) {
      elements.mapSummary.textContent = "Add watches to populate the map.";
    } else {
      const freshness = summarizeMapFreshness();
      const lastUpdated = state.lastRefreshAt ? formatLastUpdated(state.lastRefreshAt) : "Waiting for first refresh";
      const warning = freshness.stale_count > 0
        ? `${freshness.stale_count} stale watch${freshness.stale_count === 1 ? "" : "es"}. Refresh recommended.`
        : freshness.aging_count > 0
          ? `${freshness.aging_count} watch${freshness.aging_count === 1 ? "" : "es"} getting older.`
          : "All watches are fresh.";
      const warningClass = freshness.stale_count > 0 ? "warning" : freshness.aging_count > 0 ? "caution" : "ok";
      const incidentSummary = visualizedFires > 0
        ? `${visualizedFires} fire${visualizedFires === 1 ? "" : "s"} visualized with ${spreadTracks} spread path${spreadTracks === 1 ? "" : "s"} and ${forecastNodes} forecast node${forecastNodes === 1 ? "" : "s"}.`
        : "No active nearby fire visualization for current watches.";
      elements.mapSummary.innerHTML = `
        <span class="map-summary-line">${total} watch${total === 1 ? "" : "es"} mapped: ${counts.DANGER_NOW} danger now, ${counts.PREPARE} prepare, ${counts.WATCH_CLOSELY} watch closely, ${counts.SAFE} safe, ${counts.SYNCING} syncing.</span>
        <span class="map-summary-line">${escapeHtml(incidentSummary)}</span>
        <span class="map-summary-line"><strong>Last updated:</strong> ${escapeHtml(lastUpdated)}</span>
        <span class="map-summary-line map-summary-warning ${warningClass}">${escapeHtml(warning)}</span>
      `;
    }
  }

  state.mapView.invalidateSize();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url);
  if (response.ok) {
    return response.json();
  }
  if ((options.allow404 && response.status === 404) || (options.allow400 && response.status === 400)) {
    return null;
  }
  throw new Error(`Request failed with status ${response.status}`);
}

async function fetchJsonOrNull(url, options = {}) {
  try {
    return await fetchJson(url, options);
  } catch (error) {
    console.warn("Non-blocking data fetch failure for %s:", url, error);
    return null;
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "The request could not be completed.");
  }
  return data;
}

// Complete for text and quoted attributes, the only places it is used; never
// use it for URLs, where javascript: survives escaping.
function escapeHtml(value) {
  // nosemgrep: javascript.audit.detect-replaceall-sanitization.detect-replaceall-sanitization
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatEtaHours(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "Unknown";
  }
  const numeric = Number(value);
  if (numeric <= 0) {
    return "Now";
  }
  if (numeric >= 72) {
    return ">72h";
  }
  return `${numeric.toFixed(1)}h`;
}

function cardinalFromBearing(bearing) {
  const directions = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
  ];
  const normalized = ((Number(bearing) % 360) + 360) % 360;
  const index = Math.floor((normalized + 11.25) / 22.5) % 16;
  return directions[index];
}

function oppositeDirection(directionLabel) {
  const text = String(directionLabel || "").trim();
  if (!text) {
    return null;
  }
  const degreeMatch = text.match(/\(([-+]?\d+(?:\.\d+)?)\s*deg\)/i);
  if (!degreeMatch) {
    return null;
  }
  const bearing = Number(degreeMatch[1]);
  if (!Number.isFinite(bearing)) {
    return null;
  }
  const oppositeBearing = (bearing + 180) % 360;
  return `${cardinalFromBearing(oppositeBearing)} (${Math.round(oppositeBearing)} deg)`;
}

function formatCountdown(remainingMs) {
  const totalSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatNextRefreshLabel() {
  if (!state.locations.length) {
    return "Paused";
  }
  if (!state.nextRefreshAt) {
    return "Scheduling...";
  }
  return formatCountdown(Math.max(0, state.nextRefreshAt - Date.now()));
}

function fallbackDecisionState(alertLevel) {
  if (alertLevel === "EVACUATE" || alertLevel === "WARNING") {
    return "DANGER_NOW";
  }
  if (alertLevel === "PREPARE") {
    return "PREPARE";
  }
  if (alertLevel === "MONITOR") {
    return "WATCH_CLOSELY";
  }
  return "SYNCING";
}

function formatDecisionState(stateValue) {
  const labels = {
    SYNCING: "Syncing Data",
    SAFE: "Safe",
    WATCH_CLOSELY: "Watch Closely",
    PREPARE: "Prepare",
    DANGER_NOW: "Danger Now",
  };
  return labels[stateValue] || "Syncing Data";
}

function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildLocationSnapshot({ assessment, prediction, fetchedAt }) {
  const distanceMiles = toFiniteNumber(assessment?.distance_miles);
  const threatScore = toFiniteNumber(prediction?.threat_score);
  const decisionState = prediction?.decision_state || fallbackDecisionState(assessment?.alert_level);
  const updatedAt = prediction?.generated_at || fetchedAt;
  return {
    distance_miles: distanceMiles,
    threat_score: threatScore,
    decision_state: decisionState,
    updated_at: updatedAt,
  };
}

function buildLocationChange(previous, current) {
  if (!previous) {
    return {
      has_baseline: false,
      distance_delta_miles: null,
      threat_delta: null,
      previous_state: null,
      current_state: current?.decision_state || null,
      state_changed: false,
    };
  }

  const previousDistance = toFiniteNumber(previous.distance_miles);
  const currentDistance = toFiniteNumber(current?.distance_miles);
  const previousThreat = toFiniteNumber(previous.threat_score);
  const currentThreat = toFiniteNumber(current?.threat_score);
  const previousState = String(previous.decision_state || "");
  const currentState = String(current?.decision_state || "");

  return {
    has_baseline: true,
    distance_delta_miles: previousDistance == null || currentDistance == null
      ? null
      : Number((currentDistance - previousDistance).toFixed(1)),
    threat_delta: previousThreat == null || currentThreat == null
      ? null
      : Math.round(currentThreat - previousThreat),
    previous_state: previousState || null,
    current_state: currentState || null,
    state_changed: Boolean(previousState && currentState && previousState !== currentState),
  };
}

function freshnessStateFor(updatedAt) {
  if (!updatedAt) {
    return {
      className: "stale",
      message: "Waiting for first refresh",
    };
  }

  const updatedAtMs = Number(new Date(updatedAt));
  if (!Number.isFinite(updatedAtMs)) {
    return {
      className: "stale",
      message: "Timestamp unavailable",
    };
  }

  const ageSeconds = Math.max(0, Math.floor((Date.now() - updatedAtMs) / 1000));
  if (ageSeconds <= 90) {
    return {
      className: "fresh",
      message: `Fresh (${formatRelativeAge(updatedAt)} ago)`,
    };
  }
  if (ageSeconds <= 300) {
    return {
      className: "aging",
      message: `Aging (${formatRelativeAge(updatedAt)} ago)`,
    };
  }
  return {
    className: "stale",
    message: `Stale (${formatRelativeAge(updatedAt)} ago). Refresh recommended.`,
  };
}

function formatRelativeAge(updatedAt) {
  const updatedAtMs = Number(new Date(updatedAt));
  if (!Number.isFinite(updatedAtMs)) {
    return "unknown";
  }

  const seconds = Math.max(0, Math.floor((Date.now() - updatedAtMs) / 1000));
  if (seconds < 10) {
    return "just now";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h`;
  }
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

function formatLastUpdated(updatedAt) {
  if (!updatedAt) {
    return "Waiting for first refresh";
  }

  const timestamp = new Date(updatedAt);
  if (Number.isNaN(timestamp.getTime())) {
    return "Timestamp unavailable";
  }

  const absolute = timestamp.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
  return `${absolute} (${formatRelativeAge(updatedAt)} ago)`;
}

function formatDistanceDelta(change) {
  if (!change?.has_baseline) {
    return "Baseline set";
  }

  const delta = toFiniteNumber(change.distance_delta_miles);
  if (delta == null) {
    return "No distance data";
  }
  if (Math.abs(delta) < 0.05) {
    return "No meaningful change";
  }
  if (delta < 0) {
    return `${delta.toFixed(1)} mi (closer)`;
  }
  return `+${delta.toFixed(1)} mi (farther)`;
}

function formatThreatDelta(change) {
  if (!change?.has_baseline) {
    return "Baseline set";
  }

  const delta = toFiniteNumber(change.threat_delta);
  if (delta == null) {
    return "No threat data";
  }
  if (delta === 0) {
    return "Unchanged";
  }
  return delta > 0 ? `+${delta} points` : `${delta} points`;
}

function formatStateDelta(change) {
  if (!change?.has_baseline) {
    return "Baseline set";
  }
  if (!change.previous_state || !change.current_state) {
    return "No state data";
  }
  if (!change.state_changed) {
    return "Unchanged";
  }
  return `${formatDecisionState(change.previous_state)} -> ${formatDecisionState(change.current_state)}`;
}

function summarizeMapFreshness() {
  const summary = {
    fresh_count: 0,
    aging_count: 0,
    stale_count: 0,
  };

  for (const location of state.locations) {
    const snapshot = state.locationSnapshots.get(location.location_id);
    const freshness = freshnessStateFor(snapshot?.updated_at);
    if (freshness.className === "fresh") {
      summary.fresh_count += 1;
    } else if (freshness.className === "aging") {
      summary.aging_count += 1;
    } else {
      summary.stale_count += 1;
    }
  }

  return summary;
}