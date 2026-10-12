import assert from "node:assert/strict";
import test from "node:test";

import worker from "../app/static/_worker.js";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function textResponse(payload, status = 200) {
  return new Response(String(payload), {
    status,
    headers: { "content-type": "text/plain" },
  });
}

function buildCsv(rows) {
  const headers = ["latitude", "longitude", "confidence", "type", "frp"];
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push([
      row.latitude,
      row.longitude,
      row.confidence,
      row.type,
      row.frp,
    ].join(","));
  }
  return `${lines.join("\n")}\n`;
}

function createFetchMock({
  firmsRowsBySource = {},
  eonetEvents = [],
  airQuality = null,
  geocode = null,
}) {
  return async (input) => {
    const url = String(typeof input === "string" ? input : input.url);

    if (url.includes("nominatim.openstreetmap.org/search")) {
      if (!geocode) {
        return jsonResponse([]);
      }
      return jsonResponse([
        {
          lat: String(geocode.latitude),
          lon: String(geocode.longitude),
        },
      ]);
    }

    if (url.includes("air-quality-api.open-meteo.com")) {
      if (!airQuality) {
        return jsonResponse({ current: {} });
      }
      return jsonResponse({
        current: {
          us_aqi: airQuality.us_aqi,
          pm2_5: airQuality.pm2_5,
          time: airQuality.time || "2026-05-21T04:00",
        },
      });
    }

    if (url.includes("eonet.gsfc.nasa.gov/api/v3/events")) {
      return jsonResponse({ events: eonetEvents });
    }

    if (url.includes("firms.modaps.eosdis.nasa.gov/api/area/csv/")) {
      const sourceMatch = url.match(/\/api\/area\/csv\/[^/]+\/([^/]+)\//);
      const source = sourceMatch?.[1] || "";
      const rows = firmsRowsBySource[source] || [];
      return textResponse(buildCsv(rows));
    }

    return new Response("not mocked", { status: 404 });
  };
}

class MockD1Statement {
  constructor(db, sql) {
    this.db = db;
    this.sql = sql;
    this.args = [];
  }

  bind(...args) {
    this.args = args;
    return this;
  }

  async first() {
    if (this.sql.includes("FROM locations") && this.sql.includes("WHERE location_id = ?1")) {
      const location = this.db.locations.get(this.args[0]);
      if (!location || !location.is_active) {
        return null;
      }
      return { ...location };
    }
    return null;
  }

  async all() {
    if (this.sql.includes("FROM locations") && this.sql.includes("WHERE is_active = 1")) {
      const rows = [...this.db.locations.values()]
        .filter((row) => row.is_active)
        .sort((a, b) => (b.id - a.id))
        .map((row) => ({ ...row }));
      return { results: rows };
    }
    return { results: [] };
  }

  async run() {
    if (this.sql.startsWith("INSERT INTO locations")) {
      const [locationId, displayName, icon, address, latitude, longitude, geocodedAt] = this.args;
      const existing = this.db.locations.get(locationId);
      const now = "2026-05-21 00:00:00";
      if (existing) {
        this.db.locations.set(locationId, {
          ...existing,
          display_name: displayName,
          icon,
          address,
          latitude,
          longitude,
          geocoded_at: geocodedAt,
          is_active: 1,
          updated_at: now,
        });
      } else {
        this.db.locations.set(locationId, {
          id: this.db.nextId++,
          location_id: locationId,
          display_name: displayName,
          icon,
          address,
          latitude,
          longitude,
          geocoded_at: geocodedAt,
          is_active: 1,
          created_at: now,
          updated_at: now,
        });
      }
      return { meta: { changes: 1 } };
    }

    if (this.sql.startsWith("UPDATE locations SET is_active = 0")) {
      const locationId = this.args[0];
      const existing = this.db.locations.get(locationId);
      if (!existing || !existing.is_active) {
        return { meta: { changes: 0 } };
      }
      this.db.locations.set(locationId, {
        ...existing,
        is_active: 0,
      });
      return { meta: { changes: 1 } };
    }

    return { meta: { changes: 0 } };
  }
}

class MockD1Database {
  constructor(initialRows = []) {
    this.locations = new Map();
    this.nextId = 1;
    for (const row of initialRows) {
      this.locations.set(row.location_id, {
        id: this.nextId++,
        location_id: row.location_id,
        display_name: row.display_name,
        icon: row.icon || "P",
        address: row.address ?? null,
        latitude: row.latitude,
        longitude: row.longitude,
        geocoded_at: row.geocoded_at || "2026-05-21T00:00:00Z",
        is_active: row.is_active ?? 1,
        created_at: "2026-05-21 00:00:00",
        updated_at: "2026-05-21 00:00:00",
      });
    }
  }

  prepare(sql) {
    return new MockD1Statement(this, sql);
  }
}

function makeEnv(db) {
  return {
    FIREWATCH_DB: db,
    FIRMS_MAP_KEY: "test-key",
    FIRMS_SOURCE_IDS: "VIIRS_NOAA21_NRT",
    FIRMS_RADIUS_MILES: "75",
    FIRMS_DAY_RANGE: "1",
    FIRMS_MIN_CONFIDENCE: "60",
    FIRMS_MIN_FRP: "6",
    FIRMS_VEGETATION_ONLY: "true",
    MAX_FIRE_DISTANCE_MILES: "300",
    ASSETS: {
      fetch: async () => new Response("asset", { status: 200 }),
    },
  };
}

async function callEdge(path, { method = "GET", body = null, env, fetchMock }) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = fetchMock;
  try {
    const request = new Request(`https://example.pages.dev${path}`, {
      method,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const response = await worker.fetch(request, env);
    return response;
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("filters noisy FIRMS anomalies and uses named wildfire", async () => {
  const db = new MockD1Database([
    {
      location_id: "zip-91604",
      display_name: "ZIP 91604",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/alerts/assessment/zip-91604", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.1435, longitude: -118.3905, confidence: 95, type: 2, frp: 22 },
          { latitude: 34.1436, longitude: -118.3906, confidence: 40, type: 0, frp: 18 },
          { latitude: 34.1437, longitude: -118.3907, confidence: 90, type: 0, frp: 2 },
        ],
      },
      eonetEvents: [
        {
          id: "EONET_1",
          title: "SANDY Wildfire, Ventura, California",
          geometry: [{ coordinates: [-118.0, 34.2] }],
        },
      ],
      airQuality: { us_aqi: 82, pm2_5: 41.7, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.nearest_fire_name, "SANDY Wildfire, Ventura, California");
  assert.ok(payload.nearest_fire_location);
  assert.ok(Number.isFinite(payload.nearest_fire_location.latitude));
  assert.ok(Number.isFinite(payload.nearest_fire_location.longitude));
  assert.equal(payload.alert_level, "MONITOR");
});

test("uses FIRMS thermal anomaly when high-quality and nearest", async () => {
  const db = new MockD1Database([
    {
      location_id: "close-watch",
      display_name: "Close Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/alerts/assessment/close-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.144, longitude: -118.391, confidence: 92, type: 0, frp: 25 },
        ],
      },
      eonetEvents: [
        {
          id: "EONET_2",
          title: "Far Wildfire",
          geometry: [{ coordinates: [-117.6, 34.5] }],
        },
      ],
      airQuality: { us_aqi: 45, pm2_5: 10.0, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.nearest_fire_name, "VIIRS NOAA-21 thermal anomaly");
  assert.ok(payload.distance_miles < 1);
});

test("supports zero coordinates for location create and assessment", async () => {
  const db = new MockD1Database();
  const env = makeEnv(db);
  const fetchMock = createFetchMock({
    firmsRowsBySource: { VIIRS_NOAA21_NRT: [] },
    eonetEvents: [],
    airQuality: null,
  });

  const createResponse = await callEdge("/edge/v1/locations", {
    method: "POST",
    body: {
      location_id: "zero-point",
      display_name: "Zero Point",
      icon: "P",
      latitude: 0,
      longitude: 0,
      persist_address: false,
    },
    env,
    fetchMock,
  });

  assert.equal(createResponse.status, 200);

  const assessmentResponse = await callEdge("/edge/v1/alerts/assessment/zero-point", {
    env,
    fetchMock,
  });

  assert.equal(assessmentResponse.status, 200);
  const payload = await assessmentResponse.json();
  assert.equal(payload.location_id, "zero-point");
  assert.equal(payload.nearest_fire_name, null);
});

test("rejects out-of-range coordinates for location create", async () => {
  const db = new MockD1Database();
  const env = makeEnv(db);
  const fetchMock = createFetchMock({
    firmsRowsBySource: { VIIRS_NOAA21_NRT: [] },
    eonetEvents: [],
    airQuality: null,
  });

  const createResponse = await callEdge("/edge/v1/locations", {
    method: "POST",
    body: {
      location_id: "bad-coords",
      display_name: "Bad Coords",
      icon: "P",
      latitude: 91,
      longitude: -118,
      persist_address: false,
    },
    env,
    fetchMock,
  });

  assert.equal(createResponse.status, 400);
  const payload = await createResponse.json();
  assert.match(payload.detail, /latitude must be between -90 and 90/i);
});

test("returns 422 when address cannot be resolved", async () => {
  const db = new MockD1Database();
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/locations/resolve", {
    method: "POST",
    body: { address: "not-a-real-place-xyz" },
    env,
    fetchMock: createFetchMock({ geocode: null }),
  });

  assert.equal(response.status, 422);
  const payload = await response.json();
  assert.match(payload.detail, /could not resolve/i);
});

test("serves docs directly from Worker without backend proxy", async () => {
  const db = new MockD1Database();
  const env = makeEnv(db);
  let upstreamCalls = 0;

  const response = await callEdge("/docs", {
    env,
    fetchMock: async () => {
      upstreamCalls += 1;
      return new Response("unexpected upstream", { status: 500 });
    },
  });

  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Firewatch Edge API/i);
  assert.match(html, /GET \/edge\/v1\/health\/diagnostics/i);
  assert.equal(upstreamCalls, 0);
});

test("serves openapi json directly from Worker", async () => {
  const db = new MockD1Database();
  const env = makeEnv(db);
  let upstreamCalls = 0;

  const response = await callEdge("/openapi.json", {
    env,
    fetchMock: async () => {
      upstreamCalls += 1;
      return new Response("unexpected upstream", { status: 500 });
    },
  });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "application/json");
  const payload = await response.json();
  assert.equal(payload.info.title, "Firewatch Edge API");
  assert.ok(payload.paths["/edge/v1/health"]);
  assert.ok(payload.paths["/edge/v1/health/diagnostics"]);
  assert.ok(payload.paths["/edge/v1/alerts/assessment/{location_id}"]);
  assert.ok(payload.paths["/edge/v1/predictions/evacuation/{location_id}"]);
  assert.ok(payload.paths["/edge/v1/predictions/weather-impact/{location_id}"]);
  assert.ok(payload.paths["/edge/v1/predictions/anomalies/{location_id}"]);
  assert.ok(payload.paths["/edge/v1/intelligence/{location_id}"]);
  assert.ok(payload.paths["/edge/v1/audit/provenance/{location_id}"]);
  assert.equal(upstreamCalls, 0);
});

test("returns structured dependency diagnostics for upstream services", async () => {
  const db = new MockD1Database([
    {
      location_id: "diag-watch",
      display_name: "Diag Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/health/diagnostics", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.144, longitude: -118.392, confidence: 95, type: 0, frp: 21 },
        ],
      },
      eonetEvents: [
        {
          id: "EONET_DIAG_1",
          title: "Diagnostic Wildfire",
          geometry: [{ coordinates: [-118.0, 34.2] }],
        },
      ],
      airQuality: { us_aqi: 62, pm2_5: 15.1, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.ok(["healthy", "degraded", "unhealthy"].includes(payload.status));
  assert.equal(payload.edge, true);
  assert.ok(payload.dependencies);
  assert.equal(payload.dependencies.d1.status, "ok");
  assert.equal(payload.dependencies.eonet.status, "ok");
  assert.equal(payload.dependencies.firms.status, "ok");
  assert.equal(payload.dependencies.aqi.status, "ok");
  assert.ok(payload.summary);
});

test("recovers EONET diagnostics via fallback probe when primary events endpoint fails", async () => {
  const db = new MockD1Database([
    {
      location_id: "diag-eonet-fallback",
      display_name: "Diag EONET Fallback",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);
  const baseFetch = createFetchMock({
    firmsRowsBySource: {
      VIIRS_NOAA21_NRT: [
        { latitude: 34.144, longitude: -118.392, confidence: 95, type: 0, frp: 21 },
      ],
    },
    eonetEvents: [],
    airQuality: { us_aqi: 62, pm2_5: 15.1, time: "2026-05-21T04:00" },
  });

  const fetchMock = async (input) => {
    const url = String(typeof input === "string" ? input : input.url);
    if (url.includes("eonet.gsfc.nasa.gov/api/v3/events")) {
      return new Response("temporary upstream issue", { status: 503 });
    }
    if (url.includes("eonet.gsfc.nasa.gov/api/v3/categories/wildfires")) {
      return jsonResponse({ title: "Wildfires" });
    }
    return baseFetch(input);
  };

  const response = await callEdge("/edge/v1/health/diagnostics", {
    env,
    fetchMock,
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.dependencies.eonet.status, "ok");
  assert.equal(payload.dependencies.eonet.probe, "category_wildfires");
  assert.equal(payload.dependencies.eonet.recovered_via_fallback, true);
  assert.ok(Array.isArray(payload.dependencies.eonet.failed_probes));
  assert.ok(payload.dependencies.eonet.failed_probes.length >= 1);
});

test("supports /api/v1 compatibility alias without backend proxy", async () => {
  const db = new MockD1Database();
  const env = makeEnv(db);
  let upstreamCalls = 0;

  const response = await callEdge("/api/v1/health", {
    env,
    fetchMock: async () => {
      upstreamCalls += 1;
      return new Response("unexpected upstream", { status: 500 });
    },
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.status, "healthy");
  assert.equal(upstreamCalls, 0);
});

test("lists active locations via edge endpoint", async () => {
  const db = new MockD1Database([
    {
      location_id: "loc-active-1",
      display_name: "Active One",
      latitude: 34.1,
      longitude: -118.3,
      is_active: 1,
    },
    {
      location_id: "loc-inactive",
      display_name: "Inactive",
      latitude: 35.1,
      longitude: -119.3,
      is_active: 0,
    },
    {
      location_id: "loc-active-2",
      display_name: "Active Two",
      latitude: 36.1,
      longitude: -120.3,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/locations", {
    env,
    fetchMock: createFetchMock({}),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.ok(Array.isArray(payload));
  assert.equal(payload.length, 2);
  assert.deepEqual(payload.map((item) => item.location_id), ["loc-active-2", "loc-active-1"]);
});

test("returns edge-native threat prediction for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "prediction-watch",
      display_name: "Prediction Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/threat/prediction-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.152, longitude: -118.405, confidence: 96, type: 0, frp: 30 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 88, pm2_5: 33.2, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "prediction-watch");
  assert.ok(Number.isFinite(payload.threat_score));
  assert.ok(payload.nearest_fire_location);
  assert.ok(Number.isFinite(payload.nearest_fire_location.latitude));
  assert.ok(Number.isFinite(payload.nearest_fire_location.longitude));
  assert.ok(payload.spread_prediction);
  assert.ok(Number.isFinite(payload.spread_prediction.estimated_spread_rate_mph));
  assert.equal(payload.inference_mode, "heuristic_only");
  assert.equal(payload.prediction_engine?.kind, "edge_heuristic");
  assert.ok(payload.forecast_summary);
  assert.ok(payload.impact_assessment);
  assert.ok(["SAFE", "WATCH_CLOSELY", "PREPARE", "DANGER_NOW"].includes(payload.decision_state));
  assert.ok(payload.decision_summary);
  assert.ok(Array.isArray(payload.decision_factors));
  assert.equal(typeof payload.impact_assessment.likely_impact_within_24h, "boolean");
  assert.equal(typeof payload.impact_assessment.likely_impact_within_72h, "boolean");
  assert.equal(typeof payload.impact_assessment.impact_probability_24h_pct, "number");
  assert.equal(typeof payload.impact_assessment.impact_probability_72h_pct, "number");
  assert.ok(payload.impact_assessment.impact_probability_24h_pct >= 0 && payload.impact_assessment.impact_probability_24h_pct <= 100);
  assert.ok(payload.impact_assessment.impact_probability_72h_pct >= 0 && payload.impact_assessment.impact_probability_72h_pct <= 100);
  assert.ok(Array.isArray(payload.impact_assessment.impact_driver_breakdown));
  assert.ok(Array.isArray(payload.impact_assessment.top_impact_drivers));
  assert.ok(payload.impact_assessment.top_impact_drivers.length > 0);
  assert.ok(payload.impact_assessment.note);
});

test("prefers stronger SNPP signal over slightly closer weaker source", async () => {
  const db = new MockD1Database([
    {
      location_id: "priority-watch",
      display_name: "Priority Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);
  env.FIRMS_SOURCE_IDS = "VIIRS_SNPP_NRT,VIIRS_NOAA21_NRT";

  const response = await callEdge("/edge/v1/predictions/threat/priority-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.2150, longitude: -118.3906, confidence: 60, type: 0, frp: 6 },
        ],
        VIIRS_SNPP_NRT: [
          { latitude: 34.2220, longitude: -118.3906, confidence: 99, type: 0, frp: 95 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 74, pm2_5: 21.4, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.nearest_fire_name, "VIIRS SNPP thermal anomaly");
  assert.equal(payload.satellite_signal?.source_id, "VIIRS_SNPP_NRT");
});

test("warning alert applies minimum 24h and 72h impact probability floors", async () => {
  const db = new MockD1Database([
    {
      location_id: "warning-floor-watch",
      display_name: "Warning Floor Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/threat/warning-floor-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.1800, longitude: -118.4200, confidence: 95, type: 0, frp: 20 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 185, pm2_5: 85.0, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.alert_level, "WARNING");
  assert.ok(payload.impact_assessment.impact_probability_24h_pct >= 72);
  assert.ok(payload.impact_assessment.impact_probability_72h_pct >= 84);
  assert.ok(payload.impact_assessment.impact_probability_72h_pct >= payload.impact_assessment.impact_probability_24h_pct);
});

test("threat payload includes satellite signal metadata fields", async () => {
  const db = new MockD1Database([
    {
      location_id: "signal-watch",
      display_name: "Signal Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/threat/signal-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.145, longitude: -118.392, confidence: 97, type: 0, frp: 33 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 80, pm2_5: 29.0, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.ok(payload.satellite_signal);
  assert.equal(payload.satellite_signal.source_id, "VIIRS_NOAA21_NRT");
  assert.equal(payload.satellite_signal.source_label, "VIIRS NOAA-21 thermal anomaly");
  assert.ok(Number.isFinite(payload.satellite_signal.source_priority));
  assert.equal(payload.satellite_signal.confidence_pct, 97);
  assert.equal(payload.satellite_signal.frp, 33);
});

test("classifies nearby elevated wildfire as possible 24h impact", async () => {
  const db = new MockD1Database([
    {
      location_id: "nearby-impact-watch",
      display_name: "Nearby Impact Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/threat/nearby-impact-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: { VIIRS_NOAA21_NRT: [] },
      eonetEvents: [
        {
          id: "EONET_NEAR",
          title: "SANDY Wildfire, Ventura, California",
          geometry: [{ coordinates: [-118.0, 34.2] }],
        },
      ],
      airQuality: { us_aqi: 92, pm2_5: 41.8, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.model_status, "active_fire_modeled");
  assert.equal(payload.impact_assessment.likely_impact_within_24h, true);
  assert.equal(payload.impact_assessment.risk_to_watch_level, "HIGH");
  assert.ok(payload.impact_assessment.impact_probability_24h_pct >= 55);
  assert.ok(payload.impact_assessment.impact_probability_72h_pct >= payload.impact_assessment.impact_probability_24h_pct);
  assert.equal(payload.decision_state, "WATCH_CLOSELY");
});

test("returns explicit no-active-fire prediction state", async () => {
  const db = new MockD1Database([
    {
      location_id: "no-fire-watch",
      display_name: "No Fire Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/threat/no-fire-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: { VIIRS_NOAA21_NRT: [] },
      eonetEvents: [],
      airQuality: { us_aqi: 92, pm2_5: 41.8, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.model_status, "no_active_fire_nearby");
  assert.equal(payload.inference_mode, "heuristic_only");
  assert.equal(payload.threat_score, 0);
  assert.equal(payload.nearest_fire_location, null);
  assert.equal(payload.spread_prediction.estimated_spread_rate_mph, null);
  assert.equal(payload.spread_prediction.eta_to_5_miles_hours, null);
  assert.equal(payload.forecast_summary.trend, "NO_ACTIVE_FIRE");
  assert.equal(payload.weather_impact.active_fire_modeled, false);
  assert.equal(payload.impact_assessment.likely_impact_within_24h, false);
  assert.equal(payload.impact_assessment.likely_impact_within_72h, false);
  assert.equal(payload.impact_assessment.impact_probability_24h_pct, 0);
  assert.equal(payload.impact_assessment.impact_probability_72h_pct, 0);
  assert.deepEqual(payload.impact_assessment.top_impact_drivers, []);
  assert.equal(payload.decision_state, "SAFE");
});

test("returns forecast timeline for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "forecast-watch",
      display_name: "Forecast Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/forecast/forecast-watch?hours=48", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.149, longitude: -118.398, confidence: 92, type: 0, frp: 24 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 74, pm2_5: 21.4, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "forecast-watch");
  assert.equal(payload.horizon_hours, 48);
  assert.ok(payload.nearest_fire_location);
  assert.ok(Number.isFinite(payload.nearest_fire_location.latitude));
  assert.ok(Number.isFinite(payload.nearest_fire_location.longitude));
  assert.ok(Array.isArray(payload.forecast_points));
  assert.ok(payload.forecast_points.length > 0);
});

test("returns evacuation guidance for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "evac-watch",
      display_name: "Evac Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/evacuation/evac-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.144, longitude: -118.392, confidence: 99, type: 0, frp: 34 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 102, pm2_5: 42.0, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "evac-watch");
  assert.ok(payload.urgency);
  assert.ok(Array.isArray(payload.actions));
});

test("returns weather impact for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "weather-watch",
      display_name: "Weather Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/weather-impact/weather-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.15, longitude: -118.4, confidence: 91, type: 0, frp: 26 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 88, pm2_5: 34.5, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "weather-watch");
  assert.ok(Number.isFinite(payload.fire_weather_index));
  assert.ok(Number.isFinite(payload.spread_multiplier));
});

test("returns anomaly analysis for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "anomaly-watch",
      display_name: "Anomaly Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/predictions/anomalies/anomaly-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.144, longitude: -118.391, confidence: 99, type: 0, frp: 40 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 118, pm2_5: 49.0, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "anomaly-watch");
  assert.ok(Number.isFinite(payload.anomaly_score));
  assert.equal(typeof payload.is_anomalous, "boolean");
});

test("returns external intelligence payload for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "intel-watch",
      display_name: "Intel Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/intelligence/intel-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: { VIIRS_NOAA21_NRT: [] },
      eonetEvents: [],
      airQuality: { us_aqi: 60, pm2_5: 14.5, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "intel-watch");
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "nws_alerts"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "hrrr_wind_proxy"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "incident_perimeters"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "hms_smoke"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "purpleair"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "road_closures"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "terrain_context"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "utility_status"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "lightning_nowcast"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "feed_status"));
  assert.ok(Object.prototype.hasOwnProperty.call(payload, "reliability"));
  assert.equal(typeof payload.reliability.success_ratio, "number");
});

test("returns one-call provenance audit payload for watched location", async () => {
  const db = new MockD1Database([
    {
      location_id: "audit-watch",
      display_name: "Audit Watch",
      latitude: 34.1434152,
      longitude: -118.3906478,
      is_active: 1,
    },
  ]);
  const env = makeEnv(db);

  const response = await callEdge("/edge/v1/audit/provenance/audit-watch", {
    env,
    fetchMock: createFetchMock({
      firmsRowsBySource: {
        VIIRS_NOAA21_NRT: [
          { latitude: 34.145, longitude: -118.392, confidence: 95, type: 0, frp: 28 },
        ],
      },
      eonetEvents: [],
      airQuality: { us_aqi: 70, pm2_5: 19.1, time: "2026-05-21T04:00" },
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.location_id, "audit-watch");
  assert.ok(payload.edge_provenance);
  assert.equal(payload.edge_provenance.inference_mode, "heuristic_only");
  assert.equal(payload.edge_provenance.prediction_engine?.kind, "edge_heuristic");
  assert.ok(payload.backend_provenance);
  assert.equal(payload.backend_provenance.status, "not_configured");
  assert.ok(payload.diagnostics_summary);
});
