# Firewatch

Wildfire monitoring and evacuation alerts for a list of places you care about.
Firewatch tracks nearby fires, measures how far away each one is and whether it
is closing in, and rates each place from SAFE to EVACUATE.

It runs in two forms that share a browser client (`app/static/`):

- **Edge app** (`app/static/_worker.js`): a Cloudflare Pages worker backed by a
  D1 database, serving `/edge/v1/*`. Its predictions are heuristics and say so
  (`inference_mode: heuristic_only`).
- **Python backend** (`app/`): a FastAPI service with SQLite or PostgreSQL,
  serving `/api/v1/*`. It keeps distance history, trains a spread model on it,
  and serves that model only once it passes a promotion gate.

## Data sources

| What | Source | Needs |
|---|---|---|
| Incidents | CAL FIRE incident API | |
| Satellite hotspots | NASA FIRMS (VIIRS, MODIS) | `FIRMS_MAP_KEY` |
| Scene metadata | Microsoft Planetary Computer (Sentinel-2, Landsat) | |
| Weather | NWS, Open-Meteo, Meteostat, blended | |
| Air quality | Open-Meteo; EPA AirNow merged in when keyed | `AIRNOW_API_KEY` |
| Smoke | NOAA HMS smoke polygons | |
| Perimeters, road closures, utility status, lightning | GeoJSON or JSON feeds you configure | the matching `*_URL` |
| PurpleAir sensors | PurpleAir API | `PURPLEAIR_API_KEY` |
| Geocoding | Nominatim, with the US Census geocoder as fallback | |

`GET /api/v1/predictions/data-sources` lists what the backend uses and what it
still lacks.

## Alert levels

Distance thresholds come from `EVACUATION_DISTANCE_EVACUATE`, `_WARNING`,
`_PREPARE` and `_MONITOR` (defaults 1, 2, 3 and 5 miles); anything farther is
SAFE. Large or poorly contained fires raise the level.

Fire weather danger uses the Fosberg Fire Weather Index (Fosberg 1978), which
needs only temperature, humidity and wind. The Canadian FWI system would need
a record of prior days that Firewatch does not keep.

## Running the backend

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env    # every key is optional; see below
make db-init
make run                # http://localhost:8000, API docs at /docs
```

Or with Docker: `make docker-build && make docker-run`. For PostgreSQL,
uncomment the `postgres` service in `docker-compose.yml` and set
`DATABASE_URL=postgresql://firewatch:password@postgres:5432/firewatch`.

### Configuration

`.env.example` lists every setting with its default. Settings rejects unknown
keys, so a typo stops startup rather than being ignored. The ones you are most
likely to change:

- `DATABASE_URL`, `FIRE_DATA_FETCH_INTERVAL` (seconds, default 600),
  `MAX_FIRE_DISTANCE_MILES`
- `FIRE_STALE_HOURS` (default 6): a fire no successful fetch has reported
  for this long is treated as out; it becomes active again if reported. A
  failed fetch retires nothing.
- API keys and feed URLs from the table above
- `HISTORICAL_DATA_RETENTION_HOURS`: how long movement history is kept in
  memory (default 24)
- `SPREAD_MODEL_*`: the promotion gate (below)

## API

Backend routes, under `/api/v1`:

| Route | |
|---|---|
| `GET /health`, `GET /status` | liveness and service status |
| `GET, POST /locations`; `GET, PUT, DELETE /locations/{id}` | monitored places |
| `POST /locations/resolve` | geocode an address without storing it |
| `GET /fires`, `GET /fires/{id}`, `GET /fires/{id}/alerts`, `GET /fires/recent` | fires and their alerts |
| `GET /alerts`, `GET /alerts/{id}`, `PUT /alerts/{id}/acknowledge` | alerts |
| `GET /alerts/assessment/{location_id}` | current threat for a place |
| `GET /predictions/threat/{location_id}`, `/evacuation/{location_id}`, `/forecast/{location_id}`, `/weather-impact/{location_id}` | model-backed analysis |
| `GET /predictions/spread-rate/{fire_id}`, `/anomalies/{fire_id}`, `/impact-timeline/{fire_id}/{location_id}`, `/risk-score/{location_id}/{fire_id}` | per-fire analysis |

Prediction responses include `model_provenance`. The edge app serves the same
shapes under `/edge/v1` (locations, assessment, predictions, `/intelligence/{location_id}`,
`/health/diagnostics`); `/api/v1` there is an alias.

## Models and benchmarks

- **Spread rate**: a regressor trained on stored distance history, chosen by
  benchmark from random-forest, extra-trees, gradient-boosting and voting
  ensembles against a random-forest baseline. It replaces the empirical
  estimate only after the promotion gate (`SPREAD_MODEL_PROMOTION_*`) passes,
  and is then written to `SPREAD_MODEL_PROMOTED_PATH`. Direction uses a random
  forest.
- **Anomalies**: an isolation forest plus fixed thresholds on spread and
  approach speed.
- **Risk score**: a weighted combination of fire size, containment, distance
  and trajectory.

The benchmark CLI measures whether the data can support a model and whether a
model beats the baseline:

```bash
python -m app.ml.benchmark_cli report        # label quality and current metrics
python -m app.ml.benchmark_cli capture --iterations 8 --interval-seconds 90 --run-benchmark
python -m app.ml.benchmark_cli seed --pair-limit 24 --points-per-pair 18 --run-benchmark
python -m app.ml.benchmark_cli bakeoff       # adds XGBoost, LightGBM, CatBoost if installed
python -m app.ml.benchmark_cli real-eval --dataset-name all --download-if-missing
```

`seed` writes synthetic trajectories, which help exercise the pipeline but are
not evidence that a model works. `real-eval` uses the UCI Forest Fires and
Algerian Forest Fires datasets. Each command has a `make benchmark-*` target.

## Development

```bash
make dev      # install with test and lint tools
make lint     # ruff and mypy
make format
make test
make audit    # tests, JS tests and syntax checks, as CI runs them
```

`audit.py --require-production-ready --min-scorecard-grade B` also enforces the
model scorecard. It needs a database with fire history, so use it against a
deployment's data, not a fresh checkout.

## Deploying the edge app

```bash
wrangler d1 create firewatch-edge        # put the ID in wrangler.toml
wrangler d1 execute firewatch-edge --remote --file d1/schema.sql
make deploy-edge                          # PAGES_PROJECT defaults to "firewatch"
make verify-edge EDGE_BASE_URL=https://<project>.pages.dev
```

Set API keys as Pages secrets. `PROVENANCE_BACKEND_BASE_URL`, if set, lets
`/edge/v1/audit/provenance/{location_id}` compare against a running backend.

## License

AGPL-3.0-or-later. If you run a modified Firewatch for others, offer them its
source; the web client links to it in its footer. The test fixtures under
`tests/fixtures/` are UCI excerpts under CC BY 4.0 (see their README).
