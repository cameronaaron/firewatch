import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_frontend_uses_edge_api_prefix():
    app_js = (_repo_root() / "app" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'const API_PREFIX = "/edge/v1";' in app_js
    assert 'const API_PREFIX = "/api/v1";' not in app_js
    assert "predictions/threat/" in app_js
    assert "predictions/forecast/" in app_js
    assert "predictions/evacuation/" in app_js
    assert "predictions/weather-impact/" in app_js
    assert "predictions/anomalies/" in app_js
    assert "/intelligence/" in app_js


def test_wrangler_has_d1_binding_and_firms_filters():
    wrangler = (_repo_root() / "wrangler.toml").read_text(encoding="utf-8")
    config = tomllib.loads(wrangler)

    vars_section = config.get("vars", {})
    assert vars_section.get("FIRMS_DAY_RANGE") == "1"
    assert vars_section.get("FIRMS_MIN_CONFIDENCE") == "60"
    assert vars_section.get("FIRMS_MIN_FRP") == "6"
    assert vars_section.get("FIRMS_VEGETATION_ONLY") == "true"

    d1_databases = config.get("d1_databases", [])
    assert any(
        entry.get("binding") == "FIREWATCH_DB" and entry.get("database_name") == "firewatch-edge"
        for entry in d1_databases
    )


def test_d1_schema_contains_locations_and_indexes():
    schema = (_repo_root() / "d1" / "schema.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS locations" in schema
    assert "CREATE INDEX IF NOT EXISTS idx_locations_active" in schema
    assert "CREATE INDEX IF NOT EXISTS idx_locations_coords" in schema


def test_worker_is_cloudflare_native_without_backend_proxy_dependency():
    worker_js = (_repo_root() / "app" / "static" / "_worker.js").read_text(encoding="utf-8")

    assert "BACKEND_ORIGIN" not in worker_js
    assert "proxyLegacyRequest" not in worker_js
    assert 'const EDGE_API_ALIASES = [EDGE_API_PREFIX, "/api/v1"]' in worker_js
