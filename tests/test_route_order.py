from fastapi.routing import APIRoute

from app.main import app


def test_no_route_is_shadowed_by_an_earlier_one():
    # Starlette matches in declaration order, so "/fires/{fire_id}" declared
    # first swallows "/fires/recent".
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    shadowed = [
        (later.path, earlier.path)
        for i, later in enumerate(routes)
        for earlier in routes[:i]
        if later.methods & earlier.methods and earlier.path_regex.match(later.path)
    ]
    assert shadowed == []


def test_recent_alerts_route_is_reachable(client):
    response = client.get("/api/v1/fires/recent")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
