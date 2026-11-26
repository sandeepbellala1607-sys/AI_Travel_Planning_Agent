"""Integration tests for FastAPI validation, SQLite persistence, and endpoints."""

from datetime import date, timedelta
from fastapi.testclient import TestClient
import pytest

from app.db import TravelDatabase
from app.main import app, set_db
from app.schemas import (
    AttractionCandidate,
    HotelCandidate,
    TransportCandidate,
    WeatherReport,
)


@pytest.fixture()
def client(tmp_path):
    db_file = tmp_path / "test_travel.db"
    db = TravelDatabase(str(db_file))
    set_db(db)
    with TestClient(app) as tc:
        yield tc
    set_db(None)


def test_health_and_swagger_docs(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert client.get("/docs").status_code == 200
    paths = openapi.json()["paths"]
    assert "/api/trips" in paths
    assert "/api/trips/{trip_id}/budget" in paths
    assert "/api/trips/{trip_id}/itinerary" in paths
    assert "/api/jobs/{job_id}" in paths


def test_trip_request_validation_rejects_invalid_inputs(client: TestClient):
    today = date.today()
    r1 = client.post(
        "/api/trips",
        json={
            "destination": "Goa",
            "start_date": (today - timedelta(days=1)).isoformat(),
            "end_date": (today + timedelta(days=2)).isoformat(),
            "budget": 20000,
            "currency": "INR",
            "adults": 2,
        },
    )
    assert r1.status_code == 422

    r2 = client.post(
        "/api/trips",
        json={
            "destination": "Goa",
            "start_date": (today + timedelta(days=2)).isoformat(),
            "end_date": (today + timedelta(days=2)).isoformat(),
            "budget": 20000,
            "currency": "INR",
            "adults": 2,
        },
    )
    assert r2.status_code == 422

    r3 = client.post(
        "/api/trips",
        json={
            "destination": "Goa",
            "start_date": (today + timedelta(days=2)).isoformat(),
            "end_date": (today + timedelta(days=5)).isoformat(),
            "budget": 0,
            "currency": "INR",
            "adults": 2,
        },
    )
    assert r3.status_code == 422

    r4 = client.post(
        "/api/trips",
        json={
            "destination": "Goa",
            "start_date": (today + timedelta(days=2)).isoformat(),
            "end_date": (today + timedelta(days=5)).isoformat(),
            "budget": 20000,
            "currency": "INR",
            "adults": 0,
        },
    )
    assert r4.status_code == 422


def test_create_plan_background_and_budget_endpoint(client: TestClient, monkeypatch):
    today = date.today()
    start = today + timedelta(days=2)
    end = today + timedelta(days=5)  # 3 nights / 4 days

    monkeypatch.setattr(
        "app.graph.workflow.search_attractions",
        lambda req, tt: [
            AttractionCandidate(
                name="Baga Beach",
                description="Popular North Goa beach",
                source_url="https://example.com/baga",
                category="beach",
                price_per_person=0.0,
                is_free=True,
            ),
            AttractionCandidate(
                name="Aguada Fort",
                description="Historic seaside fort",
                source_url="https://example.com/aguada",
                category="landmark",
                price_per_person=0.0,
                is_free=True,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_hotels",
        lambda req, tt: [
            HotelCandidate(
                name="Zostel Goa",
                description="Budget backpacker hostel",
                source_url="https://example.com/zostel-goa",
                tier="budget",
                price_per_night=1200.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_transport",
        lambda req, tt: [
            TransportCandidate(
                mode="Konkan Railway Express",
                description="Affordable round-trip train",
                source_url="https://example.com/konkan-rail",
                price_per_person=1400.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.fetch_destination_weather",
        lambda destination, start_date, end_date: WeatherReport(
            available=False,
            destination_name=destination,
            message="Mocked weather unavailable",
            daily_forecasts=[],
        ),
    )

    create_resp = client.post(
        "/api/trips",
        json={
            "destination": "Goa, India",
            "origin": "Pune, India",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "budget": 20000,
            "currency": "INR",
            "adults": 2,
            "children": 0,
            "preferences": ["beaches", "hostels", "street food"],
            "constraints": ["no late nights"],
        },
    )
    assert create_resp.status_code == 201
    trip_id = create_resp.json()["trip"]["id"]

    plan_resp = client.post(f"/api/trips/{trip_id}/plan")
    assert plan_resp.status_code == 200
    job_id = plan_resp.json()["job_id"]

    job_resp = client.get(f"/api/jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "completed"

    list_resp = client.get("/api/trips")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    itin_resp = client.get(f"/api/trips/{trip_id}/itinerary")
    assert itin_resp.status_code == 200
    itin = itin_resp.json()
    assert itin["trip_type"] == "budget"
    assert len(itin["days"]) == 4
    assert itin["cost_breakdown"]["fits_budget"] is True
    assert "does not book" in itin["disclaimer"].lower()

    budget_resp = client.get(f"/api/trips/{trip_id}/budget")
    assert budget_resp.status_code == 200
    budget_data = budget_resp.json()
    assert budget_data["trip_id"] == trip_id
    assert len(budget_data["cost_breakdown"]["line_items"]) == 6
    assert budget_data["cost_breakdown"]["fits_budget"] is True
