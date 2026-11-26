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


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_trip_request_validation_rejects_invalid_inputs(client: TestClient):
    today = date.today()
    # 1. Start date in the past
    r1 = client.post(
        "/api/trips",
        json={
            "destination": "Rome",
            "start_date": (today - timedelta(days=1)).isoformat(),
            "end_date": (today + timedelta(days=2)).isoformat(),
            "budget": 1000,
            "adults": 2,
        },
    )
    assert r1.status_code == 422

    # 2. End date not after start date
    r2 = client.post(
        "/api/trips",
        json={
            "destination": "Rome",
            "start_date": (today + timedelta(days=2)).isoformat(),
            "end_date": (today + timedelta(days=2)).isoformat(),
            "budget": 1000,
            "adults": 2,
        },
    )
    assert r2.status_code == 422

    # 3. Budget <= 0
    r3 = client.post(
        "/api/trips",
        json={
            "destination": "Rome",
            "start_date": (today + timedelta(days=2)).isoformat(),
            "end_date": (today + timedelta(days=4)).isoformat(),
            "budget": 0,
            "adults": 2,
        },
    )
    assert r3.status_code == 422

    # 4. Adults < 1
    r4 = client.post(
        "/api/trips",
        json={
            "destination": "Rome",
            "start_date": (today + timedelta(days=2)).isoformat(),
            "end_date": (today + timedelta(days=4)).isoformat(),
            "budget": 500,
            "adults": 0,
        },
    )
    assert r4.status_code == 422


def test_create_plan_and_persist_itinerary(client: TestClient, monkeypatch):
    today = date.today()
    start = today + timedelta(days=2)
    end = today + timedelta(days=4)

    monkeypatch.setattr(
        "app.graph.workflow.search_attractions",
        lambda req, tt: [
            AttractionCandidate(
                name="Colosseum",
                description="Historic arena",
                source_url="https://example.com/colosseum",
                category="museum",
                price_per_person=18.0,
            ),
            AttractionCandidate(
                name="Villa Borghese Park",
                description="Scenic park",
                source_url="https://example.com/park",
                category="park",
                price_per_person=0.0,
                is_free=True,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_hotels",
        lambda req, tt: [
            HotelCandidate(
                name="Hotel Roma",
                description="Central hotel",
                source_url="https://example.com/hotel-roma",
                tier="budget",
                price_per_night=90.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_transport",
        lambda req, tt: [
            TransportCandidate(
                mode="Regional Train",
                description="Rail pass",
                source_url="https://example.com/train",
                price_per_person=40.0,
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

    resp = client.post(
        "/api/trips?auto_plan=true&sync=true",
        json={
            "destination": "Rome, Italy",
            "origin": "Milan",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "budget": 1000,
            "currency": "USD",
            "adults": 2,
            "children": 0,
            "preferences": ["history", "parks"],
            "constraints": ["no late nights"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    trip_id = body["trip"]["id"]
    job_id = body["job"]["job_id"]
    assert body["job"]["status"] == "completed"

    # Verify job endpoint
    job_resp = client.get(f"/api/jobs/{job_id}")
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "completed"

    # Verify list & detail endpoints
    list_resp = client.get("/api/trips")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    itin_resp = client.get(f"/api/trips/{trip_id}/itinerary")
    assert itin_resp.status_code == 200
    itin = itin_resp.json()
    assert len(itin["days"]) == 3
    assert itin["cost_breakdown"]["fits_budget"] is True
    assert "does not book" in itin["disclaimer"].lower()
