"""Workflow tests for LangGraph parallel execution, budget loop, and search tool calling."""

from datetime import date, datetime, timedelta, timezone
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.graph.workflow import run_travel_planning_workflow
from app.schemas import (
    AttractionCandidate,
    HotelCandidate,
    TransportCandidate,
    TripRequestRecord,
    WeatherReport,
)
from app.tools.search import RawSearchResult, search_attractions


def _make_trip_record(budget: float, nights: int = 2) -> TripRequestRecord:
    start = date.today() + timedelta(days=2)
    end = start + timedelta(days=nights)
    return TripRequestRecord(
        id="trip-test-1",
        destination="Prague",
        origin="Berlin",
        start_date=start,
        end_date=end,
        budget=budget,
        currency="USD",
        adults=2,
        children=0,
        preferences=["walking", "history"],
        constraints=["no late nights"],
        created_at=datetime.now(timezone.utc),
    )


def test_workflow_adjusts_plan_when_initially_over_budget(monkeypatch):
    # Initial hotel ($300/nt) + transport ($150) exceeds budget ($600),
    # while cheaper hotel ($70/nt) + bus ($30) + free attractions fits ($70*2 + 30*2 + 35*2*3 = 140+60+210 = 410 <= 600)!
    monkeypatch.setattr(
        "app.graph.workflow.search_hotels",
        lambda req, tt: [
            HotelCandidate(
                name="Luxury Palace Hotel",
                description="Expensive option",
                source_url="https://example.com/lux",
                tier="family",
                price_per_night=300.0,
            ),
            HotelCandidate(
                name="Old Town Hostel",
                description="Budget stay",
                source_url="https://example.com/budget-stay",
                tier="budget",
                price_per_night=70.0,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_transport",
        lambda req, tt: [
            TransportCandidate(
                mode="Private Transfer",
                description="Pricey ride",
                source_url="https://example.com/private",
                price_per_person=150.0,
            ),
            TransportCandidate(
                mode="EuroCity Coach",
                description="Affordable bus",
                source_url="https://example.com/bus",
                price_per_person=30.0,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_attractions",
        lambda req, tt: [
            AttractionCandidate(
                name="Castle Tour",
                description="Guided tour",
                source_url="https://example.com/tour",
                category="tour",
                price_per_person=60.0,
            ),
            AttractionCandidate(
                name="Charles Bridge Walk",
                description="Free historic walk",
                source_url="https://example.com/bridge",
                category="walking",
                price_per_person=0.0,
                is_free=True,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.fetch_destination_weather",
        lambda destination, start_date, end_date: WeatherReport(
            available=True,
            destination_name="Prague, Czechia",
            latitude=50.08,
            longitude=14.42,
            message="Mocked forecast",
            daily_forecasts=[],
        ),
    )

    trip = _make_trip_record(budget=600.0, nights=2)
    itinerary = run_travel_planning_workflow(trip)

    assert len(itinerary.adjustment_history) >= 1
    assert itinerary.selected_hotel is not None
    assert itinerary.selected_hotel.name == "Old Town Hostel"
    assert itinerary.selected_transport is not None
    assert itinerary.selected_transport.mode == "EuroCity Coach"
    assert itinerary.cost_breakdown.fits_budget is True
    assert itinerary.validation.is_valid is True


def test_workflow_clearly_warns_when_budget_impossible(monkeypatch):
    monkeypatch.setattr(
        "app.graph.workflow.search_hotels",
        lambda req, tt: [
            HotelCandidate(
                name="Only Hotel",
                description="Hotel",
                source_url="https://example.com/h",
                price_per_night=250.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_transport",
        lambda req, tt: [
            TransportCandidate(
                mode="Train",
                description="Train",
                source_url="https://example.com/t",
                price_per_person=100.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_attractions",
        lambda req, tt: [
            AttractionCandidate(
                name="City Square",
                description="Free walk",
                source_url="https://example.com/sq",
                category="walking",
                price_per_person=0.0,
                is_free=True,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.fetch_destination_weather",
        lambda destination, start_date, end_date: WeatherReport(
            available=False,
            destination_name="Prague",
            message="Forecast unavailable",
        ),
    )

    trip = _make_trip_record(budget=100.0, nights=2)
    itinerary = run_travel_planning_workflow(trip)

    assert itinerary.cost_breakdown.fits_budget is False
    assert itinerary.budget_warning is not None
    assert "over budget" in itinerary.budget_warning.lower()
    assert any("budget" in w.lower() for w in itinerary.validation.warnings)


def test_search_attractions_leaves_unfound_price_none(monkeypatch):
    monkeypatch.setattr(
        "app.tools.search.execute_web_search",
        lambda query, max_results=5, settings=None: [
            RawSearchResult(
                title="National Gallery Prague",
                url="https://example.com/gallery",
                snippet="Art museum in the city center open daily.",
            )
        ],
    )

    # Simulate an LLM that emits a tool_call then fails structured output so raw hits are parsed cleanly
    class FakeToolLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "Prague museums"}, "id": "call_1"}
                ],
            )

        def with_structured_output(self, schema):
            raise RuntimeError("Force deterministic extractor check")

    req = _make_trip_record(budget=800.0)
    results = search_attractions(req, "budget", llm=FakeToolLLM())
    assert len(results) == 1
    assert results[0].source_url == "https://example.com/gallery"
    assert results[0].price_per_person is None
