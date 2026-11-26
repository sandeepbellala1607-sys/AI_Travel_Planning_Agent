"""Workflow tests for LangGraph parallel execution, budget loop, validation rules, and search tool calling."""

from datetime import date, datetime, timedelta, timezone
from langchain_core.messages import AIMessage

from app.graph.workflow import (
    node_validate_itinerary,
    run_travel_planning_workflow,
)
from app.schemas import (
    AttractionCandidate,
    CostBreakdown,
    DayPlan,
    HotelCandidate,
    TimedActivity,
    TransportCandidate,
    TripRequestRecord,
    WeatherReport,
)
from app.tools.search import RawSearchResult, _extract_price_from_text, search_attractions


def _make_trip_record(
    budget: float,
    nights: int = 2,
    currency: str = "USD",
    destination: str = "Prague",
    origin: str = "Berlin",
    adults: int = 2,
    children: int = 0,
    preferences: list[str] | None = None,
    constraints: list[str] | None = None,
) -> TripRequestRecord:
    start = date.today() + timedelta(days=2)
    end = start + timedelta(days=nights)
    return TripRequestRecord(
        id="trip-test-1",
        destination=destination,
        origin=origin,
        start_date=start,
        end_date=end,
        budget=budget,
        currency=currency,
        adults=adults,
        children=children,
        preferences=preferences or ["walking", "history"],
        constraints=constraints or ["no late nights"],
        created_at=datetime.now(timezone.utc),
    )


def test_goa_four_day_inr_budget_scenario(monkeypatch):
    monkeypatch.setattr(
        "app.graph.workflow.search_hotels",
        lambda req, tt: [
            HotelCandidate(
                name="Beachfront Resort Goa",
                description="Pricey resort",
                source_url="https://example.com/resort",
                tier="family",
                price_per_night=5500.0,
            ),
            HotelCandidate(
                name="Anjuna Backpacker Hostel",
                description="Affordable hostel",
                source_url="https://example.com/hostel",
                tier="budget",
                price_per_night=1100.0,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_transport",
        lambda req, tt: [
            TransportCandidate(
                mode="Sleeper Bus / Express Train",
                description="Budget round trip",
                source_url="https://example.com/bus",
                price_per_person=1600.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_attractions",
        lambda req, tt: [
            AttractionCandidate(
                name="Baga Beach",
                description="Beachfront hangout",
                source_url="https://example.com/baga",
                category="beach",
                price_per_person=0.0,
                is_free=True,
            ),
            AttractionCandidate(
                name="Chapora Fort",
                description="Sunset viewpoint",
                source_url="https://example.com/chapora",
                category="viewpoint",
                price_per_person=0.0,
                is_free=True,
            ),
            AttractionCandidate(
                name="Anjuna Flea Market",
                description="Local market walk",
                source_url="https://example.com/anjuna",
                category="market",
                price_per_person=0.0,
                is_free=True,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.fetch_destination_weather",
        lambda destination, start_date, end_date: WeatherReport(
            available=True,
            destination_name="Goa, India",
            latitude=15.49,
            longitude=73.82,
            message="Mocked Goa weather",
            daily_forecasts=[],
        ),
    )

    goa_trip = _make_trip_record(
        budget=20000.0,
        nights=3,  # 4-day trip
        currency="INR",
        destination="Goa, India",
        origin="Pune, India",
        adults=2,
        children=0,
        preferences=["beaches", "hostels", "street food", "budget travel"],
        constraints=["no late nights"],
    )
    itinerary = run_travel_planning_workflow(goa_trip)

    assert itinerary.trip_type == "budget"
    assert len(itinerary.days) == 4
    assert itinerary.selected_hotel.name == "Anjuna Backpacker Hostel"
    assert itinerary.cost_breakdown.currency == "INR"
    assert itinerary.cost_breakdown.fits_budget is True
    assert itinerary.cost_breakdown.total_cost <= 20000.0
    assert itinerary.validation.is_valid is True


def test_workflow_adjusts_plan_when_initially_over_budget(monkeypatch):
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


def test_family_and_adventure_validation_rules(monkeypatch):
    monkeypatch.setattr(
        "app.graph.workflow.search_hotels",
        lambda req, tt: [
            HotelCandidate(
                name="Basecamp Lodge",
                description="Comfortable lodge",
                source_url="https://example.com/lodge",
                price_per_night=80.0,
            )
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_transport",
        lambda req, tt: [
            TransportCandidate(
                mode="Overnight Night Bus",
                description="Sleeper overnight ride",
                source_url="https://example.com/night-bus",
                price_per_person=20.0,
            ),
            TransportCandidate(
                mode="Daytime Rail",
                description="Scenic morning train",
                source_url="https://example.com/day-rail",
                price_per_person=50.0,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.search_attractions",
        lambda req, tt: [
            AttractionCandidate(
                name="Canyon Hike",
                description="Day hike",
                source_url="https://example.com/hike",
                category="adventure",
                price_per_person=20.0,
            ),
            AttractionCandidate(
                name="Science Museum",
                description="Indoor interactive museum",
                source_url="https://example.com/museum",
                category="museum",
                price_per_person=15.0,
            ),
        ],
    )
    monkeypatch.setattr(
        "app.graph.workflow.fetch_destination_weather",
        lambda destination, start_date, end_date: WeatherReport(
            available=True,
            destination_name="Prague",
            latitude=50.08,
            longitude=14.42,
            message="Mocked forecast",
            daily_forecasts=[],
        ),
    )

    family_trip = _make_trip_record(budget=2500.0, nights=2, adults=2, children=1, preferences=["museums"])
    family_itin = run_travel_planning_workflow(family_trip)
    assert family_itin.trip_type == "family"
    assert family_itin.validation.is_valid is True
    for day in family_itin.days:
        assert len(day.activities) <= 3
        assert day.rain_backup != ""

    adv_trip = _make_trip_record(budget=2500.0, nights=3, adults=2, children=0, preferences=["hiking", "adventure"])
    adv_itin = run_travel_planning_workflow(adv_trip)
    assert adv_itin.trip_type == "adventure"
    assert adv_itin.validation.is_valid is True
    assert any(day.is_buffer_day for day in adv_itin.days)
    assert adv_itin.selected_transport.mode == "Daytime Rail"


def test_validation_detects_rule_violations():
    trip = _make_trip_record(budget=1000.0, nights=2, adults=2, children=1)
    bad_state = {
        "request": trip,
        "trip_type": "family",
        "attractions": [
            AttractionCandidate(
                name="Park",
                description="Park",
                source_url="https://example.com/park",
                category="park",
            )
        ],
        "selected_hotel": HotelCandidate(
            name="H", description="D", source_url="https://example.com/h"
        ),
        "selected_transport": TransportCandidate(
            mode="Train", description="D", source_url="https://example.com/t"
        ),
        "weather_report": WeatherReport(
            available=True, destination_name="Prague", message="ok"
        ),
        "cost_breakdown": CostBreakdown(
            currency="USD",
            nights=2,
            total_days=3,
            rooms_needed=2,
            hotel_cost=200,
            hotel_is_estimated=False,
            transport_cost=100,
            transport_is_estimated=False,
            local_travel_cost=50,
            food_cost=100,
            activities_cost=0,
            activities_has_estimates=False,
            buffer_cost=45,
            total_cost=495,
            budget=1000,
            remaining_budget=505,
            over_budget_amount=0,
            fits_budget=True,
            line_items=[],
        ),
        "days": [
            DayPlan(
                day_number=1,
                date=trip.start_date,
                theme="Overpacked Day",
                weather_note="Clear",
                rain_backup="",
                activities=[
                    TimedActivity(
                        time_slot="09:00",
                        title="Unresearched Spot",
                        category="park",
                        description="Not from candidates",
                        source_url="https://example.com/unknown",
                    )
                    for _ in range(4)
                ],
            )
        ],
        "validation_attempts": 0,
    }
    res = node_validate_itinerary(bad_state)
    validation = res["validation"]
    assert validation.is_valid is False
    assert any("maximum of 3" in err for err in validation.errors)
    assert any("rain backup" in err for err in validation.errors)
    assert any("researched candidates" in err for err in validation.errors)


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
    assert _extract_price_from_text("Dorm bed starts at Rs. 1,250 per night") == 1250.0
    assert _extract_price_from_text("Entry ₹450") == 450.0
