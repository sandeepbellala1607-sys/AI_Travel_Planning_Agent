"""Unit tests for deterministic Python budget and cost calculations."""

from datetime import date, timedelta
import pytest

from app.budget import (
    calculate_trip_cost,
    compute_nights_and_days,
    compute_rooms_needed,
)
from app.schemas import (
    DayPlan,
    HotelCandidate,
    TimedActivity,
    TransportCandidate,
    TripRequestCreate,
)


def _sample_request(
    days_ahead: int = 2,
    nights: int = 3,
    budget: float = 1500.0,
    adults: int = 2,
    children: int = 0,
) -> TripRequestCreate:
    start = date.today() + timedelta(days=days_ahead)
    end = start + timedelta(days=nights)
    return TripRequestCreate(
        destination="Lisbon",
        origin="Madrid",
        start_date=start,
        end_date=end,
        budget=budget,
        currency="USD",
        adults=adults,
        children=children,
        preferences=["museums", "food"],
        constraints=[],
    )


def test_nights_and_rooms_computation():
    start = date.today() + timedelta(days=5)
    end = start + timedelta(days=4)
    assert compute_nights_and_days(start, end) == (4, 5)

    assert compute_rooms_needed(1, 0) == 1
    assert compute_rooms_needed(2, 0) == 1
    assert compute_rooms_needed(2, 1) == 2
    assert compute_rooms_needed(2, 2) == 2
    assert compute_rooms_needed(3, 0) == 2


def test_exact_budget_calculation_with_listed_prices():
    req = _sample_request(nights=3, budget=1500.0, adults=2, children=0)
    hotel = HotelCandidate(
        name="Lisbon Budget Inn",
        description="Central stay",
        source_url="https://example.com/hotel",
        tier="budget",
        price_per_night=100.0,
    )
    transport = TransportCandidate(
        mode="Direct Train",
        description="Rail ticket",
        source_url="https://example.com/train",
        price_per_person=75.0,
    )
    days = [
        DayPlan(
            day_number=1,
            date=req.start_date,
            theme="Arrival",
            weather_note="Clear",
            activities=[
                TimedActivity(
                    time_slot="10:00 - 12:00",
                    title="Tile Museum",
                    category="museum",
                    description="Museum visit",
                    source_url="https://example.com/museum",
                    unit_price=15.0,
                ),
                TimedActivity(
                    time_slot="14:00 - 16:00",
                    title="City Park",
                    category="park",
                    description="Public park walk",
                    source_url="https://example.com/park",
                    unit_price=None,
                ),
            ],
        )
    ]

    result = calculate_trip_cost(req, "budget", hotel, transport, days)
    cb = result.cost_breakdown

    # hotel: 100 * 1 room * 3 nights = 300
    assert cb.hotel_cost == 300.0
    assert cb.hotel_is_estimated is False
    # transport: 75 * 2 adults = 150
    assert cb.transport_cost == 150.0
    assert cb.transport_is_estimated is False
    # activities: (15 + 0) * 2 adults = 30
    assert cb.activities_cost == 30.0
    assert cb.activities_has_estimates is False
    # meals/local (budget=35): 35 * 2 adults * 4 days = 280
    assert cb.meals_local_cost == 280.0
    assert cb.total_cost == 760.0
    assert cb.fits_budget is True
    assert cb.remaining_budget == 740.0
    assert cb.over_budget_amount == 0.0
    assert result.budget_warning is None


def test_missing_prices_use_estimates_and_flag_clearly():
    req = _sample_request(nights=2, budget=2000.0, adults=2, children=1)
    hotel = HotelCandidate(
        name="Mystery Hotel",
        description="No price listed",
        source_url="https://example.com/h",
        price_per_night=None,
    )
    transport = TransportCandidate(
        mode="Bus",
        description="No fare listed",
        source_url="https://example.com/t",
        price_per_person=None,
    )
    days = [
        DayPlan(
            day_number=1,
            date=req.start_date,
            theme="Explore",
            weather_note="Sunny",
            activities=[
                TimedActivity(
                    time_slot="10:00 - 12:00",
                    title="History Museum",
                    category="museum",
                    description="Museum without price",
                    source_url="https://example.com/m",
                    unit_price=None,
                )
            ],
        )
    ]

    result = calculate_trip_cost(req, "family", hotel, transport, days)
    cb = result.cost_breakdown

    assert cb.hotel_is_estimated is True
    assert cb.transport_is_estimated is True
    assert cb.activities_has_estimates is True
    assert result.updated_days[0].activities[0].is_estimated is True
    assert result.updated_days[0].activities[0].effective_unit_price == 18.0
    # rooms: ceil((2 + 0.5)/2) = 2 rooms * 2 nights * 130 (family) = 520
    assert cb.rooms_needed == 2
    assert cb.hotel_cost == 520.0
    # transport: 140 * (2 + 0.75*1) = 140 * 2.75 = 385
    assert cb.transport_cost == 385.0
    # activities: 18 * 2.5 = 45
    assert cb.activities_cost == 45.0
    # meals: 55 * 2.5 * 3 days = 412.5
    assert cb.meals_local_cost == 412.5
    assert cb.total_cost == 1362.5


def test_over_budget_reports_warning_explicitly():
    req = _sample_request(nights=3, budget=300.0, adults=2, children=0)
    hotel = HotelCandidate(
        name="Grand Hotel",
        description="Upscale",
        source_url="https://example.com/grand",
        price_per_night=200.0,
    )
    transport = TransportCandidate(
        mode="Flight",
        description="Round trip",
        source_url="https://example.com/flight",
        price_per_person=150.0,
    )

    result = calculate_trip_cost(req, "budget", hotel, transport, [])
    cb = result.cost_breakdown

    # hotel 600 + transport 300 + meals 280 = 1180 > 300
    assert cb.total_cost == 1180.0
    assert cb.fits_budget is False
    assert cb.over_budget_amount == 880.0
    assert cb.remaining_budget == -880.0
    assert result.budget_warning is not None
    assert "880.00 USD" in result.budget_warning
