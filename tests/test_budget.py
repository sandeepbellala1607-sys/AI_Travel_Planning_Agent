"""Unit tests for deterministic Python budget calculations and option selection rules."""

from datetime import date, timedelta

from app.budget import (
    calculate_trip_cost,
    compute_nights_and_days,
    compute_rooms_needed,
    max_activities_for_day,
    select_hotel_option,
    select_transport_option,
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
    currency: str = "USD",
    destination: str = "Lisbon",
    adults: int = 2,
    children: int = 0,
) -> TripRequestCreate:
    start = date.today() + timedelta(days=days_ahead)
    end = start + timedelta(days=nights)
    return TripRequestCreate(
        destination=destination,
        origin="Madrid",
        start_date=start,
        end_date=end,
        budget=budget,
        currency=currency,
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


def test_exact_budget_calculation_all_six_categories():
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

    assert cb.hotel_cost == 300.0
    assert cb.hotel_is_estimated is False
    assert cb.transport_cost == 150.0
    assert cb.transport_is_estimated is False
    assert cb.local_travel_cost == 80.0
    assert cb.food_cost == 200.0
    assert cb.activities_cost == 30.0
    assert cb.activities_has_estimates is False
    assert cb.buffer_cost == 38.0
    assert cb.total_cost == 798.0
    assert len(cb.line_items) == 6
    assert {item.category for item in cb.line_items} == {
        "stay",
        "transport",
        "local_travel",
        "food",
        "activities",
        "buffer",
    }
    assert cb.fits_budget is True
    assert cb.remaining_budget == 702.0
    assert cb.over_budget_amount == 0.0
    assert result.budget_warning is None


def test_inr_budget_calculation_for_goa_trip():
    # Two college friends, 4-day (3 nights) Goa trip with INR 20,000 budget
    req = _sample_request(
        nights=3,
        budget=20000.0,
        currency="INR",
        destination="Goa, India",
        adults=2,
        children=0,
    )
    hotel = HotelCandidate(
        name="Anjuna Backpacker Hostel",
        description="Shared twin room",
        source_url="https://example.com/goa-hostel",
        tier="budget",
        price_per_night=1200.0,
    )
    transport = TransportCandidate(
        mode="Overnight Train / Sleeper Coach",
        description="Round trip rail fare",
        source_url="https://example.com/goa-train",
        price_per_person=1500.0,
    )
    days = [
        DayPlan(
            day_number=1,
            date=req.start_date,
            theme="North Goa Beaches",
            weather_note="Sunny",
            activities=[
                TimedActivity(
                    time_slot="10:00 - 13:00",
                    title="Baga & Calangute Beach",
                    category="beach",
                    description="Free beach visit",
                    source_url="https://example.com/baga",
                    unit_price=0.0,
                )
            ],
        )
    ]

    result = calculate_trip_cost(req, "budget", hotel, transport, days)
    cb = result.cost_breakdown

    # stay: 1200 * 1 * 3 = 3600; transport: 1500 * 2 = 3000;
    # local_travel: 200 * 2 * 4 = 1600; food: 500 * 2 * 4 = 4000;
    # activities: 0; subtotal = 12200; buffer 5% = 610 -> total = 12810 <= 20000
    assert cb.currency == "INR"
    assert cb.hotel_cost == 3600.0
    assert cb.transport_cost == 3000.0
    assert cb.local_travel_cost == 1600.0
    assert cb.food_cost == 4000.0
    assert cb.buffer_cost == 610.0
    assert cb.total_cost == 12810.0
    assert cb.fits_budget is True
    assert cb.remaining_budget == 7190.0


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
    assert cb.rooms_needed == 2
    assert cb.hotel_cost == 520.0
    assert cb.transport_cost == 385.0
    assert cb.local_travel_cost == 135.0
    assert cb.food_cost == 285.0
    assert cb.activities_cost == 45.0
    assert cb.buffer_cost == 137.0
    assert cb.total_cost == 1507.0


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

    assert cb.total_cost == 1239.0
    assert cb.fits_budget is False
    assert cb.over_budget_amount == 939.0
    assert cb.remaining_budget == -939.0
    assert result.budget_warning is not None
    assert "939.00 USD" in result.budget_warning


def test_option_selection_rules():
    hotels = [
        HotelCandidate(
            name="Boutique Plaza",
            description="Central family suite",
            source_url="https://example.com/1",
            price_per_night=150.0,
        ),
        HotelCandidate(
            name="Budget Hostel",
            description="Cheap hostel",
            source_url="https://example.com/2",
            price_per_night=45.0,
        ),
    ]
    assert select_hotel_option(hotels, "budget").name == "Budget Hostel"
    assert select_hotel_option(hotels, "family").name == "Boutique Plaza"
    assert select_hotel_option(hotels, "family", prefer_cheapest=True).name == "Budget Hostel"

    transports = [
        TransportCandidate(
            mode="Overnight Night Bus",
            description="Cheap sleeper overnight coach",
            source_url="https://example.com/night",
            price_per_person=25.0,
        ),
        TransportCandidate(
            mode="Morning Express Train",
            description="Daytime rail",
            source_url="https://example.com/day",
            price_per_person=60.0,
        ),
    ]
    chosen_adv = select_transport_option(transports, "adventure", prefer_cheapest=True)
    assert chosen_adv.mode == "Morning Express Train"

    assert max_activities_for_day("family", 2, 4) == 3
    assert max_activities_for_day("adventure", 2, 4, is_buffer_day=True) == 1
