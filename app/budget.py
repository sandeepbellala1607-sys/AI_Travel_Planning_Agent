"""Deterministic Python cost, budget calculation, and option selection rules.

All monetary arithmetic MUST live here and never be performed by the LLM.
"""

from dataclasses import dataclass
from datetime import date
import math

from app.schemas import (
    CostBreakdown,
    CostLineItem,
    DayPlan,
    HotelCandidate,
    TimedActivity,
    TransportCandidate,
    TripRequestCreate,
    TripType,
)

DEFAULT_HOTEL_NIGHTLY_RATE: dict[TripType, float] = {
    "budget": 60.0,
    "family": 130.0,
    "adventure": 95.0,
}

DEFAULT_TRANSPORT_UNIT_RATE: dict[TripType, float] = {
    "budget": 80.0,
    "family": 140.0,
    "adventure": 120.0,
}

DEFAULT_DAILY_LOCAL_TRAVEL_RATE: dict[TripType, float] = {
    "budget": 10.0,
    "family": 18.0,
    "adventure": 15.0,
}

DEFAULT_DAILY_FOOD_RATE: dict[TripType, float] = {
    "budget": 25.0,
    "family": 38.0,
    "adventure": 32.0,
}

CONTINGENCY_BUFFER_RATE: dict[TripType, float] = {
    "budget": 0.05,
    "family": 0.10,
    "adventure": 0.10,
}

CURRENCY_FALLBACK_MULTIPLIER: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.0,
    "GBP": 0.85,
    "INR": 20.0,
}


def get_currency_multiplier(currency: str) -> float:
    """Return fallback estimate scale factor for supported currencies."""
    return CURRENCY_FALLBACK_MULTIPLIER.get(currency.upper(), 1.0)

FREE_ACTIVITY_CATEGORIES: set[str] = {
    "park",
    "beach",
    "landmark",
    "market",
    "walking",
    "viewpoint",
    "temple",
    "free",
}

DEFAULT_ACTIVITY_PRICE_BY_CATEGORY: dict[str, float] = {
    "museum": 18.0,
    "tour": 35.0,
    "adventure": 45.0,
    "theme_park": 50.0,
    "food": 20.0,
    "show": 30.0,
    "general": 20.0,
}

NIGHT_TRAVEL_KEYWORDS: tuple[str, ...] = (
    "night bus",
    "overnight",
    "red-eye",
    "sleeper bus",
    "late night",
)


@dataclass
class BudgetCalculationOutput:
    """Result of deterministic budget calculation plus updated day plans and notes."""

    cost_breakdown: CostBreakdown
    updated_days: list[DayPlan]
    assumptions: list[str]
    budget_warning: str | None


def compute_nights_and_days(start_date: date, end_date: date) -> tuple[int, int]:
    """Compute hotel nights and total calendar days covered by the trip."""
    nights = (end_date - start_date).days
    if nights <= 0:
        raise ValueError("end_date must be strictly after start_date")
    total_days = nights + 1
    return nights, total_days


def compute_rooms_needed(adults: int, children: int) -> int:
    """Compute hotel rooms needed assuming 2 adult-equivalents per room (child = 0.5)."""
    if adults < 1:
        raise ValueError("adults must be at least 1")
    if children < 0:
        raise ValueError("children cannot be negative")
    weighted_guests = adults + 0.5 * children
    return max(1, math.ceil(weighted_guests / 2.0))


def is_night_travel_option(text: str) -> bool:
    """Return True if a transport mode/description indicates forbidden night travel."""
    lower = text.lower()
    return any(keyword in lower for keyword in NIGHT_TRAVEL_KEYWORDS)


def select_hotel_option(
    hotels: list[HotelCandidate],
    trip_type: TripType,
    prefer_cheapest: bool = False,
) -> HotelCandidate | None:
    """Deterministic stay selection rule per trip type and budget mode."""
    if not hotels:
        return None
    fallback = DEFAULT_HOTEL_NIGHTLY_RATE[trip_type]

    if prefer_cheapest or trip_type == "budget":
        return min(
            hotels,
            key=lambda h: (
                0 if h.price_per_night is not None else 1,
                h.price_per_night if h.price_per_night is not None else fallback,
            ),
        )

    if trip_type == "family":
        def family_score(h: HotelCandidate) -> tuple[int, float]:
            text = f"{h.name} {h.description} {h.tier}".lower()
            family_bonus = 0 if any(k in text for k in ("family", "suite", "resort", "central")) else 1
            eff_price = h.price_per_night if h.price_per_night is not None else fallback
            return (family_bonus, eff_price)

        return min(hotels, key=family_score)

    def adventure_score(h: HotelCandidate) -> tuple[int, float]:
        text = f"{h.name} {h.description} {h.tier}".lower()
        adv_bonus = 0 if any(k in text for k in ("lodge", "camp", "hostel", "base", "trail")) else 1
        eff_price = h.price_per_night if h.price_per_night is not None else fallback
        return (adv_bonus, eff_price)

    return min(hotels, key=adventure_score)


def select_transport_option(
    transports: list[TransportCandidate],
    trip_type: TripType,
    constraints: list[str] | None = None,
    prefer_cheapest: bool = False,
) -> TransportCandidate | None:
    """Deterministic transport selection rule respecting night-travel safety constraints."""
    if not transports:
        return None
    fallback = DEFAULT_TRANSPORT_UNIT_RATE[trip_type]
    forbid_night = trip_type == "adventure" or any(
        "night" in c.lower() for c in (constraints or [])
    )

    eligible = [
        t
        for t in transports
        if not (forbid_night and is_night_travel_option(f"{t.mode} {t.description}"))
    ]
    pool = eligible or transports

    if prefer_cheapest or trip_type == "budget":
        return min(
            pool,
            key=lambda t: (
                0 if t.price_per_person is not None else 1,
                t.price_per_person if t.price_per_person is not None else fallback,
            ),
        )
    return pool[0]


def max_activities_for_day(
    trip_type: TripType,
    day_number: int,
    total_days: int,
    is_buffer_day: bool = False,
    prefer_free_only: bool = False,
) -> int:
    """Return allowed main activity count for a day under use-case rules."""
    if is_buffer_day:
        return 1
    if trip_type == "family":
        return 2 if (day_number in (1, total_days) or prefer_free_only) else 3
    if prefer_free_only or day_number in (1, total_days):
        return 2
    return 3


def resolve_activity_price(
    activity: TimedActivity, currency: str = "USD"
) -> tuple[float, bool, str]:
    """Determine effective per-person price and whether it is an estimate."""
    if activity.unit_price is not None:
        return round(float(activity.unit_price), 2), False, activity.price_note

    category = activity.category.strip().lower()
    note_lower = activity.price_note.lower()
    if category in FREE_ACTIVITY_CATEGORIES or "free" in note_lower:
        return 0.0, False, activity.price_note or "Listed as free/public access"

    scale = get_currency_multiplier(currency)
    fallback = round(
        DEFAULT_ACTIVITY_PRICE_BY_CATEGORY.get(
            category, DEFAULT_ACTIVITY_PRICE_BY_CATEGORY["general"]
        )
        * scale,
        2,
    )
    return (
        fallback,
        True,
        f"Estimated fallback ticket cost for category '{category or 'general'}' (price not found in search).",
    )


def calculate_trip_cost(
    request: TripRequestCreate,
    trip_type: TripType,
    selected_hotel: HotelCandidate | None,
    selected_transport: TransportCandidate | None,
    days: list[DayPlan],
) -> BudgetCalculationOutput:
    """Calculate exact trip costs across stay, transport, local travel, food, activities, and buffer."""
    nights, total_days = compute_nights_and_days(request.start_date, request.end_date)
    rooms_needed = compute_rooms_needed(request.adults, request.children)
    activity_travellers = request.adults + 0.5 * request.children
    transport_travellers = request.adults + 0.75 * request.children
    currency = request.currency
    scale = get_currency_multiplier(currency)

    assumptions: list[str] = [
        f"Trip duration: {nights} night(s) / {total_days} day(s).",
        f"Room allocation: {rooms_needed} room(s) for {request.adults} adult(s) and {request.children} child(ren).",
        "Children are weighted at 0.5x adult rate for activities, local travel, and food, and 0.75x for intercity transport.",
    ]

    if selected_hotel and selected_hotel.price_per_night is not None:
        nightly_rate = round(float(selected_hotel.price_per_night), 2)
        hotel_is_estimated = False
        hotel_label = f"Stay: {selected_hotel.name}"
    else:
        nightly_rate = round(DEFAULT_HOTEL_NIGHTLY_RATE[trip_type] * scale, 2)
        hotel_is_estimated = True
        hotel_name = selected_hotel.name if selected_hotel else "Fallback Accommodation"
        hotel_label = f"Stay (Estimated): {hotel_name}"
        assumptions.append(
            f"Stay nightly price was not found in search; estimated at {nightly_rate:.2f} {currency}/night "
            f"for a '{trip_type}' trip."
        )
    hotel_cost = round(nightly_rate * rooms_needed * nights, 2)

    if selected_transport and selected_transport.price_per_person is not None:
        unit_transport = round(float(selected_transport.price_per_person), 2)
        transport_is_estimated = False
        transport_label = f"Transport: {selected_transport.mode}"
    else:
        unit_transport = round(DEFAULT_TRANSPORT_UNIT_RATE[trip_type] * scale, 2)
        transport_is_estimated = True
        mode_name = selected_transport.mode if selected_transport else "Intercity Transport"
        transport_label = f"Transport (Estimated): {mode_name}"
        assumptions.append(
            f"Transport fare was not found in search; estimated at {unit_transport:.2f} {currency} per adult-equivalent."
        )
    transport_cost = round(unit_transport * transport_travellers, 2)

    local_travel_rate = round(DEFAULT_DAILY_LOCAL_TRAVEL_RATE[trip_type] * scale, 2)
    local_travel_cost = round(local_travel_rate * activity_travellers * total_days, 2)
    assumptions.append(
        f"Local travel estimated at {local_travel_rate:.2f} {currency}/day per adult-equivalent."
    )

    food_rate = round(DEFAULT_DAILY_FOOD_RATE[trip_type] * scale, 2)
    food_cost = round(food_rate * activity_travellers * total_days, 2)
    assumptions.append(
        f"Food & dining estimated at {food_rate:.2f} {currency}/day per adult-equivalent."
    )

    activities_cost = 0.0
    activities_has_estimates = False
    updated_days: list[DayPlan] = []
    for day in days:
        updated_activities: list[TimedActivity] = []
        for activity in day.activities:
            eff_price, is_est, note = resolve_activity_price(activity, currency=currency)
            if is_est:
                activities_has_estimates = True
            activities_cost += eff_price * activity_travellers
            updated_activities.append(
                activity.model_copy(
                    update={
                        "effective_unit_price": eff_price,
                        "is_estimated": is_est,
                        "price_note": note,
                    }
                )
            )
        updated_days.append(day.model_copy(update={"activities": updated_activities}))
    activities_cost = round(activities_cost, 2)
    if activities_has_estimates:
        assumptions.append(
            "One or more paid activities lacked listed ticket prices and use category fallback estimates."
        )

    subtotal = hotel_cost + transport_cost + local_travel_cost + food_cost + activities_cost
    buffer_pct = CONTINGENCY_BUFFER_RATE[trip_type]
    buffer_cost = round(subtotal * buffer_pct, 2)
    assumptions.append(
        f"Contingency buffer set at {int(buffer_pct * 100)}% of direct trip costs ({buffer_cost:.2f} {currency})."
    )

    total_cost = round(subtotal + buffer_cost, 2)
    budget = round(float(request.budget), 2)
    remaining_budget = round(budget - total_cost, 2)
    over_budget_amount = round(max(0.0, total_cost - budget), 2)
    fits_budget = total_cost <= budget

    line_items = [
        CostLineItem(
            category="stay",
            label=hotel_label,
            amount=hotel_cost,
            is_estimated=hotel_is_estimated,
            formula_note=f"{nightly_rate:.2f} x {rooms_needed} room(s) x {nights} night(s)",
        ),
        CostLineItem(
            category="transport",
            label=transport_label,
            amount=transport_cost,
            is_estimated=transport_is_estimated,
            formula_note=f"{unit_transport:.2f} x {transport_travellers:.2f} traveller units",
        ),
        CostLineItem(
            category="local_travel",
            label="Local Travel (Estimated)",
            amount=local_travel_cost,
            is_estimated=True,
            formula_note=f"{local_travel_rate:.2f} x {activity_travellers:.2f} traveller units x {total_days} day(s)",
        ),
        CostLineItem(
            category="food",
            label="Food & Dining (Estimated)",
            amount=food_cost,
            is_estimated=True,
            formula_note=f"{food_rate:.2f} x {activity_travellers:.2f} traveller units x {total_days} day(s)",
        ),
        CostLineItem(
            category="activities",
            label="Attractions & Activities",
            amount=activities_cost,
            is_estimated=activities_has_estimates,
            formula_note=f"Sum of per-activity prices x {activity_travellers:.2f} traveller units",
        ),
        CostLineItem(
            category="buffer",
            label=f"Contingency Buffer ({int(buffer_pct * 100)}%)",
            amount=buffer_cost,
            is_estimated=True,
            formula_note=f"{int(buffer_pct * 100)}% x {subtotal:.2f} direct subtotal",
        ),
    ]

    budget_warning: str | None = None
    if not fits_budget:
        budget_warning = (
            f"Over budget by {over_budget_amount:.2f} {currency}: total estimated cost is "
            f"{total_cost:.2f} {currency} against a target budget of {budget:.2f} {currency}. "
            "Consider increasing the budget, shortening the trip, or selecting lower-cost options."
        )

    breakdown = CostBreakdown(
        currency=currency,
        nights=nights,
        total_days=total_days,
        rooms_needed=rooms_needed,
        hotel_cost=hotel_cost,
        hotel_is_estimated=hotel_is_estimated,
        transport_cost=transport_cost,
        transport_is_estimated=transport_is_estimated,
        local_travel_cost=local_travel_cost,
        food_cost=food_cost,
        activities_cost=activities_cost,
        activities_has_estimates=activities_has_estimates,
        buffer_cost=buffer_cost,
        total_cost=total_cost,
        budget=budget,
        remaining_budget=remaining_budget,
        over_budget_amount=over_budget_amount,
        fits_budget=fits_budget,
        line_items=line_items,
    )
    return BudgetCalculationOutput(
        cost_breakdown=breakdown,
        updated_days=updated_days,
        assumptions=assumptions,
        budget_warning=budget_warning,
    )
