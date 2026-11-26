"""Deterministic Python cost and budget calculation module.

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

DEFAULT_DAILY_MEALS_LOCAL_RATE: dict[TripType, float] = {
    "budget": 35.0,
    "family": 55.0,
    "adventure": 45.0,
}

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


def resolve_activity_price(activity: TimedActivity) -> tuple[float, bool, str]:
    """Determine effective per-person price and whether it is an estimate."""
    if activity.unit_price is not None:
        return round(float(activity.unit_price), 2), False, activity.price_note

    category = activity.category.strip().lower()
    note_lower = activity.price_note.lower()
    if category in FREE_ACTIVITY_CATEGORIES or "free" in note_lower:
        return 0.0, False, activity.price_note or "Listed as free/public access"

    fallback = DEFAULT_ACTIVITY_PRICE_BY_CATEGORY.get(
        category, DEFAULT_ACTIVITY_PRICE_BY_CATEGORY["general"]
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
    """Calculate exact trip costs in Python and flag any estimates or budget overruns."""
    nights, total_days = compute_nights_and_days(request.start_date, request.end_date)
    rooms_needed = compute_rooms_needed(request.adults, request.children)
    activity_travellers = request.adults + 0.5 * request.children
    transport_travellers = request.adults + 0.75 * request.children
    currency = request.currency

    assumptions: list[str] = [
        f"Trip duration: {nights} night(s) / {total_days} day(s).",
        f"Hotel room allocation: {rooms_needed} room(s) for {request.adults} adult(s) and {request.children} child(ren).",
        "Children are weighted at 0.5x adult rate for activities/meals and 0.75x for intercity transport.",
    ]

    # 1. Hotel cost
    if selected_hotel and selected_hotel.price_per_night is not None:
        nightly_rate = round(float(selected_hotel.price_per_night), 2)
        hotel_is_estimated = False
        hotel_label = f"Accommodation: {selected_hotel.name}"
    else:
        nightly_rate = DEFAULT_HOTEL_NIGHTLY_RATE[trip_type]
        hotel_is_estimated = True
        hotel_name = selected_hotel.name if selected_hotel else "Fallback Accommodation"
        hotel_label = f"Accommodation (Estimated): {hotel_name}"
        assumptions.append(
            f"Hotel nightly price was not found in search; estimated at {nightly_rate:.2f} {currency}/night "
            f"for a '{trip_type}' trip."
        )
    hotel_cost = round(nightly_rate * rooms_needed * nights, 2)

    # 2. Transport cost
    if selected_transport and selected_transport.price_per_person is not None:
        unit_transport = round(float(selected_transport.price_per_person), 2)
        transport_is_estimated = False
        transport_label = f"Transportation: {selected_transport.mode}"
    else:
        unit_transport = DEFAULT_TRANSPORT_UNIT_RATE[trip_type]
        transport_is_estimated = True
        mode_name = selected_transport.mode if selected_transport else "Intercity Transport"
        transport_label = f"Transportation (Estimated): {mode_name}"
        assumptions.append(
            f"Transport price was not found in search; estimated at {unit_transport:.2f} {currency} per adult-equivalent."
        )
    transport_cost = round(unit_transport * transport_travellers, 2)

    # 3. Activities cost
    activities_cost = 0.0
    activities_has_estimates = False
    updated_days: list[DayPlan] = []
    for day in days:
        updated_activities: list[TimedActivity] = []
        for activity in day.activities:
            eff_price, is_est, note = resolve_activity_price(activity)
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

    # 4. Daily meals & local transit allowance (explicit estimate)
    daily_meal_rate = DEFAULT_DAILY_MEALS_LOCAL_RATE[trip_type]
    meals_local_cost = round(daily_meal_rate * activity_travellers * total_days, 2)
    assumptions.append(
        f"Meals & local transit allowance estimated at {daily_meal_rate:.2f} {currency}/day per adult-equivalent."
    )

    # 5. Totals
    total_cost = round(hotel_cost + transport_cost + activities_cost + meals_local_cost, 2)
    budget = round(float(request.budget), 2)
    remaining_budget = round(budget - total_cost, 2)
    over_budget_amount = round(max(0.0, total_cost - budget), 2)
    fits_budget = total_cost <= budget

    line_items = [
        CostLineItem(
            category="hotel",
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
            category="activities",
            label="Attractions & Activities",
            amount=activities_cost,
            is_estimated=activities_has_estimates,
            formula_note=f"Sum of per-activity prices x {activity_travellers:.2f} traveller units",
        ),
        CostLineItem(
            category="meals_local",
            label="Meals & Local Transit (Estimated)",
            amount=meals_local_cost,
            is_estimated=True,
            formula_note=f"{daily_meal_rate:.2f} x {activity_travellers:.2f} traveller units x {total_days} day(s)",
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
        activities_cost=activities_cost,
        activities_has_estimates=activities_has_estimates,
        meals_local_cost=meals_local_cost,
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
