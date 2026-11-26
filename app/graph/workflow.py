"""LangGraph workflow with parallel research, deterministic budget loop, and validation."""

from datetime import timedelta
import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.budget import (
    DEFAULT_HOTEL_NIGHTLY_RATE,
    DEFAULT_TRANSPORT_UNIT_RATE,
    calculate_trip_cost,
    compute_nights_and_days,
)
from app.llm import classify_trip_type, get_chat_model, has_llm_credentials
from app.schemas import (
    AttractionCandidate,
    CostBreakdown,
    DayPlan,
    HotelCandidate,
    Itinerary,
    TimedActivity,
    TransportCandidate,
    TripRequestRecord,
    TripType,
    ValidationResult,
    WeatherReport,
)
from app.tools.search import search_attractions, search_hotels, search_transport
from app.tools.weather import fetch_destination_weather

MAX_BUDGET_ADJUSTMENTS = 2
MAX_VALIDATION_RETRIES = 1


class TravelPlannerState(TypedDict, total=False):
    """Shared state passed through the LangGraph nodes."""

    request: TripRequestRecord
    trip_type: TripType
    classification_rationale: str

    attractions: list[AttractionCandidate]
    hotels: list[HotelCandidate]
    transport_options: list[TransportCandidate]
    weather_report: WeatherReport

    selected_hotel: HotelCandidate | None
    selected_transport: TransportCandidate | None
    days: list[DayPlan]

    cost_breakdown: CostBreakdown
    assumptions: list[str]
    budget_warning: str | None

    adjustment_attempts: int
    adjustment_history: Annotated[list[str], operator.add]
    validation_attempts: int
    validation: ValidationResult


class _DraftSlot(BaseModel):
    time_slot: str
    attraction_index: int = Field(..., ge=0)
    theme_note: str = ""


class _DraftDay(BaseModel):
    day_number: int
    theme: str
    slots: list[_DraftSlot] = Field(default_factory=list)


class _DraftItinerarySelection(BaseModel):
    selected_hotel_index: int = Field(default=0, ge=0)
    selected_transport_index: int = Field(default=0, ge=0)
    days: list[_DraftDay] = Field(default_factory=list)


def _weather_note_for_day(weather: WeatherReport, target_date: Any) -> str:
    if not weather.available:
        return weather.message
    for fc in weather.daily_forecasts:
        if fc.date == target_date:
            precip = (
                f", {fc.precipitation_probability}% precip chance"
                if fc.precipitation_probability is not None
                else ""
            )
            return f"{fc.condition} ({fc.temp_min_c:.1f}°C to {fc.temp_max_c:.1f}°C{precip})"
    return "Forecast not available for this specific date (outside 16-day forecast window)."


def _fallback_attraction(destination: str) -> AttractionCandidate:
    return AttractionCandidate(
        name=f"{destination} Historic Center Walking Tour",
        description=f"Self-guided walk through the main cultural landmarks of {destination}.",
        source_url="https://www.openstreetmap.org",
        category="walking",
        price_per_person=0.0,
        is_free=True,
        price_note="Free public walking route",
    )


def _fallback_hotel(destination: str, trip_type: TripType) -> HotelCandidate:
    return HotelCandidate(
        name=f"{destination} Central Stay ({trip_type.title()})",
        description=f"Standard accommodation option in {destination} (no live listing found).",
        source_url="https://www.openstreetmap.org",
        tier=trip_type,
        price_per_night=None,
        price_note="Nightly price not found; estimated in Python",
    )


def _fallback_transport(destination: str, origin: str) -> TransportCandidate:
    route = f"{origin} to {destination}" if origin else f"Local & arrival transit for {destination}"
    return TransportCandidate(
        mode=f"Transit ({route})",
        description=f"Standard bus/rail or regional transfer option for {route}.",
        source_url="https://www.openstreetmap.org",
        price_per_person=None,
        price_note="Fare not found; estimated in Python",
    )


def _build_deterministic_days(
    request: TripRequestRecord,
    attractions: list[AttractionCandidate],
    weather: WeatherReport,
    prefer_free_only: bool = False,
) -> list[DayPlan]:
    _, total_days = compute_nights_and_days(request.start_date, request.end_date)
    pool = attractions or [_fallback_attraction(request.destination)]
    if prefer_free_only:
        free_candidates = [
            a for a in pool if a.is_free or a.price_per_person == 0
        ]
        if free_candidates:
            pool = free_candidates
        else:
            pool = sorted(
                pool,
                key=lambda a: a.price_per_person if a.price_per_person is not None else 9999.0,
            )[:1]

    no_late_nights = any("late" in c.lower() for c in request.constraints)
    slots = ["09:30 - 12:00", "14:00 - 17:00"] if no_late_nights else ["09:30 - 12:00", "14:00 - 17:00", "18:30 - 20:30"]

    days: list[DayPlan] = []
    idx = 0
    for day_num in range(1, total_days + 1):
        day_date = request.start_date + timedelta(days=day_num - 1)
        weather_note = _weather_note_for_day(weather, day_date)
        activities: list[TimedActivity] = []
        # Keep 2 activities on travel days / budget adjustment, up to 3 otherwise
        count = 2 if (day_num in (1, total_days) or prefer_free_only) else min(len(slots), max(2, len(pool) // total_days + 1))
        for s_idx in range(min(count, len(slots))):
            cand = pool[idx % len(pool)]
            idx += 1
            activities.append(
                TimedActivity(
                    time_slot=slots[s_idx],
                    title=cand.name,
                    category="walking" if prefer_free_only and cand.price_per_person is None else cand.category,
                    description=cand.description,
                    source_url=cand.source_url,
                    unit_price=cand.price_per_person,
                    price_note=cand.price_note,
                )
            )
        days.append(
            DayPlan(
                day_number=day_num,
                date=day_date,
                theme=f"Day {day_num} in {request.destination}",
                weather_note=weather_note,
                activities=activities,
            )
        )
    return days


# --- LangGraph Nodes ---

def node_classify_trip(state: TravelPlannerState) -> dict[str, Any]:
    """Classify the trip into budget, family, or adventure."""
    request = state["request"]
    classification = classify_trip_type(request)
    return {
        "trip_type": classification.trip_type,
        "classification_rationale": classification.rationale,
        "adjustment_attempts": 0,
        "validation_attempts": 0,
    }


def node_research_attractions(state: TravelPlannerState) -> dict[str, Any]:
    """Parallel research branch 1: attractions via tool-calling web search."""
    request = state["request"]
    trip_type = state["trip_type"]
    results = search_attractions(request, trip_type)
    if not results:
        results = [_fallback_attraction(request.destination)]
    return {"attractions": results}


def node_research_hotels(state: TravelPlannerState) -> dict[str, Any]:
    """Parallel research branch 2: hotels via tool-calling web search."""
    request = state["request"]
    trip_type = state["trip_type"]
    results = search_hotels(request, trip_type)
    if not results:
        results = [_fallback_hotel(request.destination, trip_type)]
    return {"hotels": results}


def node_research_transport(state: TravelPlannerState) -> dict[str, Any]:
    """Parallel research branch 3: transportation via tool-calling web search."""
    request = state["request"]
    trip_type = state["trip_type"]
    results = search_transport(request, trip_type)
    if not results:
        results = [_fallback_transport(request.destination, request.origin)]
    return {"transport_options": results}


def node_fetch_weather(state: TravelPlannerState) -> dict[str, Any]:
    """Parallel research branch 4: Open-Meteo coordinates and forecast."""
    request = state["request"]
    report = fetch_destination_weather(
        destination=request.destination,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    return {"weather_report": report}


def node_draft_itinerary(state: TravelPlannerState) -> dict[str, Any]:
    """Combine parallel research outputs into a structured day-wise itinerary (no arithmetic)."""
    request = state["request"]
    trip_type = state["trip_type"]
    attractions = state.get("attractions") or [_fallback_attraction(request.destination)]
    hotels = state.get("hotels") or [_fallback_hotel(request.destination, trip_type)]
    transports = state.get("transport_options") or [_fallback_transport(request.destination, request.origin)]
    weather = state["weather_report"]
    _, total_days = compute_nights_and_days(request.start_date, request.end_date)

    selected_hotel = hotels[0]
    selected_transport = transports[0]
    days: list[DayPlan] | None = None

    if has_llm_credentials():
        try:
            llm = get_chat_model(temperature=0.2).with_structured_output(_DraftItinerarySelection)
            prompt_payload = {
                "destination": request.destination,
                "trip_type": trip_type,
                "total_days": total_days,
                "preferences": request.preferences,
                "constraints": request.constraints,
                "weather": weather.model_dump(mode="json"),
                "hotels": [
                    {"index": i, "name": h.name, "tier": h.tier, "price_per_night": h.price_per_night}
                    for i, h in enumerate(hotels)
                ],
                "transport": [
                    {"index": i, "mode": t.mode, "price_per_person": t.price_per_person}
                    for i, t in enumerate(transports)
                ],
                "attractions": [
                    {"index": i, "name": a.name, "category": a.category, "price_per_person": a.price_per_person}
                    for i, a in enumerate(attractions)
                ],
            }
            draft = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Select one hotel index, one transport index, and create a day-by-day schedule "
                            f"for all {total_days} days using ONLY the provided attraction indices. "
                            "Include 2 to 3 timed activities per day, respect user constraints and weather "
                            "(e.g., indoor museums on rainy days), and DO NOT compute or invent any money totals."
                        )
                    ),
                    HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=False)),
                ]
            )
            if 0 <= draft.selected_hotel_index < len(hotels):
                selected_hotel = hotels[draft.selected_hotel_index]
            if 0 <= draft.selected_transport_index < len(transports):
                selected_transport = transports[draft.selected_transport_index]

            if len(draft.days) == total_days:
                built_days: list[DayPlan] = []
                for idx, d in enumerate(draft.days):
                    day_date = request.start_date + timedelta(days=idx)
                    acts: list[TimedActivity] = []
                    for slot in d.slots:
                        cand = attractions[slot.attraction_index % len(attractions)]
                        acts.append(
                            TimedActivity(
                                time_slot=slot.time_slot or "10:00 - 12:30",
                                title=cand.name,
                                category=cand.category,
                                description=slot.theme_note or cand.description,
                                source_url=cand.source_url,
                                unit_price=cand.price_per_person,
                                price_note=cand.price_note,
                            )
                        )
                    if not acts:
                        cand = attractions[idx % len(attractions)]
                        acts.append(
                            TimedActivity(
                                time_slot="10:00 - 13:00",
                                title=cand.name,
                                category=cand.category,
                                description=cand.description,
                                source_url=cand.source_url,
                                unit_price=cand.price_per_person,
                                price_note=cand.price_note,
                            )
                        )
                    built_days.append(
                        DayPlan(
                            day_number=idx + 1,
                            date=day_date,
                            theme=d.theme or f"Day {idx + 1} in {request.destination}",
                            weather_note=_weather_note_for_day(weather, day_date),
                            activities=acts,
                        )
                    )
                days = built_days
        except Exception:
            days = None

    if days is None:
        days = _build_deterministic_days(request, attractions, weather)

    return {
        "selected_hotel": selected_hotel,
        "selected_transport": selected_transport,
        "days": days,
    }


def node_calculate_budget(state: TravelPlannerState) -> dict[str, Any]:
    """Pure Python node: compute exact cost breakdown and budget fit."""
    result = calculate_trip_cost(
        request=state["request"],
        trip_type=state["trip_type"],
        selected_hotel=state.get("selected_hotel"),
        selected_transport=state.get("selected_transport"),
        days=state.get("days") or [],
    )
    return {
        "cost_breakdown": result.cost_breakdown,
        "days": result.updated_days,
        "assumptions": result.assumptions,
        "budget_warning": result.budget_warning,
    }


def node_adjust_plan(state: TravelPlannerState) -> dict[str, Any]:
    """Adjust hotel, transport, and activities to lower cost when over budget."""
    attempt = state.get("adjustment_attempts", 0) + 1
    request = state["request"]
    trip_type = state["trip_type"]
    hotels = state.get("hotels") or []
    transports = state.get("transport_options") or []
    attractions = state.get("attractions") or []
    weather = state["weather_report"]
    breakdown = state["cost_breakdown"]

    notes: list[str] = [
        f"Adjustment #{attempt}: plan was over budget by {breakdown.over_budget_amount:.2f} {breakdown.currency}."
    ]

    # Pick cheapest hotel (explicit price preferred, otherwise fallback to budget tier rate)
    default_hotel_rate = DEFAULT_HOTEL_NIGHTLY_RATE[trip_type]
    selected_hotel = state.get("selected_hotel")
    if hotels:
        cheapest_hotel = min(
            hotels,
            key=lambda h: h.price_per_night if h.price_per_night is not None else default_hotel_rate,
        )
        selected_hotel = cheapest_hotel
        notes.append(f"Selected lowest-cost accommodation candidate: {cheapest_hotel.name}.")

    # On second attempt or if still needed, switch trip_type cost tier to 'budget' if hotel/transport had no listed prices
    effective_trip_type: TripType = trip_type
    if attempt >= 2 and trip_type != "budget":
        effective_trip_type = "budget"
        notes.append("Switched estimation tier to 'budget' rates for accommodation/meals/local transit.")

    # Pick cheapest transport
    default_trans_rate = DEFAULT_TRANSPORT_UNIT_RATE[effective_trip_type]
    selected_transport = state.get("selected_transport")
    if transports:
        cheapest_transport = min(
            transports,
            key=lambda t: t.price_per_person if t.price_per_person is not None else default_trans_rate,
        )
        selected_transport = cheapest_transport
        notes.append(f"Selected lowest-cost transport candidate: {cheapest_transport.mode}.")

    # Prioritize free / walking / lowest-cost attractions
    adjusted_days = _build_deterministic_days(
        request=request,
        attractions=attractions,
        weather=weather,
        prefer_free_only=True,
    )
    notes.append("Prioritized free/low-cost attractions and reduced paid activities per day.")

    return {
        "trip_type": effective_trip_type,
        "selected_hotel": selected_hotel,
        "selected_transport": selected_transport,
        "days": adjusted_days,
        "adjustment_attempts": attempt,
        "adjustment_history": notes,
    }


def node_validate_itinerary(state: TravelPlannerState) -> dict[str, Any]:
    """Validate final itinerary against user requirements, links, constraints, and budget."""
    request = state["request"]
    days = state.get("days") or []
    breakdown = state["cost_breakdown"]
    selected_hotel = state.get("selected_hotel")
    selected_transport = state.get("selected_transport")

    errors: list[str] = []
    warnings: list[str] = []

    _, expected_days = compute_nights_and_days(request.start_date, request.end_date)
    if len(days) != expected_days:
        errors.append(f"Expected {expected_days} day plans, got {len(days)}.")

    for idx, day in enumerate(days):
        expected_date = request.start_date + timedelta(days=idx)
        if day.date != expected_date:
            errors.append(f"Day {day.day_number} date {day.date} does not match {expected_date}.")
        if not day.activities:
            errors.append(f"Day {day.day_number} has no scheduled activities.")
        for act in day.activities:
            if not act.source_url:
                errors.append(f"Activity '{act.title}' on Day {day.day_number} is missing a source_url.")

    if selected_hotel and not selected_hotel.source_url:
        errors.append("Selected hotel is missing a source_url.")
    if selected_transport and not selected_transport.source_url:
        errors.append("Selected transport option is missing a source_url.")

    # Constraint checks
    for constraint in request.constraints:
        c_lower = constraint.lower()
        if "no late" in c_lower:
            for day in days:
                for act in day.activities:
                    if any(hour in act.time_slot for hour in ("21:", "22:", "23:")):
                        warnings.append(
                            f"Constraint '{constraint}' may conflict with '{act.time_slot}' on Day {day.day_number}."
                        )

    if not breakdown.fits_budget:
        warnings.append(
            state.get("budget_warning")
            or f"Itinerary exceeds budget of {breakdown.budget:.2f} {breakdown.currency}."
        )

    if not state["weather_report"].available:
        warnings.append(state["weather_report"].message)

    return {
        "validation_attempts": state.get("validation_attempts", 0) + 1,
        "validation": ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        ),
    }


def route_after_budget(state: TravelPlannerState) -> str:
    """Route to adjust_plan if over budget and retries remain; otherwise validate."""
    breakdown = state["cost_breakdown"]
    attempts = state.get("adjustment_attempts", 0)
    if not breakdown.fits_budget and attempts < MAX_BUDGET_ADJUSTMENTS:
        return "adjust_plan"
    return "validate_itinerary"


def route_after_validation(state: TravelPlannerState) -> str:
    """Retry drafting once if structural validation failed; otherwise finish."""
    validation = state["validation"]
    attempts = state.get("validation_attempts", 0)
    if not validation.is_valid and attempts <= MAX_VALIDATION_RETRIES:
        return "draft_itinerary"
    return END


def build_travel_planner_graph():
    """Compile the LangGraph StateGraph with parallel research and budget/validation loops."""
    graph = StateGraph(TravelPlannerState)

    graph.add_node("classify_trip", node_classify_trip)
    graph.add_node("research_attractions", node_research_attractions)
    graph.add_node("research_hotels", node_research_hotels)
    graph.add_node("research_transport", node_research_transport)
    graph.add_node("fetch_weather", node_fetch_weather)
    graph.add_node("draft_itinerary", node_draft_itinerary)
    graph.add_node("calculate_budget", node_calculate_budget)
    graph.add_node("adjust_plan", node_adjust_plan)
    graph.add_node("validate_itinerary", node_validate_itinerary)

    # Start -> classification
    graph.add_edge(START, "classify_trip")

    # Parallel fan-out from classify_trip to 4 independent research nodes
    graph.add_edge("classify_trip", "research_attractions")
    graph.add_edge("classify_trip", "research_hotels")
    graph.add_edge("classify_trip", "research_transport")
    graph.add_edge("classify_trip", "fetch_weather")

    # Fan-in to draft_itinerary
    graph.add_edge(
        ["research_attractions", "research_hotels", "research_transport", "fetch_weather"],
        "draft_itinerary",
    )

    # Budget check and adjustment loop
    graph.add_edge("draft_itinerary", "calculate_budget")
    graph.add_conditional_edges(
        "calculate_budget",
        route_after_budget,
        {
            "adjust_plan": "adjust_plan",
            "validate_itinerary": "validate_itinerary",
        },
    )
    graph.add_edge("adjust_plan", "calculate_budget")

    # Validation check
    graph.add_conditional_edges(
        "validate_itinerary",
        route_after_validation,
        {
            "draft_itinerary": "draft_itinerary",
            END: END,
        },
    )

    return graph.compile()


def run_travel_planning_workflow(request: TripRequestRecord) -> Itinerary:
    """Execute the compiled LangGraph workflow and assemble the final Itinerary."""
    compiled = build_travel_planner_graph()
    final_state = compiled.invoke({"request": request})

    return Itinerary(
        trip_id=request.id,
        destination=request.destination,
        origin=request.origin,
        start_date=request.start_date,
        end_date=request.end_date,
        adults=request.adults,
        children=request.children,
        trip_type=final_state["trip_type"],
        classification_rationale=final_state["classification_rationale"],
        weather_report=final_state["weather_report"],
        selected_hotel=final_state.get("selected_hotel"),
        selected_transport=final_state.get("selected_transport"),
        days=final_state["days"],
        cost_breakdown=final_state["cost_breakdown"],
        assumptions=final_state.get("assumptions", []),
        budget_warning=final_state.get("budget_warning"),
        adjustment_history=final_state.get("adjustment_history", []),
        validation=final_state["validation"],
    )
