"""Pydantic schemas for trip requests, research candidates, weather, budget, and itineraries."""

from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

TripType = Literal["budget", "family", "adventure"]
JobStatus = Literal["pending", "running", "completed", "failed"]


class TripRequestCreate(BaseModel):
    """Validated user input for creating a travel planning request."""

    destination: str = Field(..., min_length=1, max_length=200, description="Target city or region")
    origin: str = Field(default="", max_length=200, description="Optional starting city for transport")
    start_date: date = Field(..., description="Trip start date (YYYY-MM-DD, not in the past)")
    end_date: date = Field(..., description="Trip end date (YYYY-MM-DD, strictly after start_date)")
    budget: float = Field(..., gt=0, description="Total trip budget in selected currency")
    currency: str = Field(default="USD", min_length=3, max_length=5, description="ISO currency code")
    adults: int = Field(..., ge=1, le=50, description="Number of adult travellers (minimum 1)")
    children: int = Field(default=0, ge=0, le=50, description="Number of child travellers")
    preferences: list[str] = Field(default_factory=list, description="Preferred themes/activities")
    constraints: list[str] = Field(default_factory=list, description="Hard constraints or accessibility/diet rules")

    @field_validator("destination", "origin")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("destination")
    @classmethod
    def destination_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("destination must not be empty")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned.isalpha():
            raise ValueError("currency must be an alphabetic code such as USD, EUR, or INR")
        return cleaned

    @field_validator("preferences", "constraints")
    @classmethod
    def clean_string_list(cls, items: list[str]) -> list[str]:
        return [item.strip() for item in items if item and item.strip()]

    @field_validator("start_date")
    @classmethod
    def start_not_in_past(cls, value: date) -> date:
        if value < date.today():
            raise ValueError("start_date cannot be in the past")
        return value

    @model_validator(mode="after")
    def end_after_start(self) -> "TripRequestCreate":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be strictly after start_date")
        return self


class TripRequestRecord(TripRequestCreate):
    """Persisted trip request with server-assigned ID and timestamp."""

    id: str
    created_at: datetime


class TripClassification(BaseModel):
    """LLM classification of the trip profile."""

    trip_type: TripType
    rationale: str = Field(..., min_length=1)


class AttractionCandidate(BaseModel):
    """Researched attraction or activity option with mandatory source link."""

    name: str
    description: str
    source_url: str
    category: str = "general"
    price_per_person: float | None = Field(default=None, ge=0)
    is_free: bool = False
    price_note: str = ""


class HotelCandidate(BaseModel):
    """Researched accommodation option with mandatory source link."""

    name: str
    description: str
    source_url: str
    tier: str = "standard"
    price_per_night: float | None = Field(default=None, ge=0)
    price_note: str = ""


class TransportCandidate(BaseModel):
    """Researched transportation option with mandatory source link."""

    mode: str
    description: str
    source_url: str
    price_per_person: float | None = Field(default=None, ge=0)
    price_note: str = ""


class DailyWeather(BaseModel):
    """Daily forecast summary from Open-Meteo."""

    date: date
    weather_code: int
    condition: str
    temp_min_c: float
    temp_max_c: float
    precipitation_probability: int | None = None


class WeatherReport(BaseModel):
    """Destination weather report including coordinates and availability status."""

    available: bool
    destination_name: str
    latitude: float | None = None
    longitude: float | None = None
    message: str
    daily_forecasts: list[DailyWeather] = Field(default_factory=list)


class TimedActivity(BaseModel):
    """Single scheduled activity inside a day plan."""

    time_slot: str
    title: str
    category: str
    description: str
    source_url: str
    unit_price: float | None = Field(default=None, ge=0)
    effective_unit_price: float = Field(default=0.0, ge=0)
    is_estimated: bool = False
    price_note: str = ""


class DayPlan(BaseModel):
    """One day of the personalised itinerary."""

    day_number: int = Field(..., ge=1)
    date: date
    theme: str
    weather_note: str
    rain_backup: str = ""
    is_buffer_day: bool = False
    activities: list[TimedActivity] = Field(default_factory=list)


class CostLineItem(BaseModel):
    """Deterministic Python cost line item."""

    category: str
    label: str
    amount: float
    is_estimated: bool
    formula_note: str


class CostBreakdown(BaseModel):
    """Full deterministic cost breakdown computed in Python."""

    currency: str
    nights: int
    total_days: int
    rooms_needed: int
    hotel_cost: float
    hotel_is_estimated: bool
    transport_cost: float
    transport_is_estimated: bool
    local_travel_cost: float
    food_cost: float
    activities_cost: float
    activities_has_estimates: bool
    buffer_cost: float
    total_cost: float
    budget: float
    remaining_budget: float
    over_budget_amount: float
    fits_budget: bool
    line_items: list[CostLineItem] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Final validation checks against user requirements."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Itinerary(BaseModel):
    """Complete travel plan output stored in SQLite and returned by API."""

    trip_id: str
    destination: str
    origin: str = ""
    start_date: date
    end_date: date
    adults: int
    children: int
    trip_type: TripType
    classification_rationale: str
    weather_report: WeatherReport
    selected_hotel: HotelCandidate | None = None
    selected_transport: TransportCandidate | None = None
    days: list[DayPlan] = Field(default_factory=list)
    cost_breakdown: CostBreakdown
    assumptions: list[str] = Field(default_factory=list)
    budget_warning: str | None = None
    adjustment_history: list[str] = Field(default_factory=list)
    validation: ValidationResult
    disclaimer: str = (
        "Planning only: this AI travel planner creates itineraries and cost estimates; "
        "it does not book hotels, transportation, or attraction tickets."
    )


class JobRecord(BaseModel):
    """Background or synchronous planning job status record."""

    job_id: str
    trip_id: str
    status: JobStatus
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class TripCreateResponse(BaseModel):
    """Response when creating a trip request (and optionally starting planning)."""

    trip: TripRequestRecord
    job: JobRecord | None = None


class TripDetailResponse(BaseModel):
    """Combined trip request, latest job, and stored itinerary."""

    trip: TripRequestRecord
    latest_job: JobRecord | None = None
    itinerary: Itinerary | None = None


class TripBudgetResponse(BaseModel):
    """Dedicated budget and cost breakdown response for a planned trip."""

    trip_id: str
    trip_type: TripType
    cost_breakdown: CostBreakdown
    assumptions: list[str] = Field(default_factory=list)
    budget_warning: str | None = None
    adjustment_history: list[str] = Field(default_factory=list)

