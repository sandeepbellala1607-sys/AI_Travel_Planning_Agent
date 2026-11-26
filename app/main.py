"""FastAPI backend for the AI Travel Planning Agent."""

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status

from app.config import get_settings
from app.db import TravelDatabase
from app.graph.workflow import run_travel_planning_workflow
from app.schemas import (
    Itinerary,
    JobRecord,
    TripCreateResponse,
    TripDetailResponse,
    TripRequestCreate,
    TripRequestRecord,
)

_db_instance: TravelDatabase | None = None


def get_db() -> TravelDatabase:
    """Dependency returning the initialized SQLite database store."""
    global _db_instance
    if _db_instance is None:
        settings = get_settings()
        _db_instance = TravelDatabase(settings.database_path)
    return _db_instance


def set_db(db: TravelDatabase | None) -> None:
    """Override database instance (primarily for automated tests)."""
    global _db_instance
    _db_instance = db


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_db()
    yield


app = FastAPI(
    title="AI Travel Planning Agent API",
    version="1.0.0",
    description=(
        "LangGraph-powered travel planning API that researches attractions, hotels, "
        "transport, and Open-Meteo weather, calculates budgets deterministically in Python, "
        "and persists trip requests, jobs, and itineraries in SQLite. Planning only; no booking."
    ),
    lifespan=lifespan,
)


def _execute_planning_job(db: TravelDatabase, trip: TripRequestRecord, job_id: str) -> Itinerary | None:
    """Run the LangGraph workflow for a trip request and persist results."""
    db.update_job_status(job_id, "running")
    try:
        itinerary = run_travel_planning_workflow(trip)
        db.save_itinerary(itinerary)
        db.update_job_status(job_id, "completed")
        return itinerary
    except Exception as exc:
        db.update_job_status(job_id, "failed", error=str(exc))
        return None


@app.get("/health")
def health_check() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.post(
    "/api/trips",
    response_model=TripCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_trip_request(
    payload: TripRequestCreate,
    background_tasks: BackgroundTasks,
    auto_plan: bool = Query(
        default=False,
        description="When true, automatically triggers a planning job after saving the trip request.",
    ),
    sync: bool = Query(
        default=False,
        description="When true alongside auto_plan, executes the workflow synchronously before returning.",
    ),
    db: TravelDatabase = Depends(get_db),
) -> TripCreateResponse:
    """Validate and persist a new trip request, optionally starting itinerary generation."""
    trip = db.create_trip_request(payload)
    job: JobRecord | None = None
    if auto_plan:
        job = db.create_job(trip.id, status="pending")
        if sync:
            _execute_planning_job(db, trip, job.job_id)
            job = db.get_job(job.job_id)
        else:
            background_tasks.add_task(_execute_planning_job, db, trip, job.job_id)
    return TripCreateResponse(trip=trip, job=job)


@app.post("/api/trips/{trip_id}/plan", response_model=JobRecord)
def trigger_trip_plan(
    trip_id: str,
    background_tasks: BackgroundTasks,
    sync: bool = Query(
        default=False,
        description="Run planning workflow synchronously when true; otherwise run in background.",
    ),
    db: TravelDatabase = Depends(get_db),
) -> JobRecord:
    """Trigger the LangGraph travel planning workflow for an existing trip request."""
    trip = db.get_trip_request(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail=f"Trip request '{trip_id}' not found.")

    job = db.create_job(trip_id, status="pending")
    if sync:
        _execute_planning_job(db, trip, job.job_id)
        updated = db.get_job(job.job_id)
        assert updated is not None
        return updated

    background_tasks.add_task(_execute_planning_job, db, trip, job.job_id)
    return job


@app.get("/api/trips", response_model=list[TripRequestRecord])
def list_trips(
    limit: int = Query(default=50, ge=1, le=200),
    db: TravelDatabase = Depends(get_db),
) -> list[TripRequestRecord]:
    """List saved trip requests newest first."""
    return db.list_trip_requests(limit=limit)


@app.get("/api/trips/{trip_id}", response_model=TripDetailResponse)
def get_trip_detail(
    trip_id: str, db: TravelDatabase = Depends(get_db)
) -> TripDetailResponse:
    """Retrieve a saved trip request, its latest planning job, and its latest itinerary."""
    trip = db.get_trip_request(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail=f"Trip request '{trip_id}' not found.")
    return TripDetailResponse(
        trip=trip,
        latest_job=db.get_latest_job_for_trip(trip_id),
        itinerary=db.get_latest_itinerary_for_trip(trip_id),
    )


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job_status(job_id: str, db: TravelDatabase = Depends(get_db)) -> JobRecord:
    """Retrieve the current status of a planning job."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/api/trips/{trip_id}/itinerary", response_model=Itinerary)
def get_trip_itinerary(trip_id: str, db: TravelDatabase = Depends(get_db)) -> Itinerary:
    """Retrieve the latest generated itinerary for a trip request."""
    trip = db.get_trip_request(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail=f"Trip request '{trip_id}' not found.")
    itinerary = db.get_latest_itinerary_for_trip(trip_id)
    if itinerary is None:
        raise HTTPException(
            status_code=404,
            detail=f"No itinerary has been generated yet for trip '{trip_id}'.",
        )
    return itinerary
