"""SQLite persistence layer for trip requests, planning jobs, and itineraries."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator
import uuid

from app.schemas import Itinerary, JobRecord, JobStatus, TripRequestCreate, TripRequestRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TravelDatabase:
    """Lightweight SQLite store for trip requests, jobs, and itineraries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create required tables if they do not already exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trip_requests (
                    id TEXT PRIMARY KEY,
                    destination TEXT NOT NULL,
                    origin TEXT NOT NULL DEFAULT '',
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    budget REAL NOT NULL,
                    currency TEXT NOT NULL,
                    adults INTEGER NOT NULL,
                    children INTEGER NOT NULL,
                    preferences_json TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    trip_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (trip_id) REFERENCES trip_requests (id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS itineraries (
                    id TEXT PRIMARY KEY,
                    trip_id TEXT NOT NULL,
                    trip_type TEXT NOT NULL,
                    itinerary_json TEXT NOT NULL,
                    fits_budget INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (trip_id) REFERENCES trip_requests (id) ON DELETE CASCADE
                );
                """
            )

    def create_trip_request(self, payload: TripRequestCreate) -> TripRequestRecord:
        """Insert a validated trip request and return the stored record."""
        trip_id = str(uuid.uuid4())
        now_iso = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trip_requests (
                    id, destination, origin, start_date, end_date, budget,
                    currency, adults, children, preferences_json, constraints_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trip_id,
                    payload.destination,
                    payload.origin,
                    payload.start_date.isoformat(),
                    payload.end_date.isoformat(),
                    payload.budget,
                    payload.currency,
                    payload.adults,
                    payload.children,
                    json.dumps(payload.preferences),
                    json.dumps(payload.constraints),
                    now_iso,
                ),
            )
        return TripRequestRecord(
            id=trip_id,
            created_at=datetime.fromisoformat(now_iso),
            **payload.model_dump(),
        )

    def get_trip_request(self, trip_id: str) -> TripRequestRecord | None:
        """Fetch a single trip request by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trip_requests WHERE id = ?", (trip_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_trip(row)

    def list_trip_requests(self, limit: int = 50) -> list[TripRequestRecord]:
        """List saved trip requests ordered newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trip_requests ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_trip(row) for row in rows]

    def create_job(self, trip_id: str, status: JobStatus = "pending") -> JobRecord:
        """Create a planning job record for a trip request."""
        job_id = str(uuid.uuid4())
        now_iso = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, trip_id, status, error, created_at, updated_at)
                VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (job_id, trip_id, status, now_iso, now_iso),
            )
        ts = datetime.fromisoformat(now_iso)
        return JobRecord(
            job_id=job_id,
            trip_id=trip_id,
            status=status,
            error=None,
            created_at=ts,
            updated_at=ts,
        )

    def update_job_status(
        self, job_id: str, status: JobStatus, error: str | None = None
    ) -> JobRecord | None:
        """Update job status and timestamp."""
        now_iso = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, error, now_iso, job_id),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def get_latest_job_for_trip(self, trip_id: str) -> JobRecord | None:
        """Fetch the most recent job for a trip."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE trip_id = ? ORDER BY created_at DESC LIMIT 1",
                (trip_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def save_itinerary(self, itinerary: Itinerary) -> str:
        """Persist a generated itinerary in SQLite and return its row ID."""
        itinerary_id = str(uuid.uuid4())
        now_iso = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO itineraries (
                    id, trip_id, trip_type, itinerary_json, fits_budget, total_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    itinerary_id,
                    itinerary.trip_id,
                    itinerary.trip_type,
                    itinerary.model_dump_json(),
                    1 if itinerary.cost_breakdown.fits_budget else 0,
                    itinerary.cost_breakdown.total_cost,
                    now_iso,
                ),
            )
        return itinerary_id

    def get_latest_itinerary_for_trip(self, trip_id: str) -> Itinerary | None:
        """Fetch the latest stored itinerary for a trip ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT itinerary_json FROM itineraries WHERE trip_id = ? ORDER BY created_at DESC LIMIT 1",
                (trip_id,),
            ).fetchone()
        if row is None:
            return None
        return Itinerary.model_validate_json(row["itinerary_json"])

    @staticmethod
    def _row_to_trip(row: sqlite3.Row) -> TripRequestRecord:
        # Use model_construct to preserve historical records even if dates have since passed
        return TripRequestRecord.model_construct(
            id=row["id"],
            destination=row["destination"],
            origin=row["origin"],
            start_date=datetime.fromisoformat(row["start_date"]).date(),
            end_date=datetime.fromisoformat(row["end_date"]).date(),
            budget=float(row["budget"]),
            currency=row["currency"],
            adults=int(row["adults"]),
            children=int(row["children"]),
            preferences=json.loads(row["preferences_json"]),
            constraints=json.loads(row["constraints_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            trip_id=row["trip_id"],
            status=row["status"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
