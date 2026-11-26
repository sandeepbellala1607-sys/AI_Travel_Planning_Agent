# AI Travel Planning Agent

A personalised, multi-day travel planning agent built with **FastAPI**, **LangChain**, **LangGraph**, **SQLite**, **Open-Meteo**, **DuckDuckGo/Tavily**, and **Streamlit**.

> **Scope Notice:** This agent **only plans trips**. It does not book flights, hotels, transportation, or attraction tickets.

---

## Features
- **Validated Trip Requests**: Rejects past start dates, end dates not strictly after start dates, zero/negative budgets, and requests with fewer than 1 adult.
- **Trip Classification**: Classifies each trip into `budget`, `family`, or `adventure` using LangChain (Groq or OpenAI) with deterministic fallback.
- **Parallel LangGraph Research**: Runs 4 branches concurrently:
  1. Attractions search (tool-calling web search with source URLs)
  2. Hotels search (tool-calling web search with source URLs)
  3. Transportation search (tool-calling web search with source URLs)
  4. Weather check via **Open-Meteo** (geocoding + 16-day daily forecast, explicitly reporting unavailable when dates are too far ahead)
- **Deterministic Python Budget Engine**: All cost math runs strictly in `app/budget.py` (never in the LLM) and is covered by unit tests. Unknown search prices use explicit fallback estimates flagged as `[Estimated]`.
- **Budget Adjustment & Validation Loops**: If the plan exceeds budget, the LangGraph workflow loops up to 2 times to pick cheaper options/free activities; if it still exceeds budget, the output clearly displays the over-budget warning.
- **SQLite Persistence**: Stores all trip requests, job statuses, and generated itineraries.

---

## Project Structure
```text
AI_Travel_Planning_Agent/
├── app/
│   ├── __init__.py
│   ├── config.py          # Environment-based settings
│   ├── schemas.py         # Pydantic request, candidate, weather, cost & itinerary schemas
│   ├── db.py              # SQLite persistence layer
│   ├── llm.py             # Groq / OpenAI LangChain setup & trip classification
│   ├── budget.py          # Deterministic Python budget & cost calculator
│   ├── main.py            # FastAPI backend & job execution endpoints
│   ├── tools/
│   │   ├── search.py      # DuckDuckGo / Tavily search with LLM tool calling
│   │   └── weather.py     # Open-Meteo geocoding & daily forecast client
│   └── graph/
│       └── workflow.py    # LangGraph StateGraph (parallel research + budget/validation loops)
├── frontend/
│   └── app.py             # Streamlit UI
├── tests/
│   ├── test_budget.py
│   ├── test_weather.py
│   ├── test_api.py
│   └── test_workflow.py
├── docs/
│   ├── ARCHITECTURE.md    # Use case, inputs/outputs, budget formula & workflow diagram
│   └── TASK_TRACKING.md   # Daily task tracking sheet
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API key(s):
```powershell
Copy-Item .env.example .env
```

- `LLM_PROVIDER`: `groq` (default) or `openai`
- `GROQ_API_KEY`: Your Groq API key (free tier supported)
- `OPENAI_API_KEY`: Your OpenAI API key (if `LLM_PROVIDER=openai`)
- `SEARCH_PROVIDER`: `duckduckgo` (default, no key required) or `tavily`
- `TAVILY_API_KEY`: Required only if `SEARCH_PROVIDER=tavily`
- `DATABASE_PATH`: Path to SQLite file (default `data/travel_planner.db`)

*(Never commit `.env` or API keys to Git.)*

---

## Running the Application

### Option 1: Start Backend & Frontend Together (Recommended)
```powershell
.\.venv\Scripts\python.exe run_app.py
```

### Option 2: Start FastAPI & Streamlit in Separate Terminals
Terminal 1 (FastAPI Backend):
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
Interactive OpenAPI docs are available at `http://127.0.0.1:8000/docs`.

Terminal 2 (Streamlit Frontend):
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (`{"status": "ok"}`) |
| `POST` | `/api/trips` | Validate & save a trip request. Optional query params: `auto_plan=true`, `sync=true/false`. |
| `POST` | `/api/trips/{trip_id}/plan` | Start LangGraph planning job (`sync=true` to block until done, default async background task). |
| `GET` | `/api/trips` | List saved trip requests (newest first). |
| `GET` | `/api/trips/{trip_id}` | Get trip request, latest job status, and latest itinerary. |
| `GET` | `/api/jobs/{job_id}` | Poll job status (`pending`, `running`, `completed`, `failed`). |
| `GET` | `/api/trips/{trip_id}/itinerary` | Get latest generated itinerary for a trip. |
| `GET` | `/api/trips/{trip_id}/budget` | Get deterministic Python cost breakdown (`stay`, `transport`, `local_travel`, `food`, `activities`, `buffer`) and assumptions. |

### Example `POST /api/trips?auto_plan=true&sync=true` Payload (Budget Travel Planner — 4-Day Goa Trip)
```json
{
  "destination": "Goa, India",
  "origin": "Pune, India",
  "start_date": "2026-12-01",
  "end_date": "2026-12-04",
  "budget": 20000,
  "currency": "INR",
  "adults": 2,
  "children": 0,
  "preferences": ["beaches", "hostels", "street food", "budget travel"],
  "constraints": ["no late nights"]
}
```

---

## Running Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

