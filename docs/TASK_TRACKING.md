# Task Tracking Sheet

**Project:** AI Travel Planning Agent  
**Target Stack:** Python, FastAPI, LangChain, LangGraph, Groq/OpenAI, DuckDuckGo/Tavily, Open-Meteo, SQLite, Streamlit

| Day | # | Task Description | Status | Notes / Blockers |
|---|---|---|---|---|
| Day 1 | 1 | Understand scope, choose ONE use case, define inputs/outputs, budget formula & architecture sketch | Done | Documented in `docs/ARCHITECTURE.md` |
| Day 1 | 2 | Set up backend project (FastAPI, venv, dependencies, env vars, SQLite) & validated Trip Request API | Done | `.venv` created, dependencies installed, FastAPI + SQLite implemented |
| Day 1 | 3 | Integrate LLM (Groq/OpenAI) via LangChain, trip type classification & structured itinerary schema | Done | Implemented in `app/llm.py` and `app/schemas.py` with heuristic fallback when API keys are unset |
| Day 1 | 4 | Integrate web search tool (DuckDuckGo / Tavily) with LLM tool calling & structured candidate extraction | Done | Implemented in `app/tools/search.py`; unfound prices stay `None` |
| Day 1 | 5 | Integrate Open-Meteo API for coordinates & daily forecast (with far-future unavailability handling) | Done | Implemented in `app/tools/weather.py` |
| Day 2 | 6 | Implement deterministic Python budget calculator & unit tests | Done | Implemented in `app/budget.py`, tested in `tests/test_budget.py` |
| Day 2 | 7 | Build LangGraph workflow (parallel research fan-out, budget check/adjust loop, itinerary validation) | Done | Implemented in `app/graph/workflow.py`, tested in `tests/test_workflow.py` |
| Day 2 | 8 | Wire workflow into FastAPI job runner & persist trip requests, jobs, and itineraries in SQLite | Done | Implemented in `app/main.py` and `app/db.py`, tested in `tests/test_api.py` |
| Day 3 | 9 | Build Streamlit frontend for trip submission, job polling, itinerary display & saved trips history | Done | Implemented in `frontend/app.py` |
| Day 3 | 10 | End-to-end testing, API docs, and README polish | Done | `README.md` documented; `pytest` suite passing |
