"""Web search tools (DuckDuckGo & Tavily) with LangChain tool calling and candidate extraction."""

from dataclasses import dataclass
import json
import re
from typing import Any

from ddgs import DDGS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.llm import get_chat_model, has_llm_credentials
from app.schemas import (
    AttractionCandidate,
    HotelCandidate,
    TransportCandidate,
    TripRequestCreate,
    TripType,
)


@dataclass
class RawSearchResult:
    """Normalized raw web search hit."""

    title: str
    url: str
    snippet: str


class AttractionListOutput(BaseModel):
    """Structured extraction wrapper for attraction candidates."""

    items: list[AttractionCandidate] = Field(default_factory=list)


class HotelListOutput(BaseModel):
    """Structured extraction wrapper for hotel candidates."""

    items: list[HotelCandidate] = Field(default_factory=list)


class TransportListOutput(BaseModel):
    """Structured extraction wrapper for transport candidates."""

    items: list[TransportCandidate] = Field(default_factory=list)


def execute_web_search(
    query: str, max_results: int = 6, settings: Settings | None = None
) -> list[RawSearchResult]:
    """Run web search via Tavily (if configured) or DuckDuckGo (no API key needed)."""
    cfg = settings or get_settings()

    if (
        cfg.search_provider == "tavily"
        and cfg.tavily_api_key
        and not cfg.tavily_api_key.startswith("your_")
    ):
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=cfg.tavily_api_key)
            resp = client.search(query=query, max_results=max_results)
            return [
                RawSearchResult(
                    title=item.get("title", "").strip() or "Search Result",
                    url=item.get("url", "").strip(),
                    snippet=item.get("content", "").strip(),
                )
                for item in (resp.get("results") or [])
                if item.get("url")
            ]
        except Exception:
            # Fall back to DuckDuckGo if Tavily fails
            pass

    results: list[RawSearchResult] = []
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=max_results)
            for item in hits or []:
                url = (item.get("href") or item.get("url") or "").strip()
                if not url:
                    continue
                results.append(
                    RawSearchResult(
                        title=(item.get("title") or "Search Result").strip(),
                        url=url,
                        snippet=(item.get("body") or item.get("snippet") or "").strip(),
                    )
                )
    except Exception:
        return []
    return results


@tool
def web_search(query: str) -> str:
    """Search the web for travel attractions, hotels, or transportation options with source URLs."""
    hits = execute_web_search(query=query, max_results=5)
    serialized = [
        {"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits
    ]
    return json.dumps(serialized, ensure_ascii=False)


def _extract_price_from_text(text: str) -> float | None:
    """Extract an explicit numeric price only when clearly formatted with currency symbols."""
    match = re.search(r"(?:\$|USD\s*|EUR\s*|€|£|INR\s*|₹)\s*(\d{1,5}(?:[.,]\d{1,2})?)", text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        value = float(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def _run_tool_calling_search(
    llm: BaseChatModel, system_prompt: str, user_prompt: str
) -> list[RawSearchResult]:
    """Allow the LLM to choose search queries via tool calling and collect all raw hits."""
    llm_with_tools = llm.bind_tools([web_search])
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    collected: list[RawSearchResult] = []

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    for call in tool_calls[:3]:
        args = call.get("args") or {}
        query = str(args.get("query") or "").strip()
        if not query:
            continue
        hits = execute_web_search(query=query, max_results=5)
        collected.extend(hits)
        messages.append(
            ToolMessage(
                content=json.dumps(
                    [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits],
                    ensure_ascii=False,
                ),
                tool_call_id=call["id"],
            )
        )
    return collected


def _dedupe_hits(hits: list[RawSearchResult]) -> list[RawSearchResult]:
    seen: set[str] = set()
    unique: list[RawSearchResult] = []
    for hit in hits:
        if hit.url and hit.url not in seen:
            seen.add(hit.url)
            unique.append(hit)
    return unique


def _infer_category(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ("park", "garden", "beach", "lake", "promenade")):
        return "park"
    if any(w in lower for w in ("museum", "gallery", "palace", "castle")):
        return "museum"
    if any(w in lower for w in ("hike", "trail", "rafting", "kayak", "zipline", "climb")):
        return "adventure"
    if any(w in lower for w in ("market", "bazaar", "food", "street food", "cafe")):
        return "food"
    if any(w in lower for w in ("tour", "cruise", "excursion")):
        return "tour"
    return "general"


def search_attractions(
    request: TripRequestCreate,
    trip_type: TripType,
    llm: BaseChatModel | None = None,
) -> list[AttractionCandidate]:
    """Search and extract attraction candidates with source URLs; leave unfound prices as None."""
    active_llm = llm
    if active_llm is None and has_llm_credentials():
        active_llm = get_chat_model(temperature=0.1)

    hits: list[RawSearchResult] = []
    if active_llm is not None:
        try:
            hits = _run_tool_calling_search(
                active_llm,
                system_prompt=(
                    "You are a travel researcher. Call the web_search tool 1 to 2 times to find "
                    "top attractions and activities matching the user's destination, trip style, "
                    "preferences, and constraints."
                ),
                user_prompt=(
                    f"Destination: {request.destination}\n"
                    f"Trip type: {trip_type}\n"
                    f"Preferences: {', '.join(request.preferences) or 'sightseeing, local culture'}\n"
                    f"Constraints: {', '.join(request.constraints) or 'none'}"
                ),
            )
        except Exception:
            hits = []

    if not hits:
        pref_text = " ".join(request.preferences[:2]) if request.preferences else trip_type
        hits = execute_web_search(
            f"top things to do attractions in {request.destination} {pref_text}",
            max_results=6,
        )

    hits = _dedupe_hits(hits)[:8]
    if active_llm is not None and hits:
        try:
            extractor = active_llm.with_structured_output(AttractionListOutput)
            extracted = extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract up to 6 distinct attraction candidates ONLY from the provided search results. "
                            "Copy exact source_url values from the results. "
                            "CRITICAL: Set price_per_person to null unless an explicit numeric ticket price "
                            "appears in the snippet. NEVER guess or invent prices."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits],
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            valid = [item for item in extracted.items if item.source_url]
            if valid:
                return valid
        except Exception:
            pass

    candidates: list[AttractionCandidate] = []
    for hit in hits[:6]:
        combined = f"{hit.title} {hit.snippet}"
        price = _extract_price_from_text(combined)
        is_free = "free" in combined.lower() and price is None
        candidates.append(
            AttractionCandidate(
                name=hit.title[:100],
                description=hit.snippet[:260] or f"Attraction in {request.destination}",
                source_url=hit.url,
                category=_infer_category(combined),
                price_per_person=0.0 if is_free else price,
                is_free=is_free,
                price_note="Free entry mentioned" if is_free else ("" if price else "Price not found in search"),
            )
        )
    return candidates


def search_hotels(
    request: TripRequestCreate,
    trip_type: TripType,
    llm: BaseChatModel | None = None,
) -> list[HotelCandidate]:
    """Search and extract hotel candidates with source URLs; leave unfound nightly prices as None."""
    active_llm = llm
    if active_llm is None and has_llm_credentials():
        active_llm = get_chat_model(temperature=0.1)

    hits: list[RawSearchResult] = []
    if active_llm is not None:
        try:
            hits = _run_tool_calling_search(
                active_llm,
                system_prompt=(
                    "You are a travel accommodation researcher. Call web_search 1 to 2 times to find "
                    "hotels or stays (including budget-friendly options) in the destination."
                ),
                user_prompt=(
                    f"Destination: {request.destination}\n"
                    f"Trip type: {trip_type}\n"
                    f"Budget: {request.budget} {request.currency} for {request.adults} adults, {request.children} children\n"
                    f"Constraints: {', '.join(request.constraints) or 'none'}"
                ),
            )
        except Exception:
            hits = []

    if not hits:
        tier_hint = "budget hostel hotel" if trip_type == "budget" else "family hotel stay"
        hits = execute_web_search(
            f"best {tier_hint} in {request.destination} price per night",
            max_results=6,
        )

    hits = _dedupe_hits(hits)[:8]
    if active_llm is not None and hits:
        try:
            extractor = active_llm.with_structured_output(HotelListOutput)
            extracted = extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract up to 4 hotel/accommodation candidates ONLY from the provided search results. "
                            "Copy exact source_url values from the results. "
                            "CRITICAL: Set price_per_night to null unless an explicit nightly price "
                            "appears in the snippet. NEVER guess or invent prices."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits],
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            valid = [item for item in extracted.items if item.source_url]
            if valid:
                return valid
        except Exception:
            pass

    candidates: list[HotelCandidate] = []
    for hit in hits[:4]:
        combined = f"{hit.title} {hit.snippet}"
        price = _extract_price_from_text(combined)
        candidates.append(
            HotelCandidate(
                name=hit.title[:100],
                description=hit.snippet[:260] or f"Accommodation option in {request.destination}",
                source_url=hit.url,
                tier="budget" if "budget" in combined.lower() or "hostel" in combined.lower() else trip_type,
                price_per_night=price,
                price_note="" if price else "Nightly price not found in search",
            )
        )
    return candidates


def search_transport(
    request: TripRequestCreate,
    trip_type: TripType,
    llm: BaseChatModel | None = None,
) -> list[TransportCandidate]:
    """Search and extract transportation candidates with source URLs; leave unfound prices as None."""
    active_llm = llm
    if active_llm is None and has_llm_credentials():
        active_llm = get_chat_model(temperature=0.1)

    route_text = (
        f"from {request.origin} to {request.destination}"
        if request.origin
        else f"getting to and around {request.destination}"
    )

    hits: list[RawSearchResult] = []
    if active_llm is not None:
        try:
            hits = _run_tool_calling_search(
                active_llm,
                system_prompt=(
                    "You are a travel transport researcher. Call web_search 1 to 2 times to find "
                    "practical transportation options (train, bus, flight, or local transit pass)."
                ),
                user_prompt=(
                    f"Route: {route_text}\n"
                    f"Trip type: {trip_type}\n"
                    f"Travellers: {request.adults} adults, {request.children} children"
                ),
            )
        except Exception:
            hits = []

    if not hits:
        hits = execute_web_search(
            f"how to travel {route_text} train bus flight ticket cost",
            max_results=5,
        )

    hits = _dedupe_hits(hits)[:6]
    if active_llm is not None and hits:
        try:
            extractor = active_llm.with_structured_output(TransportListOutput)
            extracted = extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract up to 4 transportation candidates ONLY from the provided search results. "
                            "Copy exact source_url values from the results. "
                            "CRITICAL: Set price_per_person to null unless an explicit fare/ticket price "
                            "appears in the snippet. NEVER guess or invent prices."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits],
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            valid = [item for item in extracted.items if item.source_url]
            if valid:
                return valid
        except Exception:
            pass

    candidates: list[TransportCandidate] = []
    for hit in hits[:4]:
        combined = f"{hit.title} {hit.snippet}"
        price = _extract_price_from_text(combined)
        candidates.append(
            TransportCandidate(
                mode=hit.title[:80],
                description=hit.snippet[:260] or f"Transport option ({route_text})",
                source_url=hit.url,
                price_per_person=price,
                price_note="" if price else "Fare not found in search",
            )
        )
    return candidates
