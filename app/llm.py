"""LangChain LLM configuration and trip classification module."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.budget import get_currency_multiplier
from app.config import Settings, get_settings
from app.schemas import TripClassification, TripRequestCreate


def has_llm_credentials(settings: Settings | None = None) -> bool:
    """Return True if an API key is configured for the selected LLM provider."""
    cfg = settings or get_settings()
    if cfg.llm_provider == "openai":
        return bool(cfg.openai_api_key and not cfg.openai_api_key.startswith("your_"))
    return bool(cfg.groq_api_key and not cfg.groq_api_key.startswith("your_"))


def get_chat_model(settings: Settings | None = None, temperature: float = 0.2) -> BaseChatModel:
    """Instantiate the configured LangChain chat model (Groq or OpenAI)."""
    cfg = settings or get_settings()
    model_name = cfg.effective_llm_model()

    if cfg.llm_provider == "openai":
        if not has_llm_credentials(cfg):
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        return ChatOpenAI(
            api_key=cfg.openai_api_key,
            model=model_name,
            temperature=temperature,
        )

    if not has_llm_credentials(cfg):
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return ChatGroq(
        api_key=cfg.groq_api_key,
        model=model_name,
        temperature=temperature,
    )


def heuristic_classify_trip(request: TripRequestCreate) -> TripClassification:
    """Deterministic fallback classification when LLM credentials are unavailable."""
    joined_prefs = " ".join(request.preferences + request.constraints).lower()
    nights = (request.end_date - request.start_date).days
    travellers = request.adults + request.children
    per_person_per_night = request.budget / max(1, nights * travellers)
    normalized_ppn = per_person_per_night / get_currency_multiplier(request.currency)

    adventure_keywords = ("hike", "hiking", "trek", "adventure", "rafting", "camp", "climb", "kayak", "ski")
    if any(word in joined_prefs for word in adventure_keywords):
        return TripClassification(
            trip_type="adventure",
            rationale="Classified as 'adventure' based on outdoor/active preferences (heuristic fallback).",
        )
    if request.children > 0 or "family" in joined_prefs or "kid" in joined_prefs:
        return TripClassification(
            trip_type="family",
            rationale="Classified as 'family' because children or family preferences are present (heuristic fallback).",
        )
    budget_keywords = ("budget", "cheap", "hostel", "backpack", "college", "affordable")
    if normalized_ppn < 200 or any(word in joined_prefs for word in budget_keywords):
        return TripClassification(
            trip_type="budget",
            rationale="Classified as 'budget' based on budget-per-traveller target or cost preferences (heuristic fallback).",
        )
    return TripClassification(
        trip_type="family",
        rationale="Classified as 'family' as balanced leisure default (heuristic fallback).",
    )


def classify_trip_type(
    request: TripRequestCreate, llm: BaseChatModel | None = None
) -> TripClassification:
    """Classify a trip request into 'budget', 'family', or 'adventure' using LangChain."""
    if llm is None:
        if not has_llm_credentials():
            return heuristic_classify_trip(request)
        llm = get_chat_model(temperature=0.0)

    structured_llm = llm.with_structured_output(TripClassification)
    system_prompt = (
        "You are a travel planning classifier. Classify the user's trip strictly into one of "
        "three categories: 'budget', 'family', or 'adventure'. "
        "Choose 'family' if children are travelling or family-friendly pacing is requested; "
        "'adventure' if outdoor/active/adrenaline experiences dominate; "
        "'budget' if cost minimization or a tight budget per person/day dominates."
    )
    user_prompt = (
        f"Destination: {request.destination}\n"
        f"Dates: {request.start_date} to {request.end_date}\n"
        f"Travellers: {request.adults} adults, {request.children} children\n"
        f"Total Budget: {request.budget} {request.currency}\n"
        f"Preferences: {', '.join(request.preferences) or 'none'}\n"
        f"Constraints: {', '.join(request.constraints) or 'none'}"
    )
    try:
        return structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except Exception:
        return heuristic_classify_trip(request)

