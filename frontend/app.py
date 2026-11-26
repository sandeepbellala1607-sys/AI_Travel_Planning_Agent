"""Streamlit frontend for the AI Travel Planning Agent."""

from datetime import date, timedelta
import os
import time
import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="AI Travel Planning Agent",
    page_icon="✈️",
    layout="wide",
)


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def render_itinerary(itinerary: dict) -> None:
    """Render structured itinerary, weather, source links, and Python-calculated budget."""
    st.info(itinerary.get("disclaimer", "Planning only — no bookings are made."))

    cost = itinerary["cost_breakdown"]
    if not cost["fits_budget"]:
        st.error(itinerary.get("budget_warning") or "Plan exceeds your target budget.")
    else:
        st.success(
            f"Fits within budget! Total estimated cost: {cost['total_cost']:.2f} {cost['currency']} "
            f"(Remaining: {cost['remaining_budget']:.2f} {cost['currency']})"
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trip Type", itinerary["trip_type"].title())
    col2.metric("Duration", f"{cost['nights']} nights / {cost['total_days']} days")
    col3.metric("Rooms Needed", str(cost["rooms_needed"]))
    col4.metric("Total Cost", f"{cost['total_cost']:.2f} {cost['currency']}")

    st.caption(f"Classification rationale: {itinerary['classification_rationale']}")

    # Adjustment history
    if itinerary.get("adjustment_history"):
        with st.expander("Budget Adjustment Steps Taken", expanded=False):
            for step in itinerary["adjustment_history"]:
                st.write(f"- {step}")

    # Weather section
    st.subheader("Weather Forecast (Open-Meteo)")
    weather = itinerary["weather_report"]
    if not weather["available"]:
        st.warning(weather["message"])
    else:
        st.write(
            f"**{weather['destination_name']}** "
            f"(Lat: {weather['latitude']}, Lon: {weather['longitude']}) — {weather['message']}"
        )
        fc_rows = weather.get("daily_forecasts") or []
        if fc_rows:
            st.dataframe(fc_rows, use_container_width=True)

    # Hotel & Transport
    st.subheader("Selected Accommodation & Transportation")
    h_col, t_col = st.columns(2)
    with h_col:
        hotel = itinerary.get("selected_hotel")
        if hotel:
            price_label = (
                f"{hotel['price_per_night']:.2f} {cost['currency']}/night"
                if hotel.get("price_per_night") is not None
                else "Estimated (price not found in search)"
            )
            st.markdown(f"**Hotel:** {hotel['name']}")
            st.write(f"Rate: {price_label}")
            st.write(hotel["description"])
            st.markdown(f"[Source Link]({hotel['source_url']})")

    with t_col:
        transport = itinerary.get("selected_transport")
        if transport:
            price_label = (
                f"{transport['price_per_person']:.2f} {cost['currency']}/traveller"
                if transport.get("price_per_person") is not None
                else "Estimated (fare not found in search)"
            )
            st.markdown(f"**Transport:** {transport['mode']}")
            st.write(f"Fare: {price_label}")
            st.write(transport["description"])
            st.markdown(f"[Source Link]({transport['source_url']})")

    # Cost Breakdown (Python calculated)
    st.subheader("Cost Breakdown (Calculated in Python)")
    for item in cost.get("line_items", []):
        est_tag = " [Estimated]" if item["is_estimated"] else ""
        st.write(
            f"- **{item['label']}{est_tag}:** {item['amount']:.2f} {cost['currency']} "
            f"*(Formula: {item['formula_note']})*"
        )

    if itinerary.get("assumptions"):
        with st.expander("Assumptions & Estimation Notes", expanded=True):
            for note in itinerary["assumptions"]:
                st.write(f"- {note}")

    # Day-by-Day Plan
    st.subheader("Day-Wise Itinerary")
    for day in itinerary.get("days", []):
        with st.expander(f"Day {day['day_number']} ({day['date']}) — {day['theme']}", expanded=True):
            st.caption(f"Weather: {day['weather_note']}")
            for act in day.get("activities", []):
                if act.get("unit_price") is not None:
                    price_txt = f"{act['unit_price']:.2f} {cost['currency']}"
                elif act.get("is_estimated"):
                    price_txt = f"{act['effective_unit_price']:.2f} {cost['currency']} [Estimated]"
                else:
                    price_txt = f"{act['effective_unit_price']:.2f} {cost['currency']} (Free)"
                st.markdown(
                    f"**{act['time_slot']} — {act['title']}** (`{act['category']}`) | "
                    f"Per-person ticket: {price_txt} | [Source]({act['source_url']})"
                )
                st.write(act["description"])

    # Validation report
    validation = itinerary.get("validation", {})
    if validation.get("warnings") or validation.get("errors"):
        with st.expander("Validation Report", expanded=False):
            for err in validation.get("errors", []):
                st.error(err)
            for warn in validation.get("warnings", []):
                st.warning(warn)


def main() -> None:
    st.title("AI Travel Planning Agent")
    st.caption(
        "LangGraph parallel research + deterministic Python budget calculation + Open-Meteo weather. "
        "Planning only — no bookings."
    )

    with st.sidebar:
        st.header("Saved Trips")
        if st.button("Refresh Saved Trips"):
            st.rerun()
        try:
            resp = httpx.get(f"{API_BASE_URL}/api/trips", timeout=5.0)
            if resp.status_code == 200:
                saved = resp.json()
                for item in saved[:15]:
                    label = f"{item['destination']} ({item['start_date']} → {item['end_date']})"
                    if st.button(label, key=f"trip_{item['id']}"):
                        st.session_state["selected_trip_id"] = item["id"]
            else:
                st.warning("Backend reachable, could not load trips.")
        except Exception:
            st.warning(f"Backend not reachable at {API_BASE_URL}. Start FastAPI first.")

    with st.form("trip_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            destination = st.text_input("Destination *", placeholder="e.g., Lisbon, Portugal")
            origin = st.text_input("Origin City (optional)", placeholder="e.g., London, UK")
            start_date = st.date_input("Start Date *", value=date.today() + timedelta(days=3))
            end_date = st.date_input("End Date *", value=date.today() + timedelta(days=6))
        with col_b:
            budget = st.number_input("Total Budget *", min_value=50.0, value=1200.0, step=50.0)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "INR"], index=0)
            adults = st.number_input("Adults *", min_value=1, value=2, step=1)
            children = st.number_input("Children", min_value=0, value=0, step=1)

        prefs_raw = st.text_input(
            "Preferences (comma-separated)",
            placeholder="e.g., museums, local food, walking tours",
        )
        constraints_raw = st.text_input(
            "Constraints (comma-separated)",
            placeholder="e.g., vegetarian, no late nights",
        )
        submitted = st.form_submit_button("Plan My Trip")

    if submitted:
        payload = {
            "destination": destination,
            "origin": origin,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "budget": float(budget),
            "currency": currency,
            "adults": int(adults),
            "children": int(children),
            "preferences": _split_csv(prefs_raw),
            "constraints": _split_csv(constraints_raw),
        }
        try:
            with st.spinner("Submitting trip request and running LangGraph workflow..."):
                create_resp = httpx.post(
                    f"{API_BASE_URL}/api/trips",
                    params={"auto_plan": True, "sync": False},
                    json=payload,
                    timeout=15.0,
                )
                if create_resp.status_code != 201:
                    st.error(f"Validation/API error: {create_resp.text}")
                    return

                data = create_resp.json()
                trip_id = data["trip"]["id"]
                job_id = data["job"]["job_id"]
                st.session_state["selected_trip_id"] = trip_id

                # Poll job until completed or failed
                for _ in range(90):
                    job_resp = httpx.get(f"{API_BASE_URL}/api/jobs/{job_id}", timeout=10.0)
                    job_data = job_resp.json()
                    if job_data["status"] == "completed":
                        break
                    if job_data["status"] == "failed":
                        st.error(f"Planning failed: {job_data.get('error')}")
                        return
                    time.sleep(1.0)
        except Exception as exc:
            st.error(f"Failed to contact FastAPI backend: {exc}")
            return

    selected_trip_id = st.session_state.get("selected_trip_id")
    if selected_trip_id:
        try:
            detail_resp = httpx.get(f"{API_BASE_URL}/api/trips/{selected_trip_id}", timeout=10.0)
            if detail_resp.status_code == 200:
                detail = detail_resp.json()
                if detail.get("itinerary"):
                    render_itinerary(detail["itinerary"])
                elif detail.get("latest_job"):
                    st.info(f"Job status: {detail['latest_job']['status']}")
        except Exception as exc:
            st.error(f"Error loading saved trip: {exc}")


if __name__ == "__main__":
    main()
