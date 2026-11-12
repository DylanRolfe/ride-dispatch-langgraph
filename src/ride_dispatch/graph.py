"""LangGraph: availability -> pickup_minutes -> eligibility -> rank_drivers (one OpenRouter call picks ONE driver).

Every threshold comes from config/settings.json through settings.load(); none is typed in this file.
"""
import json
import operator
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.ride_dispatch import settings

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"  # free tier; override with OPENROUTER_MODEL
FALLBACK_MODEL = "openrouter/free"  # OpenRouter's free-model router

# Fields sent to the model per candidate driver.
DRIVER_FIELDS = [
    "driver_id",
    "straight_line_km",
    "road_pickup_minutes",
    "driver_river_side",
    "driver_lat",
    "driver_lng",
    "status_updated_at",
    "route_note",
]
# Request context sent to the model. rider_id is deliberately NOT included.
REQUEST_FIELDS = [
    "requested_at",
    "pickup_zone",
    "pickup_lat",
    "pickup_lng",
    "pickup_river_side",
    "dropoff_zone",
    "is_bridge_case",
    "distance_time_disagree",
]
# The only shape the model may answer with.
REPLY_KEYS = {"driver_id", "reason"}


class State(TypedDict, total=False):
    request_id: str                  # input: the ride request being dispatched
    request: dict                    # input: that request's row from ride_requests.csv
    drivers: pd.DataFrame            # input: candidate drivers
    raining: bool                    # input: True applies the rain pickup margin
    available_drivers: pd.DataFrame  # node 1 (availability) output
    fastest_drivers: pd.DataFrame    # node 2 (pickup_minutes) output
    eligible_drivers: pd.DataFrame   # node 3 (eligibility) output
    result: dict                     # node 4 (rank_drivers) output: the one chosen driver
    steps: Annotated[list, operator.add]  # each node appends its name, so run order can be checked


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["request_id", *DRIVER_FIELDS])


def availability(state: State) -> State:
    """Node 1: only drivers tagged 'available' come through (on_trip / offline are cut)."""
    drivers = state["drivers"]
    if drivers.empty or "driver_status" not in drivers.columns:  # zero candidates / no data at all
        return {"available_drivers": _empty(), "steps": ["availability"]}
    return {"available_drivers": drivers[drivers["driver_status"] == "available"], "steps": ["availability"]}


def pickup_minutes(state: State) -> State:
    """Node 2: drop rows with no road_pickup_minutes, then keep the fastest `max_candidates` per request."""
    df = state["available_drivers"]
    if df.empty:
        return {"fastest_drivers": _empty(), "steps": ["pickup_minutes"]}
    keep = settings.load()["max_candidates"]
    df = df[df["road_pickup_minutes"].notna()]
    df = df.sort_values("road_pickup_minutes", kind="stable").groupby("request_id", sort=False).head(int(keep))
    return {"fastest_drivers": df, "steps": ["pickup_minutes"]}


def eligibility(state: State) -> State:
    """Node 3: keep drivers whose status was updated at most `max_status_age_minutes` before the request.

    A status stamped after the request time counts as fresh. An unreadable timestamp counts as not fresh.
    """
    df = state["fastest_drivers"]
    if df.empty:
        return {"eligible_drivers": _empty(), "steps": ["eligibility"]}
    limit = settings.load()["max_status_age_minutes"]
    requested = pd.to_datetime(state["request"]["requested_at"], utc=True)
    updated = pd.to_datetime(df["status_updated_at"], utc=True, format="ISO8601", errors="coerce")
    age_min = (requested - updated).dt.total_seconds() / 60
    return {"eligible_drivers": df[age_min <= limit], "steps": ["eligibility"]}


def _clean(value):
    """Make a pandas/numpy value JSON-safe (NaN -> None, numpy -> python)."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def make_checklist(eligible: pd.DataFrame, raining: bool, margin: float) -> list[dict]:
    """The drivers the model may pick from, fastest road pickup first.

    Normally that is every eligible driver. When it is raining, only drivers whose road_pickup_minutes
    is at most `margin` minutes slower than the fastest one stay on the list.
    """
    rows = eligible.sort_values("road_pickup_minutes", kind="stable")
    candidates = [{k: _clean(row[k]) for k in DRIVER_FIELDS} for _, row in rows.iterrows()]
    if raining and candidates:
        cutoff = candidates[0]["road_pickup_minutes"] + margin
        candidates = [c for c in candidates if c["road_pickup_minutes"] <= cutoff]
    return candidates


def build_prompt(request: dict, checklist: list[dict], raining: bool, margin: float) -> str:
    context = {k: _clean(request.get(k)) for k in REQUEST_FIELDS}
    weather = (f"It is raining, so only drivers within {margin:g} minutes of the fastest road pickup are listed.\n"
               if raining else "")
    return (
        f"Ride request context:\n{json.dumps(context, indent=1)}\n\n"
        f"{weather}"
        f"Checklist of drivers you may choose from:\n{json.dumps(checklist, indent=1)}\n\n"
        "Pick the single best driver for this ride. Prefer the lowest road_pickup_minutes (the drive time to "
        "the rider) unless a route_note makes that time unreliable. The driver_id MUST be one from the "
        "checklist above. Reply with JSON only, in exactly this shape and nothing else: "
        '{"driver_id": "<one id from the checklist>", "reason": "<one short line>"}'
    )


def _ask_model(prompt: str) -> dict | None:
    """Call the free OpenRouter model. Returns {"reply": <parsed JSON>, model info} or None if every attempt fails."""
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    for attempt_model in dict.fromkeys([model, FALLBACK_MODEL]):  # configured free model, then free router
        for attempt in range(3):
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                    json={"model": attempt_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
                    timeout=90,
                )
                if resp.status_code == 429:  # free tier rate limit
                    time.sleep(3 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                start = text.find("{")
                parsed, _ = json.JSONDecoder().raw_decode(text[start:]) if start != -1 else (None, 0)
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
                continue  # network/server error, odd response shape, or malformed JSON -> retry
            if isinstance(parsed, dict):
                # generation_id is OpenRouter's own request id: it can be looked up on the Activity page / API
                return {"reply": parsed, "model": attempt_model,
                        "generation_id": data.get("id"), "served_model": data.get("model")}
    return None  # caller falls back to the fastest road pickup


def check_reply(reply, checklist_ids: set, max_chars: int):
    """Accept the model's reply only if it has exactly {driver_id, reason} and the id is on the checklist.

    Returns (choice, None) when accepted, or (None, why) when rejected.
    """
    if not isinstance(reply, dict) or set(reply) != REPLY_KEYS:
        return None, "bad_shape"
    driver_id, reason = reply["driver_id"], reply["reason"]
    if not isinstance(driver_id, str) or not isinstance(reason, str) or not reason.strip():
        return None, "bad_shape"
    if driver_id not in checklist_ids:  # anything outside the checklist is rejected
        return None, "not_on_checklist"
    return {"driver_id": driver_id, "reason": " ".join(reason.split())[: int(max_chars)]}, None


def rank_drivers(state: State) -> State:
    """Node 4: one OpenRouter call picks ONE driver from the checklist; anything else falls back to the fastest."""
    cfg = settings.load()
    request_id, request = state["request_id"], state["request"]
    raining = bool(state.get("raining", False))
    margin = cfg["rain_pickup_margin_minutes"]

    eligible = state["eligible_drivers"]
    eligible = eligible[eligible["request_id"] == request_id]
    if eligible.empty:  # nobody left after availability / pickup_minutes / eligibility: model is not called
        return {"steps": ["rank_drivers"], "result": {
            "request_id": request_id,
            "status": "pending_approval",
            "message": "Pending approval",
        }}

    checklist = make_checklist(eligible, raining, margin)
    by_id = {c["driver_id"]: c for c in checklist}

    answer = _ask_model(build_prompt(request, checklist, raining, margin))
    choice, why = None, "model_unavailable"
    if answer is not None:
        choice, why = check_reply(answer["reply"], set(by_id), cfg["max_reason_chars"])

    status, rejected = "assigned", None
    if choice is None:  # fall back to the fastest road pickup time (the drive time to the rider)
        status = "assigned_fallback"
        said = answer["reply"].get("driver_id") if answer and isinstance(answer["reply"], dict) else None
        rejected = {"why": why, "model_said": said}
        choice = {"driver_id": checklist[0]["driver_id"], "reason": "fallback: fastest road pickup time"}

    minutes = int(by_id[choice["driver_id"]]["road_pickup_minutes"])
    arrival = pd.Timestamp(request["requested_at"]) + timedelta(minutes=minutes)
    result = {
        "request_id": request_id,
        "status": status,
        "driver_id": choice["driver_id"],
        "road_pickup_minutes": minutes,
        "arrival_time": arrival.isoformat(),
        "reason": choice["reason"],
        "raining": raining,
        "model": answer["model"] if answer else None,
        "generation_id": answer.get("generation_id") if answer else None,
        "served_model": answer.get("served_model") if answer else None,
    }
    if rejected:
        result["rejected"] = rejected
    return {"steps": ["rank_drivers"], "result": result}


def build_graph():
    g = StateGraph(State)
    g.add_node("availability", availability)
    g.add_node("pickup_minutes", pickup_minutes)
    g.add_node("eligibility", eligibility)
    g.add_node("rank_drivers", rank_drivers)
    g.add_edge(START, "availability")
    g.add_edge("availability", "pickup_minutes")
    g.add_edge("pickup_minutes", "eligibility")
    g.add_edge("eligibility", "rank_drivers")
    g.add_edge("rank_drivers", END)
    return g.compile()


def run_request(request_id: str, raining: bool = False) -> dict:
    drivers = pd.read_csv(DATA_DIR / "candidate_drivers.csv")
    requests_df = pd.read_csv(DATA_DIR / "ride_requests.csv")
    match = requests_df.loc[requests_df["request_id"] == request_id]
    if match.empty:
        raise ValueError(f"Unknown request_id {request_id!r}: not in ride_requests.csv")
    request = match.iloc[0].to_dict()
    drivers = drivers[drivers["request_id"] == request_id]
    return build_graph().invoke(
        {"request_id": request_id, "request": request, "drivers": drivers, "raining": raining}
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rid = args[0] if args else "RQ-0304-005"
    out = run_request(rid, raining="--rain" in sys.argv)
    print(f"steps run, in order: {' -> '.join(out['steps'])}")
    print(f"drivers: {len(out['drivers'])} candidates -> {len(out['available_drivers'])} available "
          f"-> {len(out['fastest_drivers'])} after pickup_minutes -> {len(out['eligible_drivers'])} eligible")
    print(json.dumps(out["result"], indent=2))
