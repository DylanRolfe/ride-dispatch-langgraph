"""LangGraph: availability -> pickup_minutes -> eligibility -> free OpenRouter model ranks the top 3-4."""
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

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"  # free tier; override with OPENROUTER_MODEL
FALLBACK_MODEL = "openrouter/free"  # OpenRouter's free-model router
MAX_RANKED = 4  # model ranks the top 3-4
MAX_CANDIDATES = 4  # pickup_minutes keeps only the fastest N drivers per request
MAX_STATUS_AGE_MIN = 120  # eligibility: status_updated_at may be at most this many minutes before the request

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


class State(TypedDict, total=False):
    request_id: str                  # input: the ride request being dispatched
    request: dict                    # input: that request's row from ride_requests.csv
    drivers: pd.DataFrame            # input: candidate drivers
    available_drivers: pd.DataFrame  # node 1 (availability) output
    fastest_drivers: pd.DataFrame    # node 2 (pickup_minutes) output
    eligible_drivers: pd.DataFrame   # node 3 (eligibility) output
    result: dict                     # node 4 (rank_drivers) output: ranked top 3-4
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
    """Node 2: drop rows with no road_pickup_minutes, then keep the fastest MAX_CANDIDATES per request."""
    df = state["available_drivers"]
    if df.empty:
        return {"fastest_drivers": _empty(), "steps": ["pickup_minutes"]}
    df = df[df["road_pickup_minutes"].notna()]
    df = df.sort_values("road_pickup_minutes", kind="stable").groupby("request_id", sort=False).head(MAX_CANDIDATES)
    return {"fastest_drivers": df, "steps": ["pickup_minutes"]}


def eligibility(state: State) -> State:
    """Node 3: keep drivers whose status was updated no more than MAX_STATUS_AGE_MIN minutes before the request.

    A status stamped after the request time counts as fresh. An unreadable timestamp counts as not fresh.
    """
    df = state["fastest_drivers"]
    if df.empty:
        return {"eligible_drivers": _empty(), "steps": ["eligibility"]}
    requested = pd.to_datetime(state["request"]["requested_at"], utc=True)
    updated = pd.to_datetime(df["status_updated_at"], utc=True, format="ISO8601", errors="coerce")
    age_min = (requested - updated).dt.total_seconds() / 60
    return {"eligible_drivers": df[age_min <= MAX_STATUS_AGE_MIN], "steps": ["eligibility"]}


def _clean(value):
    """Make a pandas/numpy value JSON-safe (NaN -> None, numpy -> python)."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _build_prompt(request: dict, candidates: list[dict], n: int) -> str:
    context = {k: _clean(request.get(k)) for k in REQUEST_FIELDS}
    return (
        f"Ride request context:\n{json.dumps(context, indent=1)}\n\n"
        f"Available candidate drivers:\n{json.dumps(candidates, indent=1)}\n\n"
        f"Rank the top {n} drivers to offer this ride, best first. Prefer the lowest "
        "road_pickup_minutes unless a route_note, the river side or the bridge context makes that "
        "time unreliable. Only use driver_ids from the list above. Reply with JSON only: "
        '{"ranking": [{"driver_id": "<id>", "reason": "<short>"}]}'
    )


def _ask_model(prompt: str) -> dict | None:
    """Call the free OpenRouter model. Returns parsed JSON (+ model name) or None if every attempt fails."""
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
                return {**parsed, "model": attempt_model,
                        "generation_id": data.get("id"), "served_model": data.get("model")}
    return None  # caller falls back to road-minutes order


def rank_drivers(state: State) -> State:
    """Node 2: send the request + all available drivers to the model; validate its top 3-4 in code."""
    request_id, request = state["request_id"], state["request"]
    avail = state["eligible_drivers"]
    avail = avail[avail["request_id"] == request_id].sort_values("road_pickup_minutes", kind="stable")

    if avail.empty:  # nobody left after availability / pickup_minutes / eligibility: model is not called
        return {"steps": ["rank_drivers"], "result": {
            "request_id": request_id,
            "status": "pending_approval",
            "message": "Pending approval",
            "ranking": [],
        }}

    candidates = [{k: _clean(row[k]) for k in DRIVER_FIELDS} for _, row in avail.iterrows()]
    by_id = {c["driver_id"]: c for c in candidates}
    n = min(MAX_RANKED, len(candidates))

    answer = _ask_model(_build_prompt(request, candidates, n))
    raw = answer.get("ranking") if answer else None
    raw = raw if isinstance(raw, list) else []

    picked, reasons = [], {}
    for item in raw:  # keep only real, unique candidate ids, in the model's order
        did = item.get("driver_id") if isinstance(item, dict) else None
        if did in by_id and did not in picked:
            picked.append(did)
            reasons[did] = item.get("reason")
    picked = picked[:n]

    status = "ranked"
    if len(picked) < n:  # model gave too few valid ids: top up with fastest road pickups
        status = "ranked_fallback"
        for c in candidates:
            if len(picked) == n:
                break
            if c["driver_id"] not in picked:
                picked.append(c["driver_id"])
                reasons[c["driver_id"]] = "added by road-minutes order (model answer incomplete)"

    t0 = pd.Timestamp(request["requested_at"])
    ranking = []
    for rank, did in enumerate(picked, start=1):
        minutes = int(by_id[did]["road_pickup_minutes"])
        ranking.append({
            "rank": rank,
            "driver_id": did,
            "arrival_minutes": minutes,
            "arrival_time": (t0 + timedelta(minutes=minutes)).isoformat(),
            "reason": reasons.get(did),
        })
    return {"steps": ["rank_drivers"], "result": {
        "request_id": request_id,
        "status": status,
        "model": answer["model"] if answer else None,
        "generation_id": answer.get("generation_id") if answer else None,
        "served_model": answer.get("served_model") if answer else None,
        "ranking": ranking,
    }}


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


def run_request(request_id: str) -> dict:
    drivers = pd.read_csv(DATA_DIR / "candidate_drivers.csv")
    requests_df = pd.read_csv(DATA_DIR / "ride_requests.csv")
    match = requests_df.loc[requests_df["request_id"] == request_id]
    if match.empty:
        raise ValueError(f"Unknown request_id {request_id!r}: not in ride_requests.csv")
    request = match.iloc[0].to_dict()
    drivers = drivers[drivers["request_id"] == request_id]
    return build_graph().invoke({"request_id": request_id, "request": request, "drivers": drivers})


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "RQ-0304-005"
    out = run_request(rid)
    print(f"steps run, in order: {' -> '.join(out['steps'])}")
    print(f"drivers: {len(out['drivers'])} candidates -> {len(out['available_drivers'])} available "
          f"-> {len(out['fastest_drivers'])} after pickup_minutes -> {len(out['eligible_drivers'])} eligible")
    print(json.dumps(out["result"], indent=2))
