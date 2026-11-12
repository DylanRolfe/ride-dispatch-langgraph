"""Two-node LangGraph: keep available drivers -> free OpenRouter model ranks the top 3-4."""
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TypedDict

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
    available_drivers: pd.DataFrame  # node 1 output
    result: dict                     # node 2 output: ranked top 3-4


def filter_available(state: State) -> State:
    """Node 1: only drivers tagged 'available' come through (on_trip / offline are cut)."""
    drivers = state["drivers"]
    if drivers.empty or "driver_status" not in drivers.columns:  # zero candidates / no data at all
        return {"available_drivers": pd.DataFrame(columns=["request_id", *DRIVER_FIELDS])}
    return {"available_drivers": drivers[drivers["driver_status"] == "available"]}


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
                text = resp.json()["choices"][0]["message"]["content"]
                start = text.find("{")
                parsed, _ = json.JSONDecoder().raw_decode(text[start:]) if start != -1 else (None, 0)
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
                continue  # network/server error, odd response shape, or malformed JSON -> retry
            if isinstance(parsed, dict):
                return {"model": attempt_model, **parsed}
    return None  # caller falls back to road-minutes order


def rank_drivers(state: State) -> State:
    """Node 2: send the request + all available drivers to the model; validate its top 3-4 in code."""
    request_id, request = state["request_id"], state["request"]
    avail = state["available_drivers"]
    avail = avail[avail["request_id"] == request_id].sort_values("road_pickup_minutes", kind="stable")

    if avail.empty:  # nobody available: needs approval, model is not called
        return {"result": {
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
    return {"result": {
        "request_id": request_id,
        "status": status,
        "model": answer["model"] if answer else None,
        "ranking": ranking,
    }}


def build_graph():
    g = StateGraph(State)
    g.add_node("filter_available", filter_available)
    g.add_node("rank_drivers", rank_drivers)
    g.add_edge(START, "filter_available")
    g.add_edge("filter_available", "rank_drivers")
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
    print(f"available drivers: {len(out['available_drivers'])} of {len(out['drivers'])}")
    print(json.dumps(out["result"], indent=2))
