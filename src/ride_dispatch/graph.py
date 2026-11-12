"""LangGraph: keep available drivers -> rank by road pickup minutes -> free OpenRouter model picks one."""
import json
import os
import re
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

DETAIL_FIELDS = [
    "request_id",
    "driver_id",
    "straight_line_km",
    "road_pickup_minutes",
    "driver_river_side",
    "driver_lat",
    "driver_lng",
    "status_updated_at",
    "route_note",
]


class State(TypedDict, total=False):
    request_id: str                  # input: the ride request being dispatched
    drivers: pd.DataFrame            # input: candidate drivers
    request_time: str                # input: requested_at of the ride (ISO 8601)
    available_drivers: pd.DataFrame  # node 1 output
    driver_details: dict             # node 2 output: request_id -> ranked driver dicts
    decision: dict                   # node 3 output: driver_id + arrival


def filter_available(state: State) -> State:
    """Node 1: only drivers tagged 'available' come through."""
    drivers = state["drivers"]
    return {"available_drivers": drivers[drivers["driver_status"] == "available"]}


def driver_details(state: State) -> State:
    """Node 2: rank available drivers by road_pickup_minutes (fastest first)."""
    df = state["available_drivers"][DETAIL_FIELDS].sort_values(
        ["request_id", "road_pickup_minutes"]
    )
    df = df.astype(object).where(df.notna(), None)
    details = {}
    for rid, group in df.groupby("request_id", sort=True):
        rows = group.drop(columns="request_id").to_dict("records")
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        details[rid] = rows
    return {"driver_details": details}


def _ask_model(request_id: str, candidates: list[dict]) -> dict:
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    prompt = (
        f"Ride request {request_id}. Candidate drivers, ranked by road_pickup_minutes (rank 1 = fastest):\n"
        f"{json.dumps(candidates, indent=1)}\n\n"
        "Pick the single best driver to dispatch. Prefer the lowest road_pickup_minutes unless a "
        "route_note makes that time unreliable. Reply with JSON only: "
        '{"driver_id": "<id>", "arrival_minutes": <number>, "reason": "<short>"}'
    )
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
    return None  # every attempt failed; caller falls back to rank 1


def choose_driver(state: State) -> State:
    """Node 3: the model picks one driver; arrival comes from the data, not the model."""
    request_id = state["request_id"]
    candidates = state["driver_details"].get(request_id, [])
    if not candidates:  # nobody available: return a clean answer, no model call
        return {"decision": {
            "request_id": request_id,
            "status": "no_driver_available",
            "message": "No drivers available",
            "driver_id": None,
            "arrival_minutes": None,
            "arrival_time": None,
        }}

    answer = _ask_model(request_id, candidates)
    chosen, status, note = None, "assigned", None
    if answer is None:
        note = "model unavailable; used rank 1"
    else:
        chosen = next((c for c in candidates if c["driver_id"] == answer.get("driver_id")), None)
        if chosen is None:
            note = "model answer invalid; used rank 1"
    if chosen is None:  # fall back to the fastest road pickup
        chosen, status = candidates[0], "assigned_fallback"
    minutes = chosen["road_pickup_minutes"]
    arrival = pd.Timestamp(state["request_time"]) + timedelta(minutes=int(minutes))
    return {
        "decision": {
            "request_id": request_id,
            "status": status,
            "driver_id": chosen["driver_id"],
            "arrival_minutes": int(minutes),
            "arrival_time": arrival.isoformat(),
            "model": answer["model"] if answer else None,
            "reason": answer.get("reason") if status == "assigned" else note,
            **({"model_said": answer.get("driver_id")} if status == "assigned_fallback" and answer else {}),
        }
    }


def build_graph():
    g = StateGraph(State)
    g.add_node("filter_available", filter_available)
    g.add_node("driver_details", driver_details)
    g.add_node("choose_driver", choose_driver)
    g.add_edge(START, "filter_available")
    g.add_edge("filter_available", "driver_details")
    g.add_edge("driver_details", "choose_driver")
    g.add_edge("choose_driver", END)
    return g.compile()


def run_request(request_id: str) -> dict:
    drivers = pd.read_csv(DATA_DIR / "candidate_drivers.csv")
    requests_df = pd.read_csv(DATA_DIR / "ride_requests.csv")
    drivers = drivers[drivers["request_id"] == request_id]
    request_time = requests_df.loc[requests_df["request_id"] == request_id, "requested_at"].iloc[0]
    return build_graph().invoke(
        {"request_id": request_id, "drivers": drivers, "request_time": request_time}
    )


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "RQ-0304-005"
    result = run_request(rid)
    print("ranked candidates:")
    for d in result["driver_details"].get(rid, []):
        print(f"  #{d['rank']} {d['driver_id']}  {d['road_pickup_minutes']} min  ({d['straight_line_km']} km)")
    print(json.dumps(result["decision"], indent=2))
