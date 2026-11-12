"""Two-node LangGraph: keep available drivers, then pass on their trip data."""
from pathlib import Path
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

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
    drivers: pd.DataFrame            # input: all candidate drivers
    available_drivers: pd.DataFrame  # node 1 output
    driver_details: dict             # node 2 output: request_id -> list of driver dicts


def filter_available(state: State) -> State:
    """Node 1: only drivers tagged 'available' come through."""
    drivers = state["drivers"]
    return {"available_drivers": drivers[drivers["driver_status"] == "available"]}


def driver_details(state: State) -> State:
    """Node 2: receives available drivers and their distance/time/route data."""
    df = state["available_drivers"][DETAIL_FIELDS].sort_values(
        ["request_id", "road_pickup_minutes"]
    )
    df = df.astype(object).where(df.notna(), None)
    details = {
        rid: group.drop(columns="request_id").to_dict("records")
        for rid, group in df.groupby("request_id", sort=True)
    }
    return {"driver_details": details}


def build_graph():
    g = StateGraph(State)
    g.add_node("filter_available", filter_available)
    g.add_node("driver_details", driver_details)
    g.add_edge(START, "filter_available")
    g.add_edge("filter_available", "driver_details")
    g.add_edge("driver_details", END)
    return g.compile()


def load_drivers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "candidate_drivers.csv")


if __name__ == "__main__":
    result = build_graph().invoke({"drivers": load_drivers()})
    print(f"available drivers: {len(result['available_drivers'])} of {len(result['drivers'])}")
    for rid, drivers in result["driver_details"].items():
        print(rid, [(d["driver_id"], d["road_pickup_minutes"]) for d in drivers])
