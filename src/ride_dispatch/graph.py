"""LangGraph pipeline that loads the ride request and candidate driver CSVs."""
from pathlib import Path
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class State(TypedDict, total=False):
    requests: pd.DataFrame
    drivers: pd.DataFrame
    summary: dict


def load_requests(state: State) -> State:
    df = pd.read_csv(DATA_DIR / "ride_requests.csv", parse_dates=["requested_at"], date_format="ISO8601")
    return {"requests": df}


def load_drivers(state: State) -> State:
    df = pd.read_csv(DATA_DIR / "candidate_drivers.csv")
    return {"drivers": df}


def summarize(state: State) -> State:
    req, drv = state["requests"], state["drivers"]
    return {
        "summary": {
            "requests": len(req),
            "candidate_rows": len(drv),
            "bridge_cases": int(req["is_bridge_case"].sum()),
            "distance_time_disagree": int(req["distance_time_disagree"].sum()),
        }
    }


def build_graph():
    g = StateGraph(State)
    g.add_node("load_requests", load_requests)
    g.add_node("load_drivers", load_drivers)
    g.add_node("summarize", summarize)
    g.add_edge(START, "load_requests")
    g.add_edge("load_requests", "load_drivers")
    g.add_edge("load_drivers", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


if __name__ == "__main__":
    print(build_graph().invoke({})["summary"])
