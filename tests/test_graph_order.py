"""Checks that the nodes run in order, that the model is only called after all three checks, and that each
check follows the numbers in config/settings.json (no threshold is typed in this file).

Run: python -m unittest tests.test_graph_order -v   (no API key or network needed; the model call is faked)
"""
import unittest
from unittest import mock

import pandas as pd

import src.ride_dispatch.graph as g
from src.ride_dispatch import settings

EXPECTED = ["availability", "pickup_minutes", "eligibility", "rank_drivers"]


def fake_answer(driver_id="DRV-47038", reason="x"):
    return {"reply": {"driver_id": driver_id, "reason": reason}, "model": "fake",
            "generation_id": "gen-test", "served_model": "fake"}


def request_row(request_id="RQ-0304-005"):
    return pd.read_csv(g.DATA_DIR / "ride_requests.csv").query("request_id == @request_id").iloc[0].to_dict()


def drivers_for(request_id="RQ-0304-005"):
    return pd.read_csv(g.DATA_DIR / "candidate_drivers.csv").query("request_id == @request_id").copy()


def run_logged(drivers=None, reply=None):
    """Run the graph with every node and the model call wrapped to log when they start."""
    events = []

    def wrap(name, fn):
        def inner(state):
            events.append(name)
            return fn(state)
        return inner

    def fake_model(prompt):
        events.append("model")
        return reply or fake_answer()

    patches = [mock.patch.object(g, n, wrap(n, getattr(g, n))) for n in EXPECTED]
    patches.append(mock.patch.object(g, "_ask_model", fake_model))
    for p in patches:
        p.start()
    try:
        drivers = drivers_for() if drivers is None else drivers
        out = g.build_graph().invoke({"request_id": "RQ-0304-005", "request": request_row(), "drivers": drivers})
    finally:
        for p in patches:
            p.stop()
    return out, events


class OrderTests(unittest.TestCase):
    def test_nodes_run_in_order_and_model_comes_last(self):
        out, events = run_logged()
        self.assertEqual(events, EXPECTED + ["model"])

    def test_state_records_the_same_order(self):
        out, _ = run_logged()
        self.assertEqual(out["steps"], EXPECTED)

    def test_langgraph_itself_reports_the_same_order(self):
        with mock.patch.object(g, "_ask_model", lambda p: fake_answer()):
            updates = list(g.build_graph().stream(
                {"request_id": "RQ-0304-005", "request": request_row(), "drivers": drivers_for()},
                stream_mode="updates"))
        self.assertEqual([next(iter(u)) for u in updates], EXPECTED)

    def test_model_never_runs_when_nobody_survives(self):
        drivers = drivers_for()
        drivers["driver_status"] = "offline"
        out, events = run_logged(drivers=drivers)
        self.assertEqual(events, EXPECTED)  # all four nodes still run, in order, but no "model"
        self.assertEqual(out["result"]["status"], "pending_approval")


class EachCheckTests(unittest.TestCase):
    def setUp(self):
        self.drivers = drivers_for()
        self.cfg = settings.load()

    def test_availability_cuts_offline_and_on_trip(self):
        out = g.availability({"drivers": self.drivers})["available_drivers"]
        self.assertEqual(set(out["driver_status"]), {"available"})
        self.assertNotIn("DRV-33907", set(out["driver_id"]))  # the offline one

    def test_pickup_minutes_keeps_only_the_fastest_n(self):
        cap = int(self.cfg["max_candidates"])
        template = self.drivers[self.drivers["driver_status"] == "available"].iloc[:1]
        many = pd.concat([template] * (cap + 2), ignore_index=True)
        many["driver_id"] = [f"D{i}" for i in range(len(many))]
        many["road_pickup_minutes"] = list(reversed(range(1, len(many) + 1)))  # slowest first in the table
        out = g.pickup_minutes({"available_drivers": many})["fastest_drivers"]
        self.assertEqual(sorted(out["road_pickup_minutes"]), list(range(1, cap + 1)))

    def test_pickup_minutes_drops_missing_minutes(self):
        df = self.drivers[self.drivers["driver_status"] == "available"].copy()
        df.iloc[0, df.columns.get_loc("road_pickup_minutes")] = None
        out = g.pickup_minutes({"available_drivers": df})["fastest_drivers"]
        self.assertEqual(len(out), len(df) - 1)

    def test_eligibility_follows_the_age_limit_in_settings(self):
        limit = self.cfg["max_status_age_minutes"]
        requested = pd.Timestamp(request_row()["requested_at"])
        df = self.drivers[self.drivers["driver_status"] == "available"].iloc[:3].copy()
        stamp = lambda minutes_before: (requested - pd.Timedelta(minutes=minutes_before)).isoformat()
        df["status_updated_at"] = [stamp(limit - 1),    # just inside the limit: fresh
                                   stamp(limit + 1),    # just outside the limit: stale
                                   stamp(-1)]           # stamped after the request: fresh
        out = g.eligibility({"fastest_drivers": df, "request": request_row()})["eligible_drivers"]
        self.assertEqual(list(out["driver_id"]), [df.iloc[0]["driver_id"], df.iloc[2]["driver_id"]])

    def test_every_node_handles_an_empty_table(self):
        a = g.availability({"drivers": pd.DataFrame()})
        b = g.pickup_minutes({"available_drivers": a["available_drivers"]})
        c = g.eligibility({"fastest_drivers": b["fastest_drivers"], "request": request_row()})
        self.assertTrue(c["eligible_drivers"].empty)


if __name__ == "__main__":
    unittest.main()
