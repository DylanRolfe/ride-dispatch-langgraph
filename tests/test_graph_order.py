"""Checks that the nodes run in order and that the model is only called after all three checks.

Run: python -m unittest tests.test_graph_order -v   (no API key or network needed; the model call is faked)
"""
import unittest
from unittest import mock

import pandas as pd

import src.ride_dispatch.graph as g

EXPECTED = ["availability", "pickup_minutes", "eligibility", "rank_drivers"]
GOOD_REPLY = {"ranking": [{"driver_id": "DRV-47038", "reason": "x"}, {"driver_id": "DRV-26148", "reason": "x"},
                          {"driver_id": "DRV-58423", "reason": "x"}], "model": "fake"}


def run_logged(request_id="RQ-0304-005", drivers=None, reply=GOOD_REPLY):
    """Run the graph with every node and the model call wrapped to log when it starts."""
    events = []

    def wrap(name, fn):
        def inner(state):
            events.append(name)
            return fn(state)
        return inner

    def fake_model(prompt):
        events.append("model")
        return reply

    patches = [mock.patch.object(g, n, wrap(n, getattr(g, n))) for n in EXPECTED]
    patches.append(mock.patch.object(g, "_ask_model", fake_model))
    for p in patches:
        p.start()
    try:
        if drivers is None:
            out = g.run_request(request_id)
        else:
            req = pd.read_csv(g.DATA_DIR / "ride_requests.csv").query("request_id == @request_id").iloc[0].to_dict()
            out = g.build_graph().invoke({"request_id": request_id, "request": req, "drivers": drivers})
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
        req = pd.read_csv(g.DATA_DIR / "ride_requests.csv").query("request_id == 'RQ-0304-005'").iloc[0].to_dict()
        drivers = pd.read_csv(g.DATA_DIR / "candidate_drivers.csv").query("request_id == 'RQ-0304-005'")
        with mock.patch.object(g, "_ask_model", lambda p: GOOD_REPLY):
            updates = list(g.build_graph().stream(
                {"request_id": "RQ-0304-005", "request": req, "drivers": drivers}, stream_mode="updates"))
        self.assertEqual([next(iter(u)) for u in updates], EXPECTED)

    def test_model_never_runs_when_nobody_survives(self):
        drivers = pd.read_csv(g.DATA_DIR / "candidate_drivers.csv").query("request_id == 'RQ-0304-005'").copy()
        drivers["driver_status"] = "offline"
        out, events = run_logged(drivers=drivers)
        self.assertEqual(events, EXPECTED)  # all four nodes still run, in order, but no "model"
        self.assertEqual(out["result"]["status"], "pending_approval")


class EachCheckTests(unittest.TestCase):
    def setUp(self):
        self.drivers = pd.read_csv(g.DATA_DIR / "candidate_drivers.csv").query("request_id == 'RQ-0304-005'")

    def test_availability_cuts_offline_and_on_trip(self):
        out = g.availability({"drivers": self.drivers})["available_drivers"]
        self.assertEqual(set(out["driver_status"]), {"available"})
        self.assertNotIn("DRV-33907", set(out["driver_id"]))  # the offline one

    def test_pickup_minutes_keeps_only_the_fastest_n(self):
        many = pd.concat([self.drivers[self.drivers["driver_status"] == "available"]] * 2, ignore_index=True)
        many["driver_id"] = [f"D{i}" for i in range(len(many))]
        many["road_pickup_minutes"] = [10, 3, 8, 1, 9, 2][: len(many)]
        out = g.pickup_minutes({"available_drivers": many})["fastest_drivers"]
        self.assertEqual(sorted(out["road_pickup_minutes"]), [1, 2, 3, 8][: g.MAX_CANDIDATES])
        self.assertEqual(len(out), g.MAX_CANDIDATES)

    def test_pickup_minutes_drops_missing_minutes(self):
        df = self.drivers[self.drivers["driver_status"] == "available"].copy()
        df.iloc[0, df.columns.get_loc("road_pickup_minutes")] = None
        out = g.pickup_minutes({"available_drivers": df})["fastest_drivers"]
        self.assertEqual(len(out), len(df) - 1)

    def test_eligibility_drops_stale_and_keeps_future_stamps(self):
        req = {"requested_at": "2025-03-04T13:00:00-08:00"}
        df = self.drivers[self.drivers["driver_status"] == "available"].iloc[:3].copy()
        df["status_updated_at"] = ["2025-03-04T12:00:00-08:00",   # 60 min before: fresh
                                   "2025-03-04T10:00:00-08:00",   # 180 min before: stale
                                   "2025-03-04T14:00:00-08:00"]   # after the request: fresh
        out = g.eligibility({"fastest_drivers": df, "request": req})["eligible_drivers"]
        self.assertEqual(list(out["driver_id"]), [df.iloc[0]["driver_id"], df.iloc[2]["driver_id"]])

    def test_every_node_handles_an_empty_table(self):
        empty = pd.DataFrame()
        a = g.availability({"drivers": empty})
        b = g.pickup_minutes({"available_drivers": a["available_drivers"]})
        c = g.eligibility({"fastest_drivers": b["fastest_drivers"], "request": {"requested_at": "2025-03-04T13:00:00-08:00"}})
        self.assertTrue(c["eligible_drivers"].empty)


if __name__ == "__main__":
    unittest.main()
