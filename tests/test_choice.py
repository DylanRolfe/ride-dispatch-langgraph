"""Checks the single-driver choice: fixed reply shape, rejection of ids outside the checklist, the fallback to the
fastest road pickup, the rain margin, and that the agent and these tests read the same settings file.

Run: python -m unittest tests.test_choice -v   (no API key or network needed; the model call is faked)
"""
import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import src.ride_dispatch.graph as g
from src.ride_dispatch import settings
from tests.test_graph_order import drivers_for, fake_answer, request_row

ROOT = g.ROOT


def run_node(eligible, raining=False, answer=None, capture=None):
    """Call rank_drivers directly with a given eligible table and a faked model reply."""
    def fake_model(prompt):
        if capture is not None:
            capture.append(prompt)
        return answer

    state = {"request_id": "RQ-0304-005", "request": request_row(), "eligible_drivers": eligible, "raining": raining}
    with mock.patch.object(g, "_ask_model", fake_model):
        return g.rank_drivers(state)["result"]


def eligible_real():
    """The real eligible drivers for RQ-0304-005, produced by the three nodes."""
    state = {"request_id": "RQ-0304-005", "request": request_row(), "drivers": drivers_for()}
    for node in (g.availability, g.pickup_minutes, g.eligibility):
        state.update(node(state))
    return state["eligible_drivers"]


def rain_table(offsets):
    """Three drivers whose road minutes are the fastest plus each offset (offsets come from settings)."""
    df = eligible_real().iloc[:len(offsets)].copy()
    fastest = int(df["road_pickup_minutes"].min())
    df["driver_id"] = [f"R{i}" for i in range(len(df))]
    df["road_pickup_minutes"] = [fastest + o for o in offsets]
    return df


class ReplyShapeTests(unittest.TestCase):
    def test_valid_reply_returns_one_driver_and_a_one_line_reason(self):
        eligible = eligible_real()
        second = eligible.sort_values("road_pickup_minutes").iloc[1]["driver_id"]
        r = run_node(eligible, answer=fake_answer(second, "line one\n  line   two"))
        self.assertEqual(r["status"], "assigned")
        self.assertEqual(r["driver_id"], second)
        self.assertNotIn("\n", r["reason"])
        self.assertEqual(r["reason"], "line one line two")

    def test_reason_is_cut_to_the_limit_in_settings(self):
        eligible = eligible_real()
        limit = int(settings.load()["max_reason_chars"])
        r = run_node(eligible, answer=fake_answer(eligible.iloc[0]["driver_id"], "x" * (limit * 3)))
        self.assertEqual(len(r["reason"]), limit)

    def test_wrong_shapes_are_rejected(self):
        eligible = eligible_real()
        good = eligible.iloc[0]["driver_id"]
        bad_replies = [
            {"driver_id": good},                                  # no reason
            {"driver_id": good, "reason": "x", "extra": 1},       # extra key
            {"driver_id": [good], "reason": "x"},                 # id is not a string
            {"driver_id": good, "reason": "   "},                 # empty reason
            ["driver_id", good],                                  # not an object
        ]
        for reply in bad_replies:
            with self.subTest(reply=reply):
                r = run_node(eligible, answer={"reply": reply, "model": "fake"})
                self.assertEqual(r["status"], "assigned_fallback")
                self.assertEqual(r["rejected"]["why"], "bad_shape")


class OutsideChecklistTests(unittest.TestCase):
    def test_ids_outside_the_checklist_are_rejected_and_the_fallback_is_used(self):
        eligible = eligible_real()
        other_request = drivers_for("RQ-0304-006").iloc[0]["driver_id"]
        outside = ["DRV-99999",                                   # invented
                   "DRV-33907",                                   # real, but offline for this request
                   other_request]                                 # real, but belongs to another request
        for bad_id in outside:
            with self.subTest(bad_id=bad_id):
                r = run_node(eligible, answer=fake_answer(bad_id))
                self.assertEqual(r["status"], "assigned_fallback")
                self.assertEqual(r["rejected"], {"why": "not_on_checklist", "model_said": bad_id})
                self.assertNotEqual(r["driver_id"], bad_id)
                self.assertIn(r["driver_id"], set(eligible["driver_id"]))

    def test_model_unavailable_uses_the_fallback(self):
        r = run_node(eligible_real(), answer=None)
        self.assertEqual(r["status"], "assigned_fallback")
        self.assertEqual(r["rejected"]["why"], "model_unavailable")
        self.assertIsNone(r["model"])


class FallbackTests(unittest.TestCase):
    def test_fallback_is_the_fastest_road_pickup_not_the_closest_on_the_map(self):
        eligible = eligible_real()
        fastest_by_road = eligible.sort_values("road_pickup_minutes").iloc[0]
        closest_on_map = eligible.sort_values("straight_line_km").iloc[0]
        self.assertNotEqual(fastest_by_road["driver_id"], closest_on_map["driver_id"])  # this request separates them
        r = run_node(eligible, answer=fake_answer("DRV-99999"))
        self.assertEqual(r["driver_id"], fastest_by_road["driver_id"])
        self.assertEqual(r["road_pickup_minutes"], int(fastest_by_road["road_pickup_minutes"]))

    def test_arrival_uses_the_drive_time_to_the_rider(self):
        r = run_node(eligible_real(), answer=None)
        requested = pd.Timestamp(request_row()["requested_at"])
        self.assertEqual(pd.Timestamp(r["arrival_time"]), requested + pd.Timedelta(minutes=r["road_pickup_minutes"]))


class RainTests(unittest.TestCase):
    def setUp(self):
        self.margin = settings.load()["rain_pickup_margin_minutes"]

    def test_in_rain_only_drivers_within_the_margin_are_accepted(self):
        table = rain_table([0, self.margin, self.margin + 1])
        fastest, at_margin, too_slow = table["driver_id"]
        self.assertEqual(run_node(table, True, fake_answer(at_margin))["status"], "assigned")
        r = run_node(table, True, fake_answer(too_slow))
        self.assertEqual((r["status"], r["driver_id"]), ("assigned_fallback", fastest))
        self.assertEqual(r["rejected"]["why"], "not_on_checklist")

    def test_without_rain_the_same_slower_driver_is_accepted(self):
        table = rain_table([0, self.margin, self.margin + 1])
        too_slow = table["driver_id"].iloc[2]
        r = run_node(table, False, fake_answer(too_slow))
        self.assertEqual((r["status"], r["driver_id"]), ("assigned", too_slow))

    def test_the_prompt_only_lists_acceptable_drivers_in_rain(self):
        table = rain_table([0, self.margin, self.margin + 1])
        too_slow = table["driver_id"].iloc[2]
        rain_prompt, dry_prompt = [], []
        run_node(table, True, fake_answer(table["driver_id"].iloc[0]), capture=rain_prompt)
        run_node(table, False, fake_answer(table["driver_id"].iloc[0]), capture=dry_prompt)
        self.assertNotIn(too_slow, rain_prompt[0])
        self.assertIn(too_slow, dry_prompt[0])


class OneSettingsSpotTests(unittest.TestCase):
    """The agent and the tests must read the same number, and no threshold may be typed in code."""

    def test_agent_follows_a_changed_settings_file(self):
        base = settings.load()
        wider = dict(base, rain_pickup_margin_minutes=base["rain_pickup_margin_minutes"] * 2 + 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "settings.json"
            path.write_text(json.dumps(wider), encoding="utf-8")
            for margin_file, expect in ((None, "assigned_fallback"), (str(path), "assigned")):
                with self.subTest(settings_file=margin_file):
                    env = {"RIDE_DISPATCH_SETTINGS": margin_file} if margin_file else {}
                    with mock.patch.dict(os.environ, env):
                        if not margin_file:
                            os.environ.pop("RIDE_DISPATCH_SETTINGS", None)
                        margin = settings.load()["rain_pickup_margin_minutes"]     # the test reads the file...
                        table = rain_table([0, base["rain_pickup_margin_minutes"] + 1])
                        r = run_node(table, True, fake_answer(table["driver_id"].iloc[1]))  # ...and so does the agent
                        self.assertEqual(r["status"], expect, f"margin in use: {margin}")

    def test_no_threshold_is_assigned_a_literal_in_the_agent_or_the_tests(self):
        names = re.compile(r"margin|max_|limit|thresh|age", re.I)
        files = [ROOT / "src" / "ride_dispatch" / "graph.py", *sorted((ROOT / "tests").glob("*.py"))]
        found = []
        for f in files:
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                value = getattr(node, "value", None)
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                    for t in targets:
                        if isinstance(t, ast.Name) and names.search(t.id):
                            found.append(f"{f.name}:{node.lineno} {t.id} = {value.value}")
        self.assertEqual(found, [])

    def test_the_settings_numbers_live_only_in_the_settings_file(self):
        for key in settings.REQUIRED:
            hits = [p.name for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
                    if re.search(rf"{key}[\"']?\s*[:=]\s*\d", p.read_text(encoding="utf-8"))]
            self.assertEqual(hits, [], f"{key} is typed as a number in {hits}")

    def test_both_sides_load_through_the_same_function(self):
        src = (ROOT / "src" / "ride_dispatch" / "graph.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("settings.load()"), 3)          # the three nodes that use a threshold
        self.assertIn("settings.load()", Path(__file__).read_text(encoding="utf-8"))  # and these tests


if __name__ == "__main__":
    unittest.main()
