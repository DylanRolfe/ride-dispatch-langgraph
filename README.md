# ride-dispatch-langgraph

Python + LangGraph project that takes a week (3-9 March 2025) of ride requests and candidate-driver snapshots and picks one driver per request.

- `data/ride_requests.csv` – 20 ride requests
- `data/candidate_drivers.csv` – candidate drivers per request

## Graph

```
START -> filter_available -> driver_details -> choose_driver -> END
```

1. `filter_available` – keeps only drivers with `driver_status == "available"` (`on_trip` and `offline` are dropped).
2. `driver_details` – ranks the remaining drivers by `road_pickup_minutes` (fastest first).
3. `choose_driver` – a free OpenRouter model (default `google/gemma-4-31b-it:free`, falls back to `openrouter/free`) picks one driver. Arrival is computed from the data (request time + road minutes), not by the model. If the model's answer is unusable, the rank 1 driver is used (see Failure handling).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your OPENROUTER_API_KEY
python -m src.ride_dispatch.graph RQ-0304-005
```

## Output

Run on `RQ-0304-005` (three available candidates):

![Graph output](docs/output.png)

```
ranked candidates:
  #1 DRV-47038  4 min  (0.67 km)
  #2 DRV-26148  8 min  (0.33 km)
  #3 DRV-58423  12 min  (1.52 km)
{
  "request_id": "RQ-0304-005",
  "status": "assigned",
  "driver_id": "DRV-47038",
  "arrival_minutes": 4,
  "arrival_time": "2025-03-04T13:10:45-08:00",
  "model": "openrouter/free",
  "reason": "Lowest road pickup time with no route note making it unreliable"
}
```

## Failure handling

The graph never crashes on a bad model reply or an empty candidate list. Every result has a `status`:

| `status` | When | Result |
|---|---|---|
| `assigned` | Model picked a valid candidate | Model's pick; arrival = request time + that driver's road minutes |
| `assigned_fallback` | Model returned null / "nobody", an ID not on the list, malformed JSON, or the API failed | Rank 1 driver (fastest road pickup); `reason` says why, and `model_said` shows what the model answered |
| `no_driver_available` | No candidate is `available` | `"message": "No drivers available"`, `driver_id`, `arrival_minutes` and `arrival_time` are `null`; the model is not called |

```json
{
  "request_id": "RQ-0304-005",
  "status": "no_driver_available",
  "message": "No drivers available",
  "driver_id": null,
  "arrival_minutes": null,
  "arrival_time": null
}
```

## Prompt excerpt

From `_ask_model` in [`src/ride_dispatch/graph.py`](src/ride_dispatch/graph.py):

```
Ride request {request_id}. Candidate drivers, ranked by road_pickup_minutes (rank 1 = fastest):
{candidates as JSON}

Pick the single best driver to dispatch. Prefer the lowest road_pickup_minutes unless a
route_note makes that time unreliable. Reply with JSON only:
{"driver_id": "<id>", "arrival_minutes": <number>, "reason": "<short>"}
```

## Pushback line

> Prefer the lowest road_pickup_minutes unless a route_note makes that time unreliable.

This is the line that pushes back on straight-line distance. In the example above, `DRV-26148` is physically closest (0.33 km) but needs 8 minutes by road because of a one-way loop, so the model correctly chose `DRV-47038` (0.67 km, 4 min).
