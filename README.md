# ride-dispatch-langgraph

Python + LangGraph project that takes a week (3-9 March 2025) of ride requests and candidate-driver snapshots and picks one driver per request, with a one-line reason.

- `data/ride_requests.csv` – 20 ride requests
- `data/candidate_drivers.csv` – candidate drivers per request

## Graph

```
START -> availability -> pickup_minutes -> eligibility -> rank_drivers -> END
```

1. `availability` – keeps only drivers with `driver_status == "available"`. `on_trip` and `offline` drivers are cut and never reach the later nodes or the model.
2. `pickup_minutes` – drops drivers with no `road_pickup_minutes`, then keeps only the fastest `max_candidates` per request.
3. `eligibility` – keeps drivers whose `status_updated_at` is at most `max_status_age_minutes` before the request time. A status stamped after the request time counts as fresh; an unreadable timestamp counts as not fresh.
4. `rank_drivers` – builds a **checklist** of the drivers that survived, sends it with the ride request to a free OpenRouter model (default `google/gemma-4-31b-it:free`, falls back to `openrouter/free`), and asks for one driver in a fixed shape: `{"driver_id": "<id>", "reason": "<one short line>"}`. The code accepts the reply only if it has exactly those two keys and the `driver_id` is on the checklist. Anything else is rejected and the **fallback** is used: the driver with the fastest `road_pickup_minutes`, which is the drive time to the rider (not the straight-line distance). Arrival time is the request time plus that drive time, computed from the data, never taken from the model.

`rider_id` is never sent to the model.

## Settings and rain

Every threshold lives in one file, [`config/settings.json`](config/settings.json), and is read through `settings.load()` in [`src/ride_dispatch/settings.py`](src/ride_dispatch/settings.py). The agent and the tests both call it, so no threshold is typed anywhere else.

| Setting | Used by | Meaning |
|---|---|---|
| `rain_pickup_margin_minutes` | `rank_drivers` | In rain, only drivers within this many minutes of the fastest road pickup stay on the checklist |
| `max_candidates` | `pickup_minutes` | Fastest drivers kept per request |
| `max_status_age_minutes` | `eligibility` | How old a status may be before the request |
| `max_reason_chars` | `rank_drivers` | Reasons are cut to one line of at most this length |

Raining is an input, not something read from the data. Run with `--rain` to apply the margin:

```bash
python -m src.ride_dispatch.graph RQ-0304-005 --rain
```

With rain on, a model pick outside the margin is on no checklist, so it is rejected like any other id outside it and the fallback picks the fastest driver. Without rain, any eligible driver is on the checklist.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your OPENROUTER_API_KEY
python -m src.ride_dispatch.graph RQ-0304-005
```

## Checking the run order

All three checks run before the model, in this order: `availability`, `pickup_minutes`, `eligibility`, then `rank_drivers`, which makes the one model call. Each node adds its name to a `steps` list in the state, and the command line prints it:

```
steps run, in order: availability -> pickup_minutes -> eligibility -> rank_drivers
drivers: 4 candidates -> 3 available -> 3 after pickup_minutes -> 3 eligible
```

The tests in [`tests/test_graph_order.py`](tests/test_graph_order.py) check the order three ways: they wrap each node and the model call to log when they start, compare the `steps` list, and compare the order LangGraph itself reports. They also check that the model is never called when no driver survives. [`tests/test_choice.py`](tests/test_choice.py) checks the single-driver reply: the fixed shape, rejection of ids outside the checklist (invented, offline, or from another request), the fastest-road-pickup fallback, and the rain margin. Both use a fake model reply, so no API key is needed:

```bash
python -m unittest tests.test_graph_order tests.test_choice -v
```

The tests and the agent read the same numbers from `config/settings.json`. One test changes the rain margin in a temporary settings file and checks that the agent's behaviour changes with it, and another scans the agent and the test files to confirm no threshold is assigned a number in code.

Because each node reads the previous node's output, running them out of order fails with a `KeyError` instead of silently giving a different answer.

## Output

Run on `RQ-0304-005` (4 candidates, 3 available, all 3 pass the other checks):

![Graph output](docs/output.png)

Full output is in [`docs/output.txt`](docs/output.txt). The pick is `DRV-47038`, the fastest by road minutes. `DRV-26148` is closer on the map but is slower by road because of a one-way loop. With `--rain`, the same request puts only the fastest driver on the checklist, and the model answers "Only driver listed".

### OpenRouter's record of this run

The output carries OpenRouter's own IDs for the call. I looked up the generation ID with the API (`python scripts/lookup_generation.py <generation_id>`); the saved record is [`docs/openrouter-record-single-pick.json`](docs/openrouter-record-single-pick.json), with account identifiers removed:

| Field | Value |
|---|---|
| Generation ID | `gen-1794494628-8VTw5doiwipBMmAl5O5J` |
| Request ID | `req-1794494628-uLnsFXJiH4lMfassFs1m` |
| Time | 2026-11-12 14:43:48 UTC |
| Requested model | `openrouter/free` (router) |
| Model that served it | `cohere/north-mini-code-20260617:free` |
| Provider | Cohere, status 200 |
| Tokens | 489 prompt, 919 completion (native); 390 / 996 normalized |
| Finish reason / cost | `stop` / $0 |

The record has no `RQ-0304-005` ID or prompt text, so the link is the generation ID the graph printed. I did not open this call on the Activity page.

### Earlier run: the top-3 version

The next two sections document an earlier run, from before the model was changed to return one driver. They are kept as the record of that run. Its output carried OpenRouter's own IDs for the call, captured from the API response in the same run that produced the `RQ-0304-005` ranking:

```
"generation_id": "gen-1794490384-DISPF4Dnm6KPevQEf61k"
"served_model":  "google/gemma-4-26b-a4b-it:free"      (requested: openrouter/free)
```

I then looked that ID up with OpenRouter's generation API (`python scripts/lookup_generation.py <generation_id>`). The single record, saved in [`docs/openrouter-record.json`](docs/openrouter-record.json) with account identifiers removed, says:

| Field | Value |
|---|---|
| Generation ID | `gen-1794490384-DISPF4Dnm6KPevQEf61k` |
| Request ID | `req-1794490384-I0f7rd1w1ZY0giEMPlsO` |
| Time | 2026-11-12 13:33:04 UTC |
| Requested model | `openrouter/free` (router) |
| Model that served it | `google/gemma-4-26b-a4b-it-20260403:free` |
| Provider | Google AI Studio, status 200 |
| Tokens | 643 prompt, 146 completion (native, as the provider counts them); 375 / 109 in OpenRouter's normalized count |
| Finish reason / cost | `stop` / $0 |

The record has no prompt text and no `RQ-0304-005` ID, because OpenRouter never sees my request IDs. The link is the generation ID: the graph printed it in the output for `RQ-0304-005`, and the record above is what OpenRouter holds for that ID.

### The same call on OpenRouter's Activity page

I also opened this request on the OpenRouter Activity page (Logs > Generations, times shown in EST) and copied its row and its Generation details as text. The API Key column, which is just my key's label, is left out.

```
Date:        Nov 12, 08:33 AM
Model:       Gemma 4 26B A4B (free)
Provider:    Google AI Studio
Cost:        $0.00
Input:       643 tok
Output:      146 tok
Usage type:  Credits
Speed:       37.4 tok/s
Routing overhead: 0 s
Time to first token: 1.11 s
Finish reason: stop

Generation details
Model ID:     google/gemma-4-26b-a4b-it:free
Canonical ID: google/gemma-4-26b-a4b-it-20260403:free
Request ID:   req-1794490384-I0f7rd1w1ZY0giEMPlsO
Generation ID: gen-1794490384-DISPF4Dnm6KPevQEf61k
Streaming:    true
Provider response: Google AI Studio, 200, 1.1 s
Total: 3.9 s
```

Screenshot of that row's Generation details (the page truncates the two IDs on screen; the full values are in the text above). I covered only the label of my API key:

![OpenRouter Generation details for the 08:33 AM request](docs/openrouter-activity-row.png)

Two things in this row look odd at first, so here is what I know:

- **Tokens, 643 vs 375.** The record has two sets of counts. The Activity page shows the provider's native counts (643 in, 146 out), which are the `native_tokens_*` fields in the API record. The `tokens_prompt` / `tokens_completion` fields (375 / 109) are OpenRouter's normalized counts of the same call. Both are in [`docs/openrouter-record.json`](docs/openrouter-record.json).
- **Streaming: true.** My code never sets a streaming option (there is no `stream` anywhere in `src/`), yet both the API record (`streamed: true`) and this page report it. I have not found out why. I only know it is what OpenRouter reports for this call, and it doesn't change the ranking, which the graph read from a normal non-streamed response.

The Request ID and Generation ID are identical to the ones the graph printed and to the API record above, and the model that served it is Gemma 4 26B A4B. The Activity page says "I/O logging is not enabled" for my account, so it shows no prompt or completion text. Because of that, the prompt cannot be checked against this row; the ID is the link.

## Prompt excerpt

From `build_prompt` in [`src/ride_dispatch/graph.py`](src/ride_dispatch/graph.py):

```
Ride request context: {requested_at, pickup_zone, pickup_lat, pickup_lng, pickup_river_side,
                       dropoff_zone, is_bridge_case, distance_time_disagree}

(In rain only:) It is raining, so only drivers within {margin} minutes of the fastest road pickup are listed.

Checklist of drivers you may choose from: [{driver_id, straight_line_km, road_pickup_minutes,
                                            driver_river_side, driver_lat, driver_lng,
                                            status_updated_at, route_note}, ...]

Pick the single best driver for this ride. Prefer the lowest road_pickup_minutes (the drive time to
the rider) unless a route_note makes that time unreliable. The driver_id MUST be one from the
checklist above. Reply with JSON only, in exactly this shape and nothing else:
{"driver_id": "<one id from the checklist>", "reason": "<one short line>"}
```

The road-minutes rule matters because straight-line distance can mislead: in the example above, `DRV-26148` is physically closest (0.33 km) but needs 8 minutes by road because of a one-way loop, so the fallback prefers `DRV-47038` (0.67 km, 4 min).

## Failure handling

The graph does not crash on a bad model reply or an empty candidate list. Every result has a `status`:

| `status` | When | Result |
|---|---|---|
| `assigned` | The model's reply has exactly `driver_id` and `reason`, and the id is on the checklist | The model's driver, its one-line reason, and arrival time from the data |
| `assigned_fallback` | Wrong shape, an id outside the checklist (invented, offline, from another request, or outside the rain margin), malformed JSON, or the API failed | The driver with the fastest `road_pickup_minutes`; `rejected` says why (`bad_shape`, `not_on_checklist` or `model_unavailable`) and what the model said |
| `pending_approval` | No driver is left after `availability`, `pickup_minutes` and `eligibility` | `"message": "Pending approval"`; the model is not called |

## Plan-mode prompt

I designed this with two prompts in plan mode (tidied up here, with filler words removed).

First (two-node graph):

> "Can you make a two-node graph? The first node receives available drivers, so it only takes in drivers with the available tag. On-trip and offline drivers don't come through. The second node takes in their other data, like how far away they are from the trip."

Second (the ranking design this repo implements):

> "Ignore my last message. Instead, I want the code to work like this: the first node, in LangGraph, scans for available drivers and cuts out the unavailable and offline ones before the second node. The second node then has all the available drivers. These are sent, along with the ride request, to OpenRouter, which ranks the top three or four. Create a plan to implement this and let me know."

Third (changed the output to one driver and added the rain rule), tidied up:

> "Send it to the model in the second node and have it return one driver ID and a one-line reason, in a fixed shape the code can read. Reject any driver ID outside the checklist. The fallback picks the fastest by road pickup minutes, the drive time to the rider. When it's raining, only accept drivers within a couple of minutes of the fastest pickup. Store that number in one setting spot, and check that the tests and the agent both read that one number, so no threshold is typed in the test code."

## Pushback

I read the plan and asked what happens if no drivers are available, since it would crash. I looked through Claude Code's plan, saw that it didn't cover what happens when there are zero available drivers, and flagged it myself with this question to get it fixed. My question, tidied up:

> "What happens if the model returns nobody, or starts returning drivers that aren't on the list? What are you going to return to the user, and what are you going to output?"

The plan had no branch for zero candidates: if every driver was `on_trip` or `offline`, nothing reached the model node and it crashed. My question is what led to adding one.

**What the user gets back:**

| Situation | What comes back |
|---|---|
| No drivers left for the request | `status: "pending_approval"`, `message: "Pending approval"`. The model is never called. |
| Model names nobody, or a driver not on the checklist | `status: "assigned_fallback"`: the fastest driver by road pickup minutes, with `rejected` explaining why. Always a real, eligible driver. |
| Normal case | `status: "assigned"`: the model's driver, its one-line reason, and the arrival time computed from the data. |

```json
{
  "request_id": "RQ-0304-005",
  "status": "pending_approval",
  "message": "Pending approval"
}
```

(This section was written when the model still ranked 3-4 drivers. The result now holds one driver, as shown above.)

## OpenRouter usage

Activity from my OpenRouter dashboard (Explore tab, Request Count by Model, minute rollup) showing the graph is calling the free models. On 12 Nov it logged 7 requests across 5 free models (Gemma 4 31B 2, Nemotron 3 Ultra 2, Flint 1, ERNIE-5.5-Flash-Preview 1, North Mini Code 1); the `openrouter/free` fallback routes between free models, which is why more than one appears.

This screenshot is general usage from my earlier live runs and tests, taken before the run shown in the Output section. It does not identify any single request. The record for the `RQ-0304-005` output run is the ID lookup above, not a bar in this chart.

![OpenRouter activity](docs/openrouter-activity-minute.png)
