# ride-dispatch-langgraph

Python + LangGraph project that loads a week (3-9 March 2025) of ride requests and candidate-driver snapshots.

- `data/ride_requests.csv` – 20 ride requests
- `data/candidate_drivers.csv` – candidate drivers per request

```bash
pip install -r requirements.txt
python -m src.ride_dispatch.graph
```
