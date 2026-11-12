"""Look up one OpenRouter request by its generation id (the `generation_id` in the graph output).

Usage: python scripts/lookup_generation.py gen-xxxxxxxx
Needs OPENROUTER_API_KEY in .env. Prints OpenRouter's own record of the call.
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

gen_id = sys.argv[1]
for attempt in range(6):  # the record can take a few seconds to appear
    resp = requests.get(
        "https://openrouter.ai/api/v1/generation",
        params={"id": gen_id},
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=30,
    )
    if resp.status_code == 404:
        time.sleep(3)
        continue
    resp.raise_for_status()
    break
else:
    sys.exit(f"No record found for {gen_id}")

import json

print(json.dumps(resp.json().get("data", resp.json()), indent=2))
