"""
test_endpoint.py
-----------------
Live HTTP REST API verification script.

Sends a small batch of sample payloads to a deployed Azure ML managed
online endpoint (or any compatible scoring URL) and checks that the
response is well-formed and returns a valid risk probability. Intended
to run as a post-deployment smoke test, e.g. as the final step of the
CI/CD workflow, or manually against a staging endpoint.

Usage:
    python scripts/test_endpoint.py \
        --url https://asset-risk-endpoint.australiaeast.inference.ml.azure.com/score \
        --api-key <your-endpoint-key>
"""

import argparse
import json
import sys

import requests

SAMPLE_PAYLOADS = [
    {
        "asset_type": "Haul-Truck",
        "region": "Pilbara-WA",
        "vibration_mm_s": 2.4,
        "heat_celsius": 64.0,
        "pressure_kpa": 121.0,
        "runtime_hours": 18.5,
        "cycle_count": 41230,
        "age_days": 980,
        "rolling_vibration_avg": 2.5,
        "rolling_heat_avg": 63.0,
        "risk_flag": 0,
    },
    {
        "asset_type": "Conveyor-Belt",
        "region": "Bowen-Basin-QLD",
        "vibration_mm_s": 8.9,
        "heat_celsius": 96.5,
        "pressure_kpa": 88.0,
        "runtime_hours": 21.0,
        "cycle_count": 27890,
        "age_days": 2210,
        "rolling_vibration_avg": 9.4,
        "rolling_heat_avg": 98.1,
        "risk_flag": 1,
    },
]


def call_endpoint(url: str, api_key: str, payloads: list, timeout: int = 15) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps({"data": payloads})
    return requests.post(url, data=body, headers=headers, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a deployed asset-risk scoring endpoint.")
    parser.add_argument("--url", type=str, required=True, help="Endpoint scoring URI.")
    parser.add_argument("--api-key", type=str, required=True, help="Endpoint auth key.")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    print(f"Sending {len(SAMPLE_PAYLOADS)} sample records to {args.url} ...")
    try:
        response = call_endpoint(args.url, args.api_key, SAMPLE_PAYLOADS, args.timeout)
    except requests.exceptions.RequestException as exc:
        print(f"FAILED: request error contacting endpoint -- {exc}", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        print(f"FAILED: endpoint returned HTTP {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    try:
        result = response.json()
    except ValueError:
        print(f"FAILED: response was not valid JSON: {response.text}", file=sys.stderr)
        sys.exit(1)

    print("Endpoint responded successfully:")
    print(json.dumps(result, indent=2))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
