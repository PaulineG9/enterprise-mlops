"""
generate_mock_data.py
----------------------
Synthetic data generator engine.

Produces a reproducible, privacy-safe telemetry dataset that mimics
sensor readings from industrial / mining / high-capital assets across
multiple regional deployment zones. The simulator injects realistic
failure signatures (rising vibration + heat prior to failure) so the
downstream ETL and model-training stages have a genuine signal to learn.

Usage:
    python scripts/generate_mock_data.py
    python scripts/generate_mock_data.py --rows 20000 --seed 7
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

REGIONS = ["Pilbara-WA", "Hunter-Valley-NSW", "Bowen-Basin-QLD", "Goldfields-WA", "Kalgoorlie-WA"]
ASSET_TYPES = ["Haul-Truck", "Conveyor-Belt", "Crusher", "Drill-Rig", "Pump-Station"]

DEFAULT_OUTPUT = os.path.join("data", "raw", "asset_telemetry.csv")


def _simulate_asset_history(asset_id: int, asset_type: str, region: str,
                             n_readings: int, rng: np.random.Generator,
                             start_date: datetime):
    """Simulate one asset's sensor history, optionally ending in failure.

    Roughly 1 in 6 assets fail by the end of their observation window.
    Failing assets get an escalating vibration/heat trend in the final
    readings so the label is learnable rather than pure noise.
    """
    will_fail = rng.random() < (1 / 6)
    base_vibration = rng.normal(2.5, 0.4)
    base_heat = rng.normal(65, 5)
    base_pressure = rng.normal(120, 10)
    cycle_count = int(rng.integers(500, 50000))
    age_days = int(rng.integers(30, 3650))

    records = []
    for i in range(n_readings):
        timestamp = start_date + timedelta(hours=6 * i)
        frac = i / max(n_readings - 1, 1)

        # Degradation trend only kicks in for assets destined to fail,
        # and only in the back half of the observation window.
        degradation = 0.0
        if will_fail and frac > 0.5:
            degradation = (frac - 0.5) * 2  # 0 -> 1 over back half

        vibration_mm_s = base_vibration + degradation * rng.normal(4.0, 0.6) + rng.normal(0, 0.15)
        heat_celsius = base_heat + degradation * rng.normal(30, 4) + rng.normal(0, 1.2)
        pressure_kpa = base_pressure - degradation * rng.normal(15, 3) + rng.normal(0, 2.0)
        runtime_hours = round(rng.uniform(4, 24), 2)

        # Inject a small amount of missingness to simulate messy raw sensor
        # feeds (dropped packets, sensor dropout) -- this is what the ETL's
        # null-imputation step exists to clean up.
        if rng.random() < 0.03:
            vibration_mm_s = np.nan
        if rng.random() < 0.02:
            heat_celsius = np.nan
        if rng.random() < 0.02:
            pressure_kpa = np.nan

        failure_within_7_days = int(will_fail and frac > 0.85)

        records.append({
            "asset_id": f"AST-{asset_id:05d}",
            "asset_type": asset_type,
            "region": region,
            "timestamp": timestamp.isoformat(),
            "vibration_mm_s": vibration_mm_s,
            "heat_celsius": heat_celsius,
            "pressure_kpa": pressure_kpa,
            "runtime_hours": runtime_hours,
            "cycle_count": cycle_count + i * int(rng.integers(1, 20)),
            "age_days": age_days + i,
            "failure_within_7_days": failure_within_7_days,
        })
    return records


def generate_dataset(n_assets: int = 400, readings_per_asset: int = 40, seed: int = 42) -> pd.DataFrame:
    """Generate the full synthetic telemetry dataset as a DataFrame."""
    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)

    all_records = []
    for asset_id in range(1, n_assets + 1):
        asset_type = rng.choice(ASSET_TYPES)
        region = rng.choice(REGIONS)
        all_records.extend(
            _simulate_asset_history(asset_id, asset_type, region, readings_per_asset, rng, start_date)
        )

    df = pd.DataFrame(all_records)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic asset telemetry data.")
    parser.add_argument("--assets", type=int, default=400, help="Number of distinct assets to simulate.")
    parser.add_argument("--readings-per-asset", type=int, default=40, help="Telemetry readings per asset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path.")
    args = parser.parse_args()

    df = generate_dataset(n_assets=args.assets, readings_per_asset=args.readings_per_asset, seed=args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    n_rows = len(df)
    n_failures = int(df["failure_within_7_days"].sum())
    print(f"Generated {n_rows:,} telemetry rows across {args.assets} assets -> {args.output}")
    print(f"Positive class ('failure_within_7_days'=1): {n_failures:,} rows ({n_failures / n_rows:.2%})")


if __name__ == "__main__":
    main()
