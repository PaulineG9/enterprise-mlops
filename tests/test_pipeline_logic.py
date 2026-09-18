"""
test_pipeline_logic.py
------------------------
Automated quality-verification tests for the data generator and ETL
feature-engineering logic. These exercise the pandas implementations in
`src/etl_pipeline.py` directly, so they run fast and require no Spark
cluster -- suitable for the GitHub Actions CI/CD workflow.

Run with:
    pytest tests/ -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from etl_pipeline import (  # noqa: E402
    NUMERIC_SENSOR_COLS,
    engineer_features_pandas,
    impute_nulls_pandas,
)
from generate_mock_data import generate_dataset  # noqa: E402


def _baseline_asset_rows(asset_id: str, asset_type: str = "Drill-Rig") -> dict:
    """Six steady, low-risk readings for one asset -- used to build a
    realistic baseline population so mean/std-based thresholds behave
    the way they would on a real fleet, not on a two-row toy sample."""
    return {
        "asset_id": [asset_id] * 6,
        "asset_type": [asset_type] * 6,
        "region": ["Pilbara-WA"] * 6,
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="6h"),
        "vibration_mm_s": [1.8, 1.9, 2.0, 2.1, 2.0, 1.95],
        "heat_celsius": [58.0, 59.0, 58.5, 59.0, 58.2, 58.8],
        "pressure_kpa": [120.0] * 6,
        "runtime_hours": [20.0] * 6,
        "cycle_count": list(range(1000, 1006)),
        "age_days": list(range(100, 106)),
        "failure_within_7_days": [0] * 6,
    }


@pytest.fixture
def raw_sample_df():
    """Three steady baseline assets plus one escalating (pre-failure)
    asset, with a couple of deliberate nulls to exercise imputation."""
    baseline_frames = [
        pd.DataFrame(_baseline_asset_rows(f"AST-BASE-{i}")) for i in range(3)
    ]
    high_risk_rows = pd.DataFrame({
        "asset_id": ["AST-HIGH"] * 6,
        "asset_type": ["Haul-Truck"] * 6,
        "region": ["Pilbara-WA"] * 6,
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="6h"),
        "vibration_mm_s": [2.0, np.nan, 5.0, 7.0, 8.5, 9.0],
        "heat_celsius": [60.0, 61.0, np.nan, 95.0, 105.0, 110.0],
        "pressure_kpa": [120.0] * 6,
        "runtime_hours": [20.0] * 6,
        "cycle_count": list(range(2000, 2006)),
        "age_days": list(range(200, 206)),
        "failure_within_7_days": [0, 0, 0, 1, 1, 1],
    })
    return pd.concat(baseline_frames + [high_risk_rows], ignore_index=True)


# --------------------------------------------------------------------------
# generate_mock_data.py
# --------------------------------------------------------------------------

class TestMockDataGenerator:
    def test_schema_matches_expected_columns(self):
        df = generate_dataset(n_assets=5, readings_per_asset=4, seed=1)
        expected_cols = {
            "asset_id", "asset_type", "region", "timestamp", "vibration_mm_s",
            "heat_celsius", "pressure_kpa", "runtime_hours", "cycle_count",
            "age_days", "failure_within_7_days",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_assets_times_readings(self):
        df = generate_dataset(n_assets=10, readings_per_asset=8, seed=1)
        assert len(df) == 80

    def test_failure_label_is_binary(self):
        df = generate_dataset(n_assets=50, readings_per_asset=10, seed=2)
        assert set(df["failure_within_7_days"].unique()).issubset({0, 1})

    def test_generation_is_reproducible_given_same_seed(self):
        df1 = generate_dataset(n_assets=20, readings_per_asset=5, seed=99)
        df2 = generate_dataset(n_assets=20, readings_per_asset=5, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_dataset(n_assets=20, readings_per_asset=5, seed=1)
        df2 = generate_dataset(n_assets=20, readings_per_asset=5, seed=2)
        assert not df1["vibration_mm_s"].equals(df2["vibration_mm_s"])

    def test_contains_some_missing_sensor_values(self):
        # The generator deliberately injects sensor dropout; over enough
        # rows we should see at least some NaNs to clean up downstream.
        df = generate_dataset(n_assets=100, readings_per_asset=20, seed=3)
        assert df[NUMERIC_SENSOR_COLS].isna().any().any()


# --------------------------------------------------------------------------
# etl_pipeline.py -- null imputation
# --------------------------------------------------------------------------

class TestImputeNulls:
    def test_no_nulls_remain_after_imputation(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        assert cleaned[NUMERIC_SENSOR_COLS].isna().sum().sum() == 0

    def test_imputed_value_matches_group_mean(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        # AST-HIGH (asset_type="Haul-Truck") has one NaN vibration
        # reading; the fill value should equal the mean of the OTHER
        # Haul-Truck readings (there's only one Haul-Truck asset here).
        nan_row = raw_sample_df[raw_sample_df["vibration_mm_s"].isna()].index[0]
        haul_truck_non_null = raw_sample_df[
            (raw_sample_df["asset_type"] == "Haul-Truck") & raw_sample_df["vibration_mm_s"].notna()
        ]["vibration_mm_s"]
        expected_fill = haul_truck_non_null.mean()
        assert cleaned.loc[nan_row, "vibration_mm_s"] == pytest.approx(expected_fill)

    def test_does_not_mutate_input_dataframe(self, raw_sample_df):
        original_na_count = raw_sample_df.isna().sum().sum()
        impute_nulls_pandas(raw_sample_df)
        assert raw_sample_df.isna().sum().sum() == original_na_count


# --------------------------------------------------------------------------
# etl_pipeline.py -- feature engineering
# --------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_adds_expected_columns(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        featured = engineer_features_pandas(cleaned)
        for col in ["rolling_vibration_avg", "rolling_heat_avg", "risk_flag"]:
            assert col in featured.columns

    def test_risk_flag_is_binary(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        featured = engineer_features_pandas(cleaned)
        assert set(featured["risk_flag"].unique()).issubset({0, 1})

    def test_high_readings_asset_flagged_higher_risk_than_baseline_asset(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        featured = engineer_features_pandas(cleaned)

        high_risk_asset_flags = featured[featured["asset_id"] == "AST-HIGH"]["risk_flag"].sum()
        baseline_asset_flags = featured[featured["asset_id"] == "AST-BASE-0"]["risk_flag"].sum()

        # AST-HIGH has a clear escalating vibration/heat trend built into
        # the fixture; the AST-BASE-* assets are flat and near baseline.
        assert high_risk_asset_flags > baseline_asset_flags

    def test_rolling_average_is_never_null(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        featured = engineer_features_pandas(cleaned)
        assert featured["rolling_vibration_avg"].isna().sum() == 0
        assert featured["rolling_heat_avg"].isna().sum() == 0

    def test_output_row_count_matches_input(self, raw_sample_df):
        cleaned = impute_nulls_pandas(raw_sample_df)
        featured = engineer_features_pandas(cleaned)
        assert len(featured) == len(raw_sample_df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
