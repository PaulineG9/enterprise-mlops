"""
etl_pipeline.py
----------------
Scalable data ingestion & feature engineering.

Production path (on Databricks): reads raw telemetry from Azure Blob
Storage Gen2, cleans it with PySpark, and writes ACID-compliant Delta
Lake tables partitioned by region.

Local / CI path: when PySpark is not available (e.g. a laptop or a
GitHub Actions runner with no Spark cluster), the same feature-
engineering logic runs on pandas instead, writing Parquet in place of
Delta. This keeps the transformation logic testable everywhere while
the Spark path stays what actually executes in production.

The pure feature-engineering functions (`impute_nulls`,
`engineer_features`) are backend-agnostic in spirit: the pandas
implementations below are what `tests/test_pipeline_logic.py` exercises
directly, and the PySpark implementation mirrors the same rules.

Usage:
    python src/etl_pipeline.py --input data/raw/asset_telemetry.csv --output data/processed/asset_features
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

NUMERIC_SENSOR_COLS = ["vibration_mm_s", "heat_celsius", "pressure_kpa"]
ROLLING_WINDOW = 5


# --------------------------------------------------------------------------
# Pandas implementation (local dev / CI / unit tests)
# --------------------------------------------------------------------------

def impute_nulls_pandas(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Fill missing sensor readings with the per-asset-type baseline mean.

    Falls back to the global column mean if an asset type has no
    non-null readings at all (e.g. a tiny test fixture).
    """
    columns = columns or NUMERIC_SENSOR_COLS
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        group_mean = df.groupby("asset_type")[col].transform("mean")
        df[col] = df[col].fillna(group_mean)
        df[col] = df[col].fillna(df[col].mean())
    return df


def engineer_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window risk features per asset.

    - rolling_vibration_avg / rolling_heat_avg: short-window rolling means
      that smooth out sensor noise and reveal degradation trends.
    - risk_flag: 1 when both vibration and heat rolling averages exceed
      a fixed operating threshold, i.e. an early-warning signal that a
      failure may be approaching.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["asset_id", "timestamp"])

    df["rolling_vibration_avg"] = (
        df.groupby("asset_id")["vibration_mm_s"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df["rolling_heat_avg"] = (
        df.groupby("asset_id")["heat_celsius"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    vibration_threshold = df["vibration_mm_s"].mean() + df["vibration_mm_s"].std()
    heat_threshold = df["heat_celsius"].mean() + df["heat_celsius"].std()

    df["risk_flag"] = (
        (df["rolling_vibration_avg"] > vibration_threshold)
        & (df["rolling_heat_avg"] > heat_threshold)
    ).astype(int)

    return df.reset_index(drop=True)


def run_pandas_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = impute_nulls_pandas(df)
    df = engineer_features_pandas(df)

    os.makedirs(output_path, exist_ok=True)
    parquet_path = os.path.join(output_path, "asset_features.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[pandas backend] Wrote {len(df):,} rows -> {parquet_path}")
    except ImportError:
        # pyarrow/fastparquet not installed in this environment -- fall
        # back to CSV so the pipeline still produces a usable artifact.
        csv_path = os.path.join(output_path, "asset_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"[pandas backend] pyarrow unavailable; wrote CSV instead -> {csv_path}")

    return df


# --------------------------------------------------------------------------
# PySpark / Delta Lake implementation (Databricks production path)
# --------------------------------------------------------------------------

def run_spark_pipeline(input_path: str, output_path: str):
    """Run the same cleaning + feature-engineering logic on a Spark cluster.

    Reads raw CSV telemetry (in production, this would point at an
    abfss:// path on Azure Data Lake Storage Gen2), imputes nulls per
    asset type, computes rolling risk features with a window function,
    and writes the result as a Delta table partitioned by region.
    """
    spark = (
        SparkSession.builder
        .appName("EnterpriseAssetETL")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    raw_df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

    # --- Null imputation: per-asset-type baseline mean ---
    means = raw_df.groupBy("asset_type").agg(
        *[F.mean(F.col(c)).alias(f"{c}_mean") for c in NUMERIC_SENSOR_COLS]
    )
    df = raw_df.join(means, on="asset_type", how="left")
    for c in NUMERIC_SENSOR_COLS:
        df = df.withColumn(c, F.coalesce(F.col(c), F.col(f"{c}_mean")))
    df = df.drop(*[f"{c}_mean" for c in NUMERIC_SENSOR_COLS])

    # --- Rolling risk features ---
    asset_window = (
        Window.partitionBy("asset_id")
        .orderBy("timestamp")
        .rowsBetween(-(ROLLING_WINDOW - 1), 0)
    )
    df = df.withColumn("rolling_vibration_avg", F.avg("vibration_mm_s").over(asset_window))
    df = df.withColumn("rolling_heat_avg", F.avg("heat_celsius").over(asset_window))

    stats = df.select(
        F.mean("vibration_mm_s").alias("v_mean"), F.stddev("vibration_mm_s").alias("v_std"),
        F.mean("heat_celsius").alias("h_mean"), F.stddev("heat_celsius").alias("h_std"),
    ).collect()[0]
    vibration_threshold = stats["v_mean"] + stats["v_std"]
    heat_threshold = stats["h_mean"] + stats["h_std"]

    df = df.withColumn(
        "risk_flag",
        ((F.col("rolling_vibration_avg") > vibration_threshold)
         & (F.col("rolling_heat_avg") > heat_threshold)).cast("int"),
    )

    (
        df.write.format("delta")
        .mode("overwrite")
        .partitionBy("region")
        .save(output_path)
    )
    print(f"[spark backend] Wrote Delta table -> {output_path}")
    spark.stop()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean raw telemetry and engineer risk features.")
    parser.add_argument("--input", type=str, default=os.path.join("data", "raw", "asset_telemetry.csv"))
    parser.add_argument("--output", type=str, default=os.path.join("data", "processed", "asset_features"))
    parser.add_argument("--force-pandas", action="store_true",
                         help="Force the pandas fallback even if PySpark is available.")
    args = parser.parse_args()

    if PYSPARK_AVAILABLE and not args.force_pandas:
        run_spark_pipeline(args.input, args.output)
    else:
        if not PYSPARK_AVAILABLE:
            print("PySpark not available in this environment -- running the pandas fallback pipeline.")
        run_pandas_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
