#!/usr/bin/env python3
"""Quick test to verify data_processor works with and without Spark."""

import data_processor

print("Testing data_processor...")

# Test 1: Try to init Spark (might fail without Java gateway)
print("\n1. Testing init_spark()...")
spark = data_processor.init_spark()
print(f"   Spark available: {spark is not None}")

# Test 2: Generate mock data (should always work, returns tuple)
print("\n2. Testing get_mock_data(spark=None, days=30, regions=10)...")
sdf, pdf = data_processor.get_mock_data(spark=None, days=30, regions=10)
print(f"   Spark DF: {sdf is not None}")
print(f"   Pandas DF shape: {pdf.shape}")
print(f"   Columns: {list(pdf.columns)}")

# Test 3: Try with Spark if available
if spark is not None:
    print("\n3. Testing get_mock_data(spark=spark, days=30, regions=10)...")
    sdf2, pdf2 = data_processor.get_mock_data(spark=spark, days=30, regions=10)
    print(f"   Spark DF: {sdf2 is not None}")
    print(f"   Pandas DF shape: {pdf2.shape}")

# Test 4: Test pandas fallback functions
print("\n4. Testing pandas fallback functions...")
summary = data_processor.compute_global_summary_from_pdf(pdf)
print(f"   Global summary: total_cases={summary['total_cases']}, recovery_rate={summary['recovery_rate']:.2%}")

ts = data_processor.compute_time_series_from_pdf(pdf)
print(f"   Time series: {len(ts)} days")

region_df = data_processor.compute_region_aggregations_from_pdf(pdf)
print(f"   Region aggregations: {len(region_df)} regions")
print(f"   Region columns: {list(region_df.columns)}")

print("\n✓ All tests passed!")
