#!/bin/bash
# Nightly ETL job for feature updates

set -e

echo "========================================"
echo "NIGHTLY ETL JOB - $(date)"
echo "========================================"

# 1. Fetch latest match results
echo "1. Fetching latest results..."
python scripts/fetch_results.py --days 7

# 2. Update feature cache
echo "2. Updating feature cache..."
python -c "
from src.feature_engineering.pipeline import FeaturePipeline
import pandas as pd

# Load recent matches
matches = pd.read_parquet('data/processed/matches.parquet')

# Update features
pipeline = FeaturePipeline()
features = pipeline.transform(matches, cache_prefix='production')

# Save
features.to_parquet('data/processed/features_latest.parquet', index=False)
print('✓ Features updated')
"

# 3. Generate daily report
echo "3. Generating daily report..."
python scripts/generate_daily_report.py

# 4. Check for drift
echo "4. Checking for data drift..."
python scripts/detect_drift.py --threshold 0.1

echo "========================================"
echo "ETL JOB COMPLETE - $(date)"
echo "========================================"
