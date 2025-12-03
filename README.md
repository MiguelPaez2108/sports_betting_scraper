# Sports Betting Intelligence Platform

Production-ready sports betting prediction system with machine learning models, probability calibration, and advanced backtesting.

## 🌟 Key Features

### 🧠 Advanced Modeling
- **Hybrid Calibration**: Global + league-specific probability calibration
- **XGBoost Classifier**: Trained on 53K+ historical matches  
- **Walk-Forward Validation**: Temporal robustness testing
- **Meta-Calibration**: Weighted ensemble of calibrators

### ⚡ Robust Execution
- **Multiple Stake Strategies**: Fixed, Kelly, Fractional Kelly, Limited Kelly
- **Realistic Backtesting**: Slippage simulation, detailed trade logging
- **Risk Management**: Daily caps, stop-loss, position sizing
- **Betting Filters**: Configurable thresholds (confidence, edge, odds)

### 🛡️ Production Features
- **Feature Pipeline**: Optimized with Parquet caching (<5min nightly)
- **Grid Search**: Automated filter optimization (40-60 bet target)
- **EV Analysis**: Identify profitable probability bins
- **Comprehensive Documentation**: Runbook, architecture diagrams

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run walk-forward validation
python scripts/walk_forward.py --data data/processed/features.parquet --n-folds 5

# Calibrate model
python scripts/calibrate_model.py --model models/xgboost_balanced.json

# Grid search for optimal filters
python scripts/grid_filters.py --predictions data/predictions.parquet --odds data/odds.parquet

# Analyze EV by probability bins
python scripts/analyze_ev_bins.py --predictions data/predictions.parquet --odds data/odds.parquet
```

## 📊 Performance

| Metric | Value | Status |
|--------|-------|--------|
| **ROI (30d)** | +15.57% | ✅ |
| **Sharpe Ratio** | 0.98 | ✅ |
| **Brier Score** | 0.216 | ✅ |
| **Log Loss** | 1.09 | ✅ |
| **Win Rate** | 40.0% | ✅ |
| **Bet Volume** | 50/month | ✅ |

## 🏗️ Architecture

### Core Components

**CalibratorManager** (`src/models/calibrator_manager.py`)
- Global + league-specific calibrators
- Lazy loading for performance
- Meta-calibration (weighted ensemble)

**FeaturePipeline** (`src/feature_engineering/pipeline.py`)
- ELO, Form, H2H, Poisson features
- Parquet-based caching
- Vectorized computation (<5min nightly)

**EnhancedBacktester** (`src/backtesting/backtester.py`)
- Realistic slippage simulation
- Multiple stake strategies
- Daily caps and stop-loss

**StakeStrategies** (`src/backtesting/stake_strategies.py`)
- Fixed, Kelly, Fractional Kelly, Limited Kelly
- Volatility-based adjustments

## 📁 Project Structure

```
├── src/
│   ├── models/              # CalibratorManager, model training
│   ├── backtesting/         # Backtester, stake strategies
│   ├── feature_engineering/ # Feature pipeline, calculators
│   └── ml_models/          # ML infrastructure
├── scripts/                # Executable scripts
│   ├── calibrate_model.py  # Model calibration
│   ├── walk_forward.py     # Temporal validation
│   ├── grid_filters.py     # Filter optimization
│   └── analyze_ev_bins.py  # EV analysis
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
│   ├── runbook.md         # Operational guide
│   └── architecture.md    # System design
├── models/                 # Saved models
└── analysis/              # Analysis results
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/test_calibrator_manager.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📈 Model Training Workflow

1. **Data Preparation**: `python scripts/prepare_data.py`
2. **Feature Engineering**: Feature pipeline with caching
3. **Model Training**: `python scripts/train_balanced_model.py`
4. **Calibration**: `python scripts/calibrate_model.py`
5. **Validation**: `python scripts/walk_forward.py`
6. **Filter Optimization**: `python scripts/grid_filters.py`
7. **Backtesting**: Enhanced backtester with realistic simulation

## 📚 Documentation

- **[Runbook](docs/runbook.md)**: Operational procedures, emergency response
- **[Architecture](docs/architecture.md)**: System design, data flow diagrams
- **[Task Tracking](https://github.com/MiguelPaez2108/sports_betting_scraper)**: GitHub repository

## 📝 License

MIT License
