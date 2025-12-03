# Sports Betting Model - Architecture Documentation

## System Overview

The Sports Betting Intelligence Platform is a production-grade system for identifying profitable betting opportunities using machine learning and statistical analysis.

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ API-Fútbol│  │  Bet365  │  │ Betfair  │              │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘              │
│        └──────────────┴──────────────┘                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   ELO    │  │   Form   │  │   H2H    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│  ┌──────────┐  ┌──────────────────────────┐            │
│  │ Poisson  │  │  Feature Pipeline (Cache) │            │
│  └──────────┘  └──────────────────────────┘            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 ML PREDICTION                            │
│  ┌────────────────────────────────────────┐             │
│  │  XGBoost Model (Trained on 53K matches)│             │
│  └──────────────────┬─────────────────────┘             │
│                     │                                    │
│  ┌──────────────────▼─────────────────────┐             │
│  │   CalibratorManager                    │             │
│  │   ├─ Global (Isotonic)                 │             │
│  │   └─ Per-League (CL, E0, SP1, etc.)    │             │
│  └────────────────────────────────────────┘             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DECISION ENGINE                             │
│  ┌──────────────────────────────────────────┐           │
│  │  Betting Filters                         │           │
│  │  ├─ min_confidence: 0.48                 │           │
│  │  ├─ min_edge: 0.02                       │           │
│  │  └─ odds_range: (1.30, 5.00)             │           │
│  └──────────────────┬───────────────────────┘           │
│                     │                                    │
│  ┌──────────────────▼───────────────────────┐           │
│  │  Stake Strategy (Fractional Kelly 0.25x) │           │
│  └──────────────────┬───────────────────────┘           │
│                     │                                    │
│  ┌──────────────────▼───────────────────────┐           │
│  │  Risk Management                         │           │
│  │  ├─ Daily loss limit: 5%                 │           │
│  │  ├─ Daily stake cap: 6%                  │           │
│  │  └─ Max single bet: 2%                   │           │
│  └──────────────────────────────────────────┘           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  EXECUTION                               │
│  ┌──────────────────────────────────────────┐           │
│  │  OrderManager                            │           │
│  │  └─ Betfair Connector (Simulation Mode)  │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline
1. **Historical Data** → Feature Engineering → **Feature Matrix**
2. **Feature Matrix** + Labels → XGBoost Training → **Base Model**
3. **Base Model** + Validation Data → Calibration → **Calibrated Model**
4. **Calibrated Model** → Walk-Forward Validation → **Performance Metrics**

### Inference Pipeline (Live)
1. **Upcoming Matches** → Feature Engineering → **Features**
2. **Features** → Calibrated Model → **Probabilities**
3. **Probabilities** + Odds → Decision Engine → **Bet Signals**
4. **Bet Signals** → Risk Management → **Approved Bets**
5. **Approved Bets** → Order Manager → **Execution**

## Key Design Decisions

### 1. Calibration Strategy
**Decision**: Hybrid (Global + Per-League)  
**Rationale**: 
- Global calibrator ensures baseline accuracy
- League-specific calibrators capture unique characteristics (e.g., Champions League)
- Fallback to global prevents overfitting on small samples

### 2. Stake Strategy
**Decision**: Fractional Kelly (0.25x)  
**Rationale**:
- Full Kelly too aggressive (high variance)
- Quarter Kelly balances growth and safety
- Hard limits prevent catastrophic losses

### 3. Feature Separation
**Decision**: Odds features only in inference  
**Rationale**:
- Prevents data leakage during training
- Odds contain market wisdom (useful signal)
- Computed on-the-fly during prediction

### 4. Caching Strategy
**Decision**: Parquet-based feature cache  
**Rationale**:
- Fast read/write (columnar format)
- Efficient storage (compression)
- Easy to partition (by season/league)

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Brier Score | < 0.22 | 0.216 ✅ |
| Log Loss | < 1.10 | 1.09 ✅ |
| ROI (30d) | > 10% | 15.6% ✅ |
| Sharpe Ratio | > 0.8 | 0.98 ✅ |
| Bet Volume | 40-60/mo | 50 ✅ |
| ETL Runtime | < 5 min | 3.2 min ✅ |

## Technology Stack

- **Language**: Python 3.11
- **ML Framework**: XGBoost, scikit-learn
- **Data**: Pandas, NumPy, Parquet
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest
- **Version Control**: Git
- **Deployment**: Docker (planned)

## Security Considerations

1. **API Keys**: Stored in `.env`, never committed
2. **Audit Logs**: Immutable JSON logs for all bets
3. **Rate Limiting**: Respect API limits (Betfair, API-Fútbol)
4. **Bankroll Protection**: Hard limits enforced at multiple levels

## Future Enhancements

- [ ] Ensemble models (LightGBM, CatBoost)
- [ ] In-play betting (requires sub-second latency)
- [ ] Automated retraining pipeline
- [ ] Real-time monitoring dashboard
- [ ] Multi-bookmaker arbitrage detection
