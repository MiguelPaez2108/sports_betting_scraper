# Sports Betting Model - Operational Runbook

## 🚨 Emergency Procedures

### Critical Alerts

#### 1. ROI Drop Below -5% (30-day rolling)
**Severity**: CRITICAL  
**Action**:
1. **PAUSE** all live betting immediately
2. Check for data drift in top features
3. Review recent model predictions vs actual results
4. Verify odds feed is working correctly
5. Run walk-forward validation on recent data

**Commands**:
```bash
# Pause betting (set flag)
echo "PAUSED" > data/status/betting_status.txt

# Check recent performance
python scripts/analyze_recent_performance.py --days 30

# Run drift detection
python scripts/detect_drift.py --reference data/train --current data/recent
```

#### 2. Fill Rate < 70%
**Severity**: HIGH  
**Action**:
1. Check connector status (Betfair/Pinnacle)
2. Verify odds feed latency
3. Review slippage logs
4. Check if filters are too restrictive

#### 3. Daily Loss Limit Hit
**Severity**: MEDIUM  
**Action**:
1. Automatic pause (handled by backtester)
2. Review trades from today
3. Check if stop-loss triggered correctly
4. Resume next day automatically

---

## 📊 Daily Operations

### Morning Checklist (9:00 AM)
- [ ] Check overnight job status
- [ ] Review yesterday's trades
- [ ] Verify data pipeline ran successfully
- [ ] Check model calibration metrics
- [ ] Review any alerts

### Nightly ETL Job
**Schedule**: 2:00 AM daily  
**Duration**: ~5 minutes  
**Script**: `scripts/run_nightly_etl.sh`

**Steps**:
1. Fetch new match results
2. Update ELO ratings
3. Recompute rolling features
4. Update feature cache
5. Generate daily report

**Monitoring**:
```bash
# Check last run status
cat logs/etl/latest.log

# Verify feature cache updated
ls -lh data/cache/*.parquet
```

---

## 🔧 Maintenance Tasks

### Weekly (Sunday)
- [ ] Review walk-forward validation results
- [ ] Check calibration drift
- [ ] Backup models and calibrators
- [ ] Review top profitable/unprofitable bets

### Monthly
- [ ] Retrain model on latest data
- [ ] Update calibrators
- [ ] Run full grid search for filters
- [ ] Generate performance report

### Quarterly
- [ ] Full system audit
- [ ] Review and update risk limits
- [ ] Optimize feature pipeline
- [ ] Update documentation

---

## 🐛 Common Issues

### Issue: Model predictions seem off
**Symptoms**: Win rate drops significantly, edge calculations negative  
**Diagnosis**:
```bash
python scripts/validate_predictions.py --recent 100
```
**Solution**:
1. Check if features are being computed correctly
2. Verify calibrator is loaded properly
3. Run prediction on known test cases
4. Compare with baseline model

### Issue: Backtester shows 0 bets
**Symptoms**: No trades in backtest output  
**Diagnosis**: Filters too restrictive  
**Solution**:
```bash
# Run with relaxed filters
python scripts/backtest.py --min-conf 0.40 --min-edge 0.00
```

### Issue: Cache corruption
**Symptoms**: Feature pipeline crashes, NaN values  
**Solution**:
```bash
# Clear cache and rebuild
python -c "from src.feature_engineering.pipeline import FeaturePipeline; FeaturePipeline().clear_cache()"
python scripts/run_nightly_etl.sh
```

---

## 📈 Performance Monitoring

### Key Metrics Dashboard

**Calibration Quality**:
- Brier Score: < 0.22 (GOOD)
- Log Loss: < 1.10 (GOOD)
- Calibration curve: R² > 0.95

**Betting Performance**:
- ROI (30d): > 10% (TARGET)
- Sharpe Ratio: > 0.8 (TARGET)
- Win Rate: 40-50% (EXPECTED)
- Bet Volume: 40-60/month (TARGET)

**System Health**:
- ETL Job Success Rate: > 99%
- Fill Rate: > 80%
- Prediction Latency: < 100ms

---

## 🚀 Deployment

### Phase A: Dry-Run (Week 1-2)
- Stake: 0% (simulation only)
- Monitor: All metrics
- Goal: Validate execution flow

### Phase B: Micro-Stakes (Week 3-4)
- Stake: 0.5% of bankroll
- Monitor: ROI, fill rate, slippage
- Criteria to proceed:
  - ROI > 5%
  - Fill rate > 75%
  - No critical errors

### Phase C: Production (Month 2+)
- Stake: 2-5% of bankroll (Kelly-based)
- Full monitoring
- Automated alerts

---

## 📞 Contacts

**On-Call Rotation**: [TBD]  
**Escalation**: [TBD]  
**Slack Channel**: #betting-model-alerts

---

## 🔐 Security

### API Keys
- Stored in `.env` (never commit)
- Rotate every 90 days
- Use separate keys for dev/prod

### Audit Logs
- Location: `logs/audit/`
- Retention: 1 year
- Format: JSON (immutable)

---

## 📝 Change Log

| Date | Change | Impact |
|------|--------|--------|
| 2024-12-03 | Initial runbook | N/A |
