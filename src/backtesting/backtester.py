"""
Enhanced Backtester with realistic simulation.

Features:
- Multiple stake strategies
- Slippage modeling
- Detailed trade logging
- Betting filters (min_conf, min_edge, odds_range)
- Daily caps and stop-loss
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .stake_strategies import StakeStrategy, FractionalKelly


@dataclass
class BettingFilters:
    """Configurable betting filters."""
    min_confidence: float = 0.48
    min_edge: float = 0.02
    min_odds: float = 1.30
    max_odds: float = 5.00
    max_book_margin: float = 0.15


@dataclass
class Trade:
    """Record of a single bet."""
    match_id: str
    date: datetime
    selection: str  # 'H', 'D', 'A'
    stake: float
    odds: float
    prob: float
    edge: float
    result: str  # 'WIN', 'LOSS'
    profit: float
    bankroll_after: float
    metadata: Dict = field(default_factory=dict)


class EnhancedBacktester:
    """
    Realistic backtesting engine with slippage and risk management.
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        stake_strategy: Optional[StakeStrategy] = None,
        filters: Optional[BettingFilters] = None,
        slippage_pct: float = 0.01,
        slippage_model: Optional[Callable] = None,
        daily_loss_limit: float = 0.05,  # 5% of bankroll
        daily_stake_cap: float = 0.06   # 6% of bankroll
    ):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.stake_strategy = stake_strategy or FractionalKelly(fraction=0.25)
        self.filters = filters or BettingFilters()
        self.slippage_pct = slippage_pct
        self.slippage_model = slippage_model
        self.daily_loss_limit = daily_loss_limit
        self.daily_stake_cap = daily_stake_cap
        
        self.trades: List[Trade] = []
        self.daily_history: List[Dict] = []
        self._daily_loss = 0.0
        self._daily_stake = 0.0
        self._current_date = None
    
    def _apply_slippage(self, odds: float) -> float:
        """Apply slippage to odds (worse odds for bettor)."""
        if self.slippage_model:
            return self.slippage_model(odds)
        
        # Simple model: reduce odds by slippage percentage
        return odds * (1 - self.slippage_pct)
    
    def _check_filters(self, prob: float, odds: float, edge: float, book_margin: float) -> bool:
        """Check if bet passes all filters."""
        if prob < self.filters.min_confidence:
            return False
        if edge < self.filters.min_edge:
            return False
        if odds < self.filters.min_odds or odds > self.filters.max_odds:
            return False
        if book_margin > self.filters.max_book_margin:
            return False
        return True
    
    def _check_daily_limits(self, stake: float) -> bool:
        """Check if bet exceeds daily limits."""
        # Check daily loss limit
        if self._daily_loss >= self.initial_bankroll * self.daily_loss_limit:
            return False
        
        # Check daily stake cap
        if self._daily_stake + stake > self.initial_bankroll * self.daily_stake_cap:
            return False
        
        return True
    
    def _reset_daily_counters(self):
        """Reset daily tracking counters."""
        self._daily_loss = 0.0
        self._daily_stake = 0.0
    
    def _process_match(self, row: pd.Series) -> None:
        """Process a single match and decide whether to bet."""
        # Reset daily counters if new day
        match_date = pd.to_datetime(row['date']).date() if 'date' in row else None
        if match_date and match_date != self._current_date:
            if self._current_date:
                self._save_daily_summary()
            self._current_date = match_date
            self._reset_daily_counters()
        
        # Extract probabilities and odds
        probs = {
            'H': row.get('pred_home', 0),
            'D': row.get('pred_draw', 0),
            'A': row.get('pred_away', 0)
        }
        
        odds_dict = {
            'H': row.get('odds_home', 0),
            'D': row.get('odds_draw', 0),
            'A': row.get('odds_away', 0)
        }
        
        # Calculate book margin
        total_imp = sum(1/o for o in odds_dict.values() if o > 0)
        book_margin = total_imp - 1.0 if total_imp > 0 else 999
        
        # Find best bet
        best_selection = None
        best_edge = -999
        
        for selection in ['H', 'D', 'A']:
            prob = probs[selection]
            odds = odds_dict[selection]
            
            if odds <= 0:
                continue
            
            imp_prob = (1/odds) / total_imp if total_imp > 0 else 0
            edge = prob - imp_prob
            
            if edge > best_edge and self._check_filters(prob, odds, edge, book_margin):
                best_edge = edge
                best_selection = selection
        
        # Place bet if profitable
        if best_selection:
            self._place_bet(
                row,
                best_selection,
                probs[best_selection],
                odds_dict[best_selection],
                best_edge
            )
    
    def _place_bet(
        self,
        row: pd.Series,
        selection: str,
        prob: float,
        odds: float,
        edge: float
    ) -> None:
        """Place a bet and record the trade."""
        # Apply slippage
        actual_odds = self._apply_slippage(odds)
        
        # Calculate stake
        stake = self.stake_strategy.compute_stake(
            bankroll=self.bankroll,
            edge=edge,
            odds=actual_odds,
            prob=prob
        )
        
        # Check daily limits
        if not self._check_daily_limits(stake):
            return
        
        # Check if we have enough bankroll
        if stake > self.bankroll:
            stake = self.bankroll
        
        if stake <= 0:
            return
        
        # Determine result
        actual_result = row.get('FTR', 'U')
        won = (actual_result == selection)
        
        # Calculate profit
        if won:
            profit = stake * (actual_odds - 1)
        else:
            profit = -stake
            self._daily_loss += abs(profit)
        
        # Update bankroll
        self.bankroll += profit
        self._daily_stake += stake
        
        # Record trade
        trade = Trade(
            match_id=row.get('match_id', ''),
            date=pd.to_datetime(row['date']) if 'date' in row else datetime.now(),
            selection=selection,
            stake=stake,
            odds=actual_odds,
            prob=prob,
            edge=edge,
            result='WIN' if won else 'LOSS',
            profit=profit,
            bankroll_after=self.bankroll,
            metadata={
                'original_odds': odds,
                'slippage': odds - actual_odds,
                'book_margin': row.get('book_margin', 0)
            }
        )
        
        self.trades.append(trade)
    
    def _save_daily_summary(self):
        """Save daily performance summary."""
        if self._current_date:
            daily_trades = [t for t in self.trades if t.date.date() == self._current_date]
            
            if daily_trades:
                self.daily_history.append({
                    'date': self._current_date,
                    'num_bets': len(daily_trades),
                    'total_stake': sum(t.stake for t in daily_trades),
                    'total_profit': sum(t.profit for t in daily_trades),
                    'win_rate': sum(1 for t in daily_trades if t.result == 'WIN') / len(daily_trades),
                    'bankroll': self.bankroll
                })
    
    def run(self, matches_df: pd.DataFrame, predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run backtest on historical data.
        
        Args:
            matches_df: Historical matches with odds and results
            predictions_df: Model predictions
            
        Returns:
            Tuple of (metrics_df, trades_df)
        """
        # Merge matches with predictions
        df = pd.merge(matches_df, predictions_df, on='match_id', how='inner')
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Process each match
        for _, row in df.iterrows():
            self._process_match(row)
        
        # Save final daily summary
        if self._current_date:
            self._save_daily_summary()
        
        return self._compute_metrics(), self._trades_to_dataframe()
    
    def _compute_metrics(self) -> pd.DataFrame:
        """Compute performance metrics."""
        if not self.trades:
            return pd.DataFrame()
        
        total_profit = sum(t.profit for t in self.trades)
        total_stake = sum(t.stake for t in self.trades)
        wins = sum(1 for t in self.trades if t.result == 'WIN')
        
        roi = (total_profit / total_stake) if total_stake > 0 else 0
        win_rate = wins / len(self.trades) if self.trades else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.profit / t.stake for t in self.trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0
        
        metrics = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'num_bets': len(self.trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'avg_stake': total_stake / len(self.trades) if self.trades else 0,
            'avg_odds': np.mean([t.odds for t in self.trades]) if self.trades else 0
        }
        
        return pd.DataFrame([metrics])
    
    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades list to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'match_id': t.match_id,
                'date': t.date,
                'selection': t.selection,
                'stake': t.stake,
                'odds': t.odds,
                'prob': t.prob,
                'edge': t.edge,
                'result': t.result,
                'profit': t.profit,
                'bankroll': t.bankroll_after
            }
            for t in self.trades
        ])
