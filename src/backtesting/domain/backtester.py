"""
Backtesting Engine

Core backtesting infrastructure for evaluating betting strategies
with historical data and real odds.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class BettingTransaction:
    """Represents a single bet transaction"""
    
    def __init__(
        self,
        match_id: int,
        date: datetime,
        home_team: str,
        away_team: str,
        prediction: int,  # 0=Away, 1=Draw, 2=Home
        actual_result: int,
        odds: float,
        stake: float,
        model_probability: float
    ):
        self.match_id = match_id
        self.date = date
        self.home_team = home_team
        self.away_team = away_team
        self.prediction = prediction
        self.actual_result = actual_result
        self.odds = odds
        self.stake = stake
        self.model_probability = model_probability
        
        # Calculate outcome
        self.won = (prediction == actual_result)
        self.returns = stake * odds if self.won else 0.0
        self.profit = self.returns - stake
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'match_id': self.match_id,
            'date': self.date,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'prediction': self.prediction,
            'actual_result': self.actual_result,
            'odds': self.odds,
            'stake': self.stake,
            'model_probability': self.model_probability,
            'won': self.won,
            'returns': self.returns,
            'profit': self.profit
        }


class Backtester:
    """
    Backtesting engine for betting strategies
    
    Features:
    - Load historical matches with odds
    - Run model predictions
    - Simulate bets with different strategies
    - Calculate performance metrics
    """
    
    def __init__(self, model, db_config: Dict):
        """
        Initialize backtester
        
        Args:
            model: Trained ML model with predict_proba method
            db_config: Database configuration
        """
        self.model = model
        self.db_config = db_config
        self.transactions: List[BettingTransaction] = []
    
    def load_test_data_with_odds(
        self,
        start_date: str,
        end_date: str,
        leagues: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load historical matches with Bet365 odds from CSVs
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            leagues: List of league codes
        
        Returns:
            DataFrame with matches and odds
        """
        logger.info(f"Loading test data from {start_date} to {end_date}")
        
        if leagues is None:
            leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']
        
        all_matches = []
        
        # Load CSVs for each league
        for league in leagues:
            csv_files = list(Path('data/historical').glob(f'{league}_*.csv'))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Convert date
                    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                    
                    # Filter by date range
                    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                    
                    # Keep only matches with odds
                    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
                    if all(col in df.columns for col in required_cols):
                        df = df[required_cols].copy()
                        df = df.dropna(subset=['B365H', 'B365D', 'B365A'])
                        df['League'] = league
                        all_matches.append(df)
                    
                except Exception as e:
                    logger.warning(f"Error loading {csv_file}: {e}")
        
        if not all_matches:
            logger.error("No matches found with odds")
            return pd.DataFrame()
        
        # Combine all matches
        combined = pd.concat(all_matches, ignore_index=True)
        combined = combined.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined)} matches with odds")
        
        return combined
    
    def run_backtest(
        self,
        test_data: pd.DataFrame,
        features_df: pd.DataFrame,
        strategy: str = 'fixed',
        initial_bankroll: float = 10000.0,
        fixed_stake: float = 100.0,
        min_confidence: float = 0.0,
        min_edge: float = 0.0
    ) -> Dict:
        """
        Run backtest simulation
        
        Args:
            test_data: DataFrame with matches and odds
            features_df: DataFrame with calculated features
            strategy: Betting strategy ('fixed', 'kelly', 'value')
            initial_bankroll: Starting bankroll
            fixed_stake: Fixed stake amount (for fixed strategy)
            min_confidence: Minimum model confidence to bet
            min_edge: Minimum edge to bet (model_prob - implied_prob)
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest with {strategy} strategy")
        
        self.transactions = []
        bankroll = initial_bankroll
        
        # Get predictions
        predictions_proba = self.model.predict_proba(features_df)
        predictions = self.model.predict(features_df)
        
        for idx, row in test_data.iterrows():
            # Get model prediction and probabilities
            pred = predictions[idx]
            probs = predictions_proba[idx]
            
            # Get odds for predicted outcome
            odds_map = {
                0: row['B365A'],  # Away
                1: row['B365D'],  # Draw
                2: row['B365H']   # Home
            }
            odds = odds_map[pred]
            model_prob = probs[pred]
            
            # Calculate implied probability from odds
            implied_prob = 1 / odds
            edge = model_prob - implied_prob
            
            # Check if we should bet
            should_bet = (
                model_prob >= min_confidence and
                edge >= min_edge
            )
            
            if not should_bet:
                continue
            
            # Calculate stake based on strategy
            if strategy == 'fixed':
                stake = fixed_stake
            elif strategy == 'kelly':
                # Kelly Criterion: f = (bp - q) / b
                # f = (odds * prob - (1-prob)) / odds
                kelly_fraction = (odds * model_prob - (1 - model_prob)) / odds
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                stake = bankroll * kelly_fraction
            elif strategy == 'value':
                # Only bet if edge > threshold, use proportional stake
                stake = fixed_stake * (1 + edge)
            else:
                stake = fixed_stake
            
            # Ensure stake doesn't exceed bankroll
            stake = min(stake, bankroll * 0.1)  # Max 10% of bankroll per bet
            
            if stake < 1.0:  # Minimum bet
                continue
            
            # Get actual result
            result_map = {'H': 2, 'D': 1, 'A': 0}
            actual_result = result_map.get(row['FTR'], 1)
            
            # Create transaction
            transaction = BettingTransaction(
                match_id=idx,
                date=row['Date'],
                home_team=row['HomeTeam'],
                away_team=row['AwayTeam'],
                prediction=pred,
                actual_result=actual_result,
                odds=odds,
                stake=stake,
                model_probability=model_prob
            )
            
            self.transactions.append(transaction)
            
            # Update bankroll
            bankroll += transaction.profit
        
        logger.info(f"Backtest complete: {len(self.transactions)} bets placed")
        
        # Calculate metrics
        metrics = self.calculate_metrics(initial_bankroll)
        
        return metrics
    
    def calculate_metrics(self, initial_bankroll: float) -> Dict:
        """
        Calculate performance metrics from transactions
        
        Returns:
            Dictionary with metrics
        """
        if not self.transactions:
            return {
                'total_bets': 0,
                'error': 'No bets placed'
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([t.to_dict() for t in self.transactions])
        
        # Basic metrics
        total_bets = len(df)
        wins = df['won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Financial metrics
        total_staked = df['stake'].sum()
        total_returns = df['returns'].sum()
        net_profit = df['profit'].sum()
        roi = (net_profit / total_staked * 100) if total_staked > 0 else 0
        
        # Risk metrics
        equity_curve = initial_bankroll + df['profit'].cumsum()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio
        returns = df['profit'] / df['stake']
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Longest streaks
        df['streak'] = (df['won'] != df['won'].shift()).cumsum()
        win_streaks = df[df['won']].groupby('streak').size()
        loss_streaks = df[~df['won']].groupby('streak').size()
        
        longest_win_streak = win_streaks.max() if len(win_streaks) > 0 else 0
        longest_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        metrics = {
            'total_bets': total_bets,
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': round(win_rate * 100, 2),
            'total_staked': round(total_staked, 2),
            'total_returns': round(total_returns, 2),
            'net_profit': round(net_profit, 2),
            'roi': round(roi, 2),
            'final_bankroll': round(initial_bankroll + net_profit, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'longest_win_streak': int(longest_win_streak),
            'longest_loss_streak': int(longest_loss_streak),
            'avg_odds': round(df['odds'].mean(), 2),
            'avg_stake': round(df['stake'].mean(), 2),
            'avg_profit_per_bet': round(df['profit'].mean(), 2)
        }
        
        return metrics
    
    def get_transactions_df(self) -> pd.DataFrame:
        """Get transactions as DataFrame"""
        if not self.transactions:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.transactions])
