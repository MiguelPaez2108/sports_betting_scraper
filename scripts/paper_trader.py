"""
Paper Trading Simulator - Test strategy with live data without real money.

Usage:
    python scripts/paper_trader.py --model models/ensemble/ensemble.pkl
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.calibrator_manager import CalibratorManager
from src.backtesting.backtester import EnhancedBacktester, BettingFilters
from src.backtesting.stake_strategies import FractionalKelly
from src.execution.betfair_connector import BetfairConnector


class PaperTrader:
    """
    Paper trading simulator for live testing.
    """
    
    def __init__(
        self,
        model_path: Path,
        calibrator_path: Path,
        initial_bankroll: float = 1000.0
    ):
        """
        Args:
            model_path: Path to trained model
            calibrator_path: Path to calibrator
            initial_bankroll: Starting capital
        """
        self.model = self._load_model(model_path)
        self.calibrator = CalibratorManager.load(calibrator_path)
        
        self.backtester = EnhancedBacktester(
            initial_bankroll=initial_bankroll,
            stake_strategy=FractionalKelly(fraction=0.25),
            filters=BettingFilters(
                min_confidence=0.48,
                min_edge=0.02,
                min_odds=1.30,
                max_odds=5.00
            ),
            slippage_pct=0.01
        )
        
        self.connector = BetfairConnector(simulation_mode=True)
        self.trades_log = []
    
    def _load_model(self, path: Path):
        """Load model from disk."""
        import joblib
        return joblib.load(path)
    
    def run_daily(self, matches_df: pd.DataFrame) -> None:
        """
        Run paper trading for today's matches.
        
        Args:
            matches_df: Today's matches with features
        """
        print("=" * 60)
        print(f"PAPER TRADING - {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        if len(matches_df) == 0:
            print("\n⚠️  No matches today")
            return
        
        print(f"\nAnalyzing {len(matches_df)} matches...")
        
        # Generate predictions
        X = matches_df.drop(columns=['match_id', 'date', 'home_team', 'away_team'], errors='ignore')
        raw_probs = self.model.predict_proba(X)
        
        # Calibrate probabilities
        calibrated_probs = []
        for i, row in matches_df.iterrows():
            league = row.get('league', 'unknown')
            probs = self.calibrator.predict_proba(
                raw_probs[i:i+1],
                league=league
            )[0]
            calibrated_probs.append(probs)
        
        calibrated_probs = np.array(calibrated_probs)
        
        # Add predictions to dataframe
        matches_df['pred_home'] = calibrated_probs[:, 2]
        matches_df['pred_draw'] = calibrated_probs[:, 1]
        matches_df['pred_away'] = calibrated_probs[:, 0]
        
        # Simulate betting decisions
        bets_placed = 0
        
        for _, match in matches_df.iterrows():
            # Get odds (would come from live API in production)
            odds_home = match.get('odds_home', 0)
            odds_draw = match.get('odds_draw', 0)
            odds_away = match.get('odds_away', 0)
            
            if odds_home == 0:
                continue
            
            # Calculate implied probabilities
            total_imp = (1/odds_home + 1/odds_draw + 1/odds_away)
            imp_home = (1/odds_home) / total_imp
            imp_draw = (1/odds_draw) / total_imp
            imp_away = (1/odds_away) / total_imp
            
            # Find best bet
            edges = {
                'home': match['pred_home'] - imp_home,
                'draw': match['pred_draw'] - imp_draw,
                'away': match['pred_away'] - imp_away
            }
            
            best_selection = max(edges, key=edges.get)
            best_edge = edges[best_selection]
            
            # Check if bet passes filters
            if best_edge > 0.02 and match[f'pred_{best_selection}'] > 0.48:
                odds_map = {'home': odds_home, 'draw': odds_draw, 'away': odds_away}
                odds = odds_map[best_selection]
                
                # Calculate stake
                stake = self.backtester.stake_strategy.compute_stake(
                    bankroll=self.backtester.bankroll,
                    edge=best_edge,
                    odds=odds,
                    prob=match[f'pred_{best_selection}']
                )
                
                if stake > 0:
                    # Place simulated bet
                    print(f"\n📍 BET SIGNAL:")
                    print(f"  Match: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                    print(f"  Selection: {best_selection.upper()}")
                    print(f"  Odds: {odds:.2f}")
                    print(f"  Confidence: {match[f'pred_{best_selection}']:.1%}")
                    print(f"  Edge: {best_edge:.2%}")
                    print(f"  Stake: ${stake:.2f}")
                    
                    bets_placed += 1
        
        print(f"\n✓ Paper trading complete: {bets_placed} bets placed")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Paper trading simulator')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--calibrator', type=str, default='models/calibrators', help='Path to calibrator')
    parser.add_argument('--matches', type=str, required=True, help='Path to today\'s matches')
    args = parser.parse_args()
    
    trader = PaperTrader(
        model_path=Path(args.model),
        calibrator_path=Path(args.calibrator)
    )
    
    # Load today's matches
    matches_df = pd.read_parquet(args.matches)
    
    # Run paper trading
    trader.run_daily(matches_df)


if __name__ == '__main__':
    main()
