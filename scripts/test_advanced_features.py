"""
Test Advanced Features Calculator

Test Elo ratings, xG, Poisson, momentum, and fatigue with real data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.feature_engineering.domain.calculators.advanced_features import AdvancedFeaturesCalculator

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def test_advanced_features():
    """Test advanced features calculation"""
    
    console.print(Panel.fit(
        "[bold cyan]Testing Advanced Features Calculator[/bold cyan]\n"
        "[yellow]Elo, xG, Poisson, Momentum, Fatigue[/yellow]",
        border_style="cyan"
    ))
    
    with AdvancedFeaturesCalculator(DB_CONFIG) as calculator:
        # Test match
        home_team = "Man City"
        away_team = "Arsenal"
        match_date = datetime(2024, 3, 31)
        
        console.print(f"\n[bold]Match:[/bold] {home_team} vs {away_team}")
        console.print(f"[bold]Date:[/bold] {match_date.strftime('%Y-%m-%d')}\n")
        
        # Calculate all advanced features
        features = calculator.calculate_all_advanced_features(
            home_team, away_team, match_date
        )
        
        # Display Elo ratings
        console.print("[bold yellow]Elo Ratings[/bold yellow]")
        elo_table = Table()
        elo_table.add_column("Metric", style="cyan")
        elo_table.add_column("Home", style="green")
        elo_table.add_column("Away", style="red")
        
        elo_table.add_row(
            "Elo Rating",
            str(features['home_elo']),
            str(features['away_elo'])
        )
        elo_table.add_row(
            "Elo Difference",
            str(features['elo_difference']),
            "-"
        )
        
        console.print(elo_table)
        
        # Display xG
        console.print("\n[bold yellow]Expected Goals (xG)[/bold yellow]")
        xg_table = Table()
        xg_table.add_column("Metric", style="cyan")
        xg_table.add_column("Home", style="green")
        xg_table.add_column("Away", style="red")
        
        xg_table.add_row(
            "xG",
            str(features['home_xg']),
            str(features['away_xg'])
        )
        xg_table.add_row(
            "xG Difference",
            str(features['xg_difference']),
            "-"
        )
        
        console.print(xg_table)
        
        # Display Poisson
        console.print("\n[bold yellow]Poisson Probabilities[/bold yellow]")
        poisson_table = Table()
        poisson_table.add_column("Outcome", style="cyan")
        poisson_table.add_column("Probability", style="yellow")
        
        poisson_table.add_row(
            "Home Win",
            f"{features['poisson_prob_home_win']*100:.1f}%"
        )
        poisson_table.add_row(
            "Draw",
            f"{features['poisson_prob_draw']*100:.1f}%"
        )
        poisson_table.add_row(
            "Away Win",
            f"{features['poisson_prob_away_win']*100:.1f}%"
        )
        poisson_table.add_row(
            "Lambda Home",
            str(features['poisson_lambda_home'])
        )
        poisson_table.add_row(
            "Lambda Away",
            str(features['poisson_lambda_away'])
        )
        
        console.print(poisson_table)
        
        # Display Momentum
        console.print("\n[bold yellow]Momentum & Streaks[/bold yellow]")
        momentum_table = Table()
        momentum_table.add_column("Metric", style="cyan")
        momentum_table.add_column("Home", style="green")
        momentum_table.add_column("Away", style="red")
        
        momentum_table.add_row(
            "Current Streak",
            str(features['home_streak']),
            str(features['away_streak'])
        )
        momentum_table.add_row(
            "Momentum Score",
            str(features['home_momentum']),
            str(features['away_momentum'])
        )
        momentum_table.add_row(
            "Points Trend",
            str(features['home_points_trend']),
            str(features['away_points_trend'])
        )
        
        console.print(momentum_table)
        
        # Display Fatigue
        console.print("\n[bold yellow]Fatigue Indicators[/bold yellow]")
        fatigue_table = Table()
        fatigue_table.add_column("Metric", style="cyan")
        fatigue_table.add_column("Home", style="green")
        fatigue_table.add_column("Away", style="red")
        
        fatigue_table.add_row(
            "Days Rest",
            str(features['home_days_rest']),
            str(features['away_days_rest'])
        )
        fatigue_table.add_row(
            "Matches (14d)",
            str(features['home_matches_14d']),
            str(features['away_matches_14d'])
        )
        fatigue_table.add_row(
            "Fatigue Index",
            str(features['home_fatigue']),
            str(features['away_fatigue'])
        )
        
        console.print(fatigue_table)
        
        # Summary
        console.print(f"\n[bold green]SUCCESS![/bold green]")
        console.print(f"Calculated [bold]{len(features)}[/bold] advanced features")
        console.print(f"\n[dim]Ready for ML model training[/dim]\n")


if __name__ == "__main__":
    test_advanced_features()
