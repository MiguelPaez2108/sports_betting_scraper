"""
Test Basic Features Calculator

Test the basic features calculator with real match data from PostgreSQL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.feature_engineering.domain.calculators.basic_features import BasicFeaturesCalculator

console = Console()

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def test_basic_features():
    """Test basic features calculation"""
    
    console.print(Panel.fit(
        "[bold cyan]Testing Basic Features Calculator[/bold cyan]\n"
        "[yellow]Using real historical data from PostgreSQL[/yellow]",
        border_style="cyan"
    ))
    
    with BasicFeaturesCalculator(DB_CONFIG) as calculator:
        # Test with a real match from our data
        # Let's use a recent Premier League match
        home_team = "Man City"
        away_team = "Arsenal"
        match_date = datetime(2024, 3, 31)  # Example date
        
        console.print(f"\n[bold]Calculating features for:[/bold]")
        console.print(f"  {home_team} vs {away_team}")
        console.print(f"  Date: {match_date.strftime('%Y-%m-%d')}\n")
        
        # Calculate all basic features
        features = calculator.calculate_all_basic_features(
            home_team, away_team, match_date
        )
        
        # Display features in organized tables
        
        # Form features
        console.print("[bold yellow]Form Features[/bold yellow]")
        form_table = Table()
        form_table.add_column("Feature", style="cyan")
        form_table.add_column("Home", style="green")
        form_table.add_column("Away", style="red")
        
        form_table.add_row(
            "Form (Last 5)",
            str(features['home_form_last_5']),
            str(features['away_form_last_5'])
        )
        form_table.add_row(
            "Form (Last 10)",
            str(features['home_form_last_10']),
            str(features['away_form_last_10'])
        )
        
        console.print(form_table)
        
        # Goals features
        console.print("\n[bold yellow]Goals Statistics[/bold yellow]")
        goals_table = Table()
        goals_table.add_column("Feature", style="cyan")
        goals_table.add_column("Home", style="green")
        goals_table.add_column("Away", style="red")
        
        goals_table.add_row(
            "Avg Goals Scored",
            str(features['home_avg_goals_scored']),
            str(features['away_avg_goals_scored'])
        )
        goals_table.add_row(
            "Avg Goals Conceded",
            str(features['home_avg_goals_conceded']),
            str(features['away_avg_goals_conceded'])
        )
        goals_table.add_row(
            "Goal Difference",
            str(features['home_goal_difference']),
            str(features['away_goal_difference'])
        )
        
        console.print(goals_table)
        
        # Win percentages
        console.print("\n[bold yellow]Win Percentages[/bold yellow]")
        win_table = Table()
        win_table.add_column("Feature", style="cyan")
        win_table.add_column("Home", style="green")
        win_table.add_column("Away", style="red")
        
        win_table.add_row(
            "Win %",
            f"{features['home_win_pct']*100:.0f}%",
            f"{features['away_win_pct']*100:.0f}%"
        )
        win_table.add_row(
            "Draw %",
            f"{features['home_draw_pct']*100:.0f}%",
            f"{features['away_draw_pct']*100:.0f}%"
        )
        win_table.add_row(
            "Clean Sheet %",
            f"{features['home_clean_sheet_pct']*100:.0f}%",
            f"{features['away_clean_sheet_pct']*100:.0f}%"
        )
        
        console.print(win_table)
        
        # Match statistics
        console.print("\n[bold yellow]Match Statistics[/bold yellow]")
        stats_table = Table()
        stats_table.add_column("Feature", style="cyan")
        stats_table.add_column("Home", style="green")
        stats_table.add_column("Away", style="red")
        
        stats_table.add_row(
            "Avg Shots",
            str(features['home_avg_shots']),
            str(features['away_avg_shots'])
        )
        stats_table.add_row(
            "Avg Shots on Target",
            str(features['home_avg_shots_on_target']),
            str(features['away_avg_shots_on_target'])
        )
        stats_table.add_row(
            "Avg Corners",
            str(features['home_avg_corners']),
            str(features['away_avg_corners'])
        )
        
        console.print(stats_table)
        
        # Summary
        console.print(f"\n[bold green]SUCCESS![/bold green]")
        console.print(f"Calculated [bold]{len(features)}[/bold] basic features")
        console.print(f"\n[dim]Features can be used for ML model training[/dim]\n")


if __name__ == "__main__":
    test_basic_features()
