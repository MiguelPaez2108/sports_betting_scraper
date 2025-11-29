"""
Test ML Data Loader

Test data loading and feature calculation with sample matches.
Uses different teams/leagues (not just Man City vs Arsenal!)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.ml_models.infrastructure.data_loader import MLDataLoader

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def test_data_loader():
    """Test data loader with sample matches"""
    
    console.print(Panel.fit(
        "[bold cyan]Testing ML Data Loader[/bold cyan]\n"
        "[yellow]Loading matches and calculating features[/yellow]",
        border_style="cyan"
    ))
    
    with MLDataLoader(DB_CONFIG) as loader:
        
        # Test 1: Load sample matches
        console.print("\n[bold yellow]Test 1: Loading Historical Matches[/bold yellow]")
        
        matches_df = loader.load_historical_matches(
            start_date='2023-01-01',
            end_date='2024-12-31',
            leagues=['E0', 'SP1'],  # Premier League + La Liga
            limit=50  # Just 50 matches for testing
        )
        
        console.print(f"Loaded [bold]{len(matches_df)}[/bold] matches")
        
        # Show sample matches (variety of teams)
        sample_table = Table(title="Sample Matches")
        sample_table.add_column("Date", style="cyan")
        sample_table.add_column("League", style="yellow")
        sample_table.add_column("Match", style="green")
        sample_table.add_column("Result", style="magenta")
        
        for idx, match in matches_df.head(10).iterrows():
            sample_table.add_row(
                str(match['match_date']),
                match['league_code'],
                f"{match['home_team']} vs {match['away_team']}",
                f"{match['fthg']}-{match['ftag']} ({match['ftr']})"
            )
        
        console.print(sample_table)
        
        # Test 2: Calculate features for sample
        console.print("\n[bold yellow]Test 2: Calculating Features (Batch)[/bold yellow]")
        console.print("[dim]This will take a few minutes...[/dim]\n")
        
        X, y = loader.calculate_features_batch(
            matches_df.head(10),  # Just 10 matches for quick test
            batch_size=5,
            use_cache=False  # Don't cache test data
        )
        
        console.print(f"Features calculated: [bold]{len(X)}[/bold] samples")
        console.print(f"Feature count: [bold]{len(X.columns)}[/bold] features")
        console.print(f"Target distribution:")
        
        target_counts = y.value_counts()
        target_names = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
        
        dist_table = Table()
        dist_table.add_column("Outcome", style="cyan")
        dist_table.add_column("Count", style="green")
        dist_table.add_column("Percentage", style="yellow")
        
        for target_val, count in target_counts.items():
            pct = count / len(y) * 100
            dist_table.add_row(
                target_names[target_val],
                str(count),
                f"{pct:.1f}%"
            )
        
        console.print(dist_table)
        
        # Show sample features
        console.print("\n[bold yellow]Sample Features (First Match)[/bold yellow]")
        
        first_match_features = X.iloc[0]
        
        features_table = Table()
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Value", style="green")
        
        # Show key features
        key_features = [
            'home_elo', 'away_elo', 'elo_difference',
            'home_form_last_5', 'away_form_last_5',
            'home_xg', 'away_xg',
            'poisson_prob_home_win', 'poisson_prob_draw', 'poisson_prob_away_win',
            'h2h_home_wins', 'h2h_away_wins'
        ]
        
        for feature in key_features:
            if feature in first_match_features:
                value = first_match_features[feature]
                if isinstance(value, float):
                    features_table.add_row(feature, f"{value:.2f}")
                else:
                    features_table.add_row(feature, str(value))
        
        console.print(features_table)
        
        # Test 3: Data splitting
        console.print("\n[bold yellow]Test 3: Train/Val/Test Split[/bold yellow]")
        
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data_temporal(
            X, y, test_size=0.2, val_size=0.2
        )
        
        split_table = Table()
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Samples", style="green")
        split_table.add_column("Percentage", style="yellow")
        
        total = len(X)
        split_table.add_row("Train", str(len(X_train)), f"{len(X_train)/total*100:.1f}%")
        split_table.add_row("Validation", str(len(X_val)), f"{len(X_val)/total*100:.1f}%")
        split_table.add_row("Test", str(len(X_test)), f"{len(X_test)/total*100:.1f}%")
        split_table.add_row("Total", str(total), "100.0%")
        
        console.print(split_table)
        
        console.print(f"\n[bold green]SUCCESS![/bold green]")
        console.print(f"[dim]Data loader ready for full dataset processing[/dim]\n")


if __name__ == "__main__":
    test_data_loader()
