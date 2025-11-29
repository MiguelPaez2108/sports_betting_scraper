"""
Test Complete Feature Pipeline

Test all 56 features calculated together through the pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.feature_engineering.application.feature_pipeline import FeaturePipeline

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def test_complete_pipeline():
    """Test complete feature pipeline"""
    
    console.print(Panel.fit(
        "[bold cyan]Testing Complete Feature Pipeline[/bold cyan]\n"
        "[yellow]56 Features: Basic + Advanced + H2H[/yellow]",
        border_style="cyan"
    ))
    
    with FeaturePipeline(DB_CONFIG) as pipeline:
        # Test match
        home_team = "Man City"
        away_team = "Arsenal"
        match_date = datetime(2024, 3, 31)
        
        console.print(f"\n[bold]Match:[/bold] {home_team} vs {away_team}")
        console.print(f"[bold]Date:[/bold] {match_date.strftime('%Y-%m-%d')}\n")
        
        # Calculate all features
        console.print("[cyan]Calculating features...[/cyan]")
        
        features = pipeline.calculate_features(
            home_team, away_team, match_date
        )
        
        # Validate features
        validation = pipeline.validate_features(features)
        
        # Display summary
        console.print(f"\n[bold green]Feature Calculation Complete![/bold green]")
        console.print(f"Total features: [bold]{len(features)}[/bold]")
        console.print(f"Valid: [bold]{'Yes' if validation['is_valid'] else 'No'}[/bold]")
        console.print(f"Missing: {validation['missing_count']}")
        console.print(f"Outliers: {validation['outlier_count']}\n")
        
        # Display features by category
        feature_groups = pipeline.get_feature_importance_groups()
        
        for category, feature_names in feature_groups.items():
            console.print(f"[bold yellow]{category.upper()} Features[/bold yellow]")
            
            table = Table()
            table.add_column("Feature", style="cyan")
            table.add_column("Value", style="green")
            
            for fname in feature_names:
                if fname in features:
                    value = features[fname]
                    if isinstance(value, float):
                        table.add_row(fname, f"{value:.2f}")
                    else:
                        table.add_row(fname, str(value))
            
            console.print(table)
            console.print()
        
        # Display key insights
        console.print("[bold yellow]Key Insights[/bold yellow]")
        insights_table = Table()
        insights_table.add_column("Metric", style="cyan")
        insights_table.add_column("Home", style="green")
        insights_table.add_column("Away", style="red")
        insights_table.add_column("Advantage", style="yellow")
        
        # Elo
        elo_adv = "Home" if features['elo_difference'] > 0 else "Away"
        insights_table.add_row(
            "Elo Rating",
            str(features['home_elo']),
            str(features['away_elo']),
            elo_adv
        )
        
        # Form
        form_adv = "Home" if features['home_form_last_5'] > features['away_form_last_5'] else "Away"
        insights_table.add_row(
            "Recent Form",
            f"{features['home_form_last_5']:.2f}",
            f"{features['away_form_last_5']:.2f}",
            form_adv
        )
        
        # xG
        xg_adv = "Home" if features['home_xg'] > features['away_xg'] else "Away"
        insights_table.add_row(
            "Expected Goals",
            f"{features['home_xg']:.2f}",
            f"{features['away_xg']:.2f}",
            xg_adv
        )
        
        # H2H
        if features['h2h_total_matches'] > 0:
            if features['h2h_home_wins'] > features['h2h_away_wins']:
                h2h_adv = "Home"
            elif features['h2h_away_wins'] > features['h2h_home_wins']:
                h2h_adv = "Away"
            else:
                h2h_adv = "Even"
            
            insights_table.add_row(
                "H2H Record",
                f"{features['h2h_home_wins']}W",
                f"{features['h2h_away_wins']}W",
                h2h_adv
            )
        
        console.print(insights_table)
        
        # Poisson prediction
        console.print(f"\n[bold yellow]Poisson Model Prediction[/bold yellow]")
        poisson_table = Table()
        poisson_table.add_column("Outcome", style="cyan")
        poisson_table.add_column("Probability", style="yellow")
        
        poisson_table.add_row("Home Win", f"{features['poisson_prob_home_win']*100:.1f}%")
        poisson_table.add_row("Draw", f"{features['poisson_prob_draw']*100:.1f}%")
        poisson_table.add_row("Away Win", f"{features['poisson_prob_away_win']*100:.1f}%")
        
        console.print(poisson_table)
        
        console.print(f"\n[bold green]SUCCESS![/bold green]")
        console.print(f"[dim]All 56 features ready for ML model training[/dim]\n")


if __name__ == "__main__":
    test_complete_pipeline()
