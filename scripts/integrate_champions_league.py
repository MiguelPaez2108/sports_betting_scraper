"""
Integrate Champions League Data

Load Champions League CSV files and integrate them into the training dataset.
"""

import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def integrate_champions_league():
    """Integrate Champions League data into historical dataset"""
    
    console.print("[bold cyan]Integrating Champions League Data[/bold cyan]\n")
    
    # Directory for Champions League data
    cl_dir = Path("data/historical/champions_league")
    cl_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for Champions League files
    cl_files = list(cl_dir.glob("*.csv"))
    
    if not cl_files:
        console.print("[yellow]No Champions League files found in data/historical/champions_league/[/yellow]")
        console.print("[dim]Please add Champions League CSV files to this directory[/dim]")
        console.print("[dim]Expected format: CL_1516.csv, CL_1617.csv, etc.[/dim]\n")
        return
    
    console.print(f"[green]Found {len(cl_files)} Champions League files[/green]\n")
    
    # Process each file
    total_matches = 0
    files_with_odds = 0
    
    summary_table = Table(title="Champions League Data Summary")
    summary_table.add_column("File", style="cyan")
    summary_table.add_column("Matches", style="green")
    summary_table.add_column("Has Odds", style="yellow")
    
    for cl_file in sorted(cl_files):
        try:
            df = pd.read_csv(cl_file)
            
            # Check if has Bet365 odds
            has_odds = all(col in df.columns for col in ['B365H', 'B365D', 'B365A'])
            
            if has_odds:
                # Check how many matches have odds
                odds_count = df[['B365H', 'B365D', 'B365A']].notna().all(axis=1).sum()
                files_with_odds += 1
                odds_status = f"Yes ({odds_count}/{len(df)})"
            else:
                odds_status = "No"
            
            summary_table.add_row(
                cl_file.name,
                str(len(df)),
                odds_status
            )
            
            total_matches += len(df)
            
        except Exception as e:
            console.print(f"[red]Error reading {cl_file.name}: {e}[/red]")
    
    console.print(summary_table)
    
    console.print(f"\n[bold green]Total Champions League matches: {total_matches}[/bold green]")
    console.print(f"[bold green]Files with odds: {files_with_odds}/{len(cl_files)}[/bold green]")
    
    # Calculate new total
    historical_dir = Path("data/historical")
    league_files = [f for f in historical_dir.glob("*.csv") if not f.parent.name == "champions_league"]
    
    league_matches = 0
    for f in league_files:
        try:
            df = pd.read_csv(f)
            league_matches += len(df)
        except:
            pass
    
    console.print(f"\n[bold cyan]Dataset Summary:[/bold cyan]")
    console.print(f"  League matches: {league_matches:,}")
    console.print(f"  Champions League: {total_matches:,}")
    console.print(f"  [bold]Total: {league_matches + total_matches:,} matches[/bold]")
    console.print(f"\n[dim]Ready for training![/dim]\n")


if __name__ == "__main__":
    integrate_champions_league()
