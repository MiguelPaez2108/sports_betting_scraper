"""
Download Historical Data

Script to download historical match data from Football-Data.co.uk
for model training and backtesting.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

from src.data_collection.infrastructure.scrapers.football_data_uk_scraper import (
    FootballDataUKScraper
)
from src.shared.logging.logger import get_logger

console = Console()
logger = get_logger(__name__)


async def download_historical_data(
    start_year: int = 2010,
    end_year: int = 2024,
):
    """
    Download historical data for all major leagues
    
    Args:
        start_year: Starting year for data download
        end_year: Ending year for data download
    """
    console.print(Panel.fit(
        "[bold cyan]Historical Data Download[/bold cyan]\n"
        f"[yellow]Downloading data from {start_year} to {end_year}[/yellow]",
        border_style="cyan"
    ))
    
    async with FootballDataUKScraper() as scraper:
        # Download all major leagues
        console.print("\n[bold]Downloading major European leagues...[/bold]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Downloading...",
                total=5  # 5 major leagues
            )
            
            results = {}
            
            # Premier League
            progress.update(task, description="[cyan]Premier League...")
            results["Premier League"] = await scraper.download_league_history(
                "E0", start_year, end_year
            )
            progress.advance(task)
            
            # La Liga
            progress.update(task, description="[cyan]La Liga...")
            results["La Liga"] = await scraper.download_league_history(
                "SP1", start_year, end_year
            )
            progress.advance(task)
            
            # Serie A
            progress.update(task, description="[cyan]Serie A...")
            results["Serie A"] = await scraper.download_league_history(
                "I1", start_year, end_year
            )
            progress.advance(task)
            
            # Bundesliga
            progress.update(task, description="[cyan]Bundesliga...")
            results["Bundesliga"] = await scraper.download_league_history(
                "D1", start_year, end_year
            )
            progress.advance(task)
            
            # Ligue 1
            progress.update(task, description="[cyan]Ligue 1...")
            results["Ligue 1"] = await scraper.download_league_history(
                "F1", start_year, end_year
            )
            progress.advance(task)
        
        # Display summary
        console.print("\n[bold green]✓ Download complete![/bold green]\n")
        
        table = Table(title="Downloaded Data Summary")
        table.add_column("League", style="cyan")
        table.add_column("Seasons", style="green")
        table.add_column("Matches", style="yellow")
        table.add_column("Date Range", style="magenta")
        
        total_matches = 0
        
        for league_name, dataframes in results.items():
            if dataframes:
                num_seasons = len(dataframes)
                num_matches = sum(len(df) for df in dataframes)
                total_matches += num_matches
                
                # Get date range
                all_dates = []
                for df in dataframes:
                    if 'Date' in df.columns:
                        all_dates.extend(df['Date'].tolist())
                
                if all_dates:
                    date_range = f"{min(all_dates)} to {max(all_dates)}"
                else:
                    date_range = "N/A"
                
                table.add_row(
                    league_name,
                    str(num_seasons),
                    str(num_matches),
                    date_range
                )
        
        console.print(table)
        
        # Stats
        stats = scraper.get_stats()
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Files Downloaded: {stats['files_downloaded']}")
        console.print(f"  Total Matches: {stats['matches_parsed']}")
        console.print(f"  Errors: {stats['errors']}")
        console.print(f"\n[dim]Data saved to: {scraper.download_dir}[/dim]")
        
        return results


async def download_specific_league(
    league_code: str,
    league_name: str,
    start_year: int = 2010,
    end_year: int = 2024,
):
    """
    Download data for a specific league
    
    Args:
        league_code: League code (e.g., "E0")
        league_name: League name for display
        start_year: Starting year
        end_year: Ending year
    """
    console.print(f"\n[bold cyan]Downloading {league_name}...[/bold cyan]")
    
    async with FootballDataUKScraper() as scraper:
        dataframes = await scraper.download_league_history(
            league_code,
            start_year,
            end_year
        )
        
        if dataframes:
            total_matches = sum(len(df) for df in dataframes)
            console.print(
                f"[green]✓[/green] Downloaded {len(dataframes)} seasons "
                f"with {total_matches} matches"
            )
            
            # Show sample data from first season
            if dataframes:
                console.print(f"\n[bold]Sample data (first 5 matches):[/bold]")
                console.print(dataframes[0].head())
        else:
            console.print(f"[red]✗[/red] No data downloaded")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download historical match data from Football-Data.co.uk"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Starting year (default: 2010)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Ending year (default: 2024)"
    )
    parser.add_argument(
        "--league",
        type=str,
        help="Specific league code (e.g., E0, SP1, I1)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all major leagues"
    )
    
    args = parser.parse_args()
    
    if args.league:
        # Download specific league
        await download_specific_league(
            args.league,
            args.league,
            args.start_year,
            args.end_year
        )
    else:
        # Download all major leagues
        await download_historical_data(
            args.start_year,
            args.end_year
        )


if __name__ == "__main__":
    asyncio.run(main())
