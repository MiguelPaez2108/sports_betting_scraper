"""
Simple Historical Data Download Test

Standalone script to test downloading historical data from Football-Data.co.uk
No complex imports needed.
"""

import asyncio
from pathlib import Path
from io import StringIO
import sys

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


class SimpleHistoricalScraper:
    """Simple scraper for Football-Data.co.uk"""
    
    BASE_URL = "https://www.football-data.co.uk"
    
    LEAGUES = {
        "Premier League": "E0",
        "La Liga": "SP1",
        "Serie A": "I1",
        "Bundesliga": "D1",
        "Ligue 1": "F1",
    }
    
    def __init__(self, download_dir="data/historical"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.files_downloaded = 0
        self.matches_parsed = 0
    
    def _season_to_string(self, year):
        """Convert year to season string (e.g., 2023 -> '2324')"""
        next_year = (year + 1) % 100
        return f"{year % 100:02d}{next_year:02d}"
    
    def _build_csv_url(self, league_code, season):
        """Build URL for CSV file"""
        return f"{self.BASE_URL}/mmz4281/{season}/{league_code}.csv"
    
    async def download_csv(self, league_code, season):
        """Download CSV file for a league/season"""
        url = self._build_csv_url(league_code, season)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                console.print(f"[dim]Downloading {league_code} season {season}...[/dim]")
                
                response = await client.get(url)
                response.raise_for_status()
                
                # Parse CSV
                df = pd.read_csv(StringIO(response.text))
                
                # Save to disk
                filename = self.download_dir / f"{league_code}_{season}.csv"
                df.to_csv(filename, index=False)
                
                self.files_downloaded += 1
                self.matches_parsed += len(df)
                
                console.print(f"[green]OK[/green] Downloaded {len(df)} matches for {league_code} {season}")
                
                return df
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.print(f"[yellow]WARN[/yellow] CSV not found: {league_code} {season}")
            else:
                console.print(f"[red]ERROR[/red] HTTP error: {e}")
            return None
        
        except Exception as e:
            console.print(f"[red]ERROR[/red] Error: {e}")
            return None
    
    async def download_league_history(self, league_code, start_year, end_year):
        """Download all seasons for a league"""
        dataframes = []
        
        for year in range(start_year, end_year + 1):
            season = self._season_to_string(year)
            df = await self.download_csv(league_code, season)
            
            if df is not None:
                df['season'] = f"{year}-{year+1}"
                df['league_code'] = league_code
                dataframes.append(df)
            
            # Be polite - small delay
            await asyncio.sleep(0.5)
        
        return dataframes


async def test_download(start_year=2022, end_year=2024):
    """Test downloading historical data"""
    
    console.print(Panel.fit(
        "[bold cyan]Football-Data.co.uk Historical Data Test[/bold cyan]\n"
        f"[yellow]Downloading Premier League {start_year}-{end_year}[/yellow]",
        border_style="cyan"
    ))
    
    scraper = SimpleHistoricalScraper()
    
    console.print("\n[bold]Starting download...[/bold]\n")
    
    # Download Premier League
    dataframes = await scraper.download_league_history("E0", start_year, end_year)
    
    # Display summary
    console.print("\n" + "="*60)
    console.print("[bold green]Download Complete![/bold green]\n")
    
    if dataframes:
        table = Table(title="Downloaded Data")
        table.add_column("Season", style="cyan")
        table.add_column("Matches", style="green")
        table.add_column("Columns", style="yellow")
        
        for df in dataframes:
            season = df['season'].iloc[0] if 'season' in df.columns else "Unknown"
            table.add_row(
                season,
                str(len(df)),
                str(len(df.columns))
            )
        
        console.print(table)
        
        # Show sample data
        console.print("\n[bold]Sample Data (first match):[/bold]")
        first_match = dataframes[0].iloc[0]
        
        console.print(f"  Date: {first_match.get('Date', 'N/A')}")
        console.print(f"  Match: {first_match.get('HomeTeam', 'N/A')} vs {first_match.get('AwayTeam', 'N/A')}")
        console.print(f"  Score: {first_match.get('FTHG', 'N/A')}-{first_match.get('FTAG', 'N/A')}")
        
        if 'B365H' in first_match:
            console.print(f"  Odds (Bet365): {first_match.get('B365H', 'N/A')} / {first_match.get('B365D', 'N/A')} / {first_match.get('B365A', 'N/A')}")
        
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Files Downloaded: {scraper.files_downloaded}")
        console.print(f"  Total Matches: {scraper.matches_parsed}")
        console.print(f"  Saved to: {scraper.download_dir.absolute()}")
        
        console.print("\n[bold green]SUCCESS - Test successful![/bold green]")
        console.print("[dim]Historical data scraper is working correctly.[/dim]")
    else:
        console.print("[red]ERROR - No data downloaded[/red]")
    
    console.print("="*60 + "\n")


async def download_all_leagues(start_year=2020, end_year=2024):
    """Download all major leagues"""
    
    console.print(Panel.fit(
        "[bold cyan]Download All Major Leagues[/bold cyan]\n"
        f"[yellow]Years: {start_year}-{end_year}[/yellow]",
        border_style="cyan"
    ))
    
    scraper = SimpleHistoricalScraper()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Downloading...", total=5)
        
        results = {}
        
        for league_name, league_code in scraper.LEAGUES.items():
            progress.update(task, description=f"[cyan]{league_name}...")
            
            dataframes = await scraper.download_league_history(
                league_code, start_year, end_year
            )
            results[league_name] = dataframes
            
            progress.advance(task)
    
    # Summary
    console.print("\n[bold green]SUCCESS - All downloads complete![/bold green]\n")
    
    table = Table(title="Downloaded Data Summary")
    table.add_column("League", style="cyan")
    table.add_column("Seasons", style="green")
    table.add_column("Matches", style="yellow")
    
    for league_name, dataframes in results.items():
        if dataframes:
            num_seasons = len(dataframes)
            num_matches = sum(len(df) for df in dataframes)
            table.add_row(league_name, str(num_seasons), str(num_matches))
    
    console.print(table)
    console.print(f"\n[dim]Data saved to: {scraper.download_dir.absolute()}[/dim]\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical football data")
    parser.add_argument("--start-year", type=int, default=2022, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--all", action="store_true", help="Download all major leagues")
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(download_all_leagues(args.start_year, args.end_year))
    else:
        asyncio.run(test_download(args.start_year, args.end_year))
