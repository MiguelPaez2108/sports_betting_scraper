"""
Download All Historical Data from Football-Data.co.uk

Downloads all available seasons from 1990s to current for all major leagues.
"""

import requests
import pandas as pd
from pathlib import Path
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Football-Data.co.uk base URL
BASE_URL = "https://www.football-data.co.uk"

# Leagues to download
LEAGUES = {
    'E0': 'Premier League',
    'SP1': 'La Liga',
    'I1': 'Serie A',
    'D1': 'Bundesliga',
    'F1': 'Ligue 1'
}

# Seasons to download (from 93-94 to 24-25)
# Football-Data.co.uk has data from 1993 onwards for most leagues
SEASONS = []
for year in range(93, 100):  # 93-94 to 99-00
    SEASONS.append(f"{year}{year+1}")
for year in range(0, 25):  # 00-01 to 24-25
    SEASONS.append(f"{year:02d}{year+1:02d}")


def download_season_data(league: str, season: str, output_dir: Path) -> bool:
    """
    Download data for a specific league and season
    
    Args:
        league: League code (E0, SP1, etc.)
        season: Season code (9394, 0001, etc.)
        output_dir: Directory to save CSV
    
    Returns:
        True if successful, False otherwise
    """
    # Construct URL
    # Format: https://www.football-data.co.uk/mmz4281/9394/E0.csv
    url = f"{BASE_URL}/mmz4281/{season}/{league}.csv"
    
    # Output filename
    output_file = output_dir / f"{league}_{season}.csv"
    
    # Skip if already exists
    if output_file.exists():
        return True
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Save CSV
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Verify it's valid CSV
            try:
                df = pd.read_csv(output_file)
                if len(df) > 0:
                    return True
                else:
                    output_file.unlink()  # Delete empty file
                    return False
            except:
                output_file.unlink()  # Delete invalid file
                return False
        else:
            return False
            
    except Exception as e:
        return False


def main():
    """Download all historical data"""
    
    console.print("[bold cyan]Downloading Historical Football Data[/bold cyan]")
    console.print(f"[yellow]Leagues: {', '.join(LEAGUES.values())}[/yellow]")
    console.print(f"[yellow]Seasons: 1993-94 to 2024-25[/yellow]\n")
    
    # Create output directory
    output_dir = Path("data/historical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_attempts = len(LEAGUES) * len(SEASONS)
    successful = 0
    failed = 0
    skipped = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("Downloading...", total=total_attempts)
        
        for league_code, league_name in LEAGUES.items():
            for season in SEASONS:
                # Update progress description
                progress.update(task, description=f"Downloading {league_name} {season[:2]}-{season[2:]}")
                
                # Check if file exists
                output_file = output_dir / f"{league_code}_{season}.csv"
                if output_file.exists():
                    skipped += 1
                    progress.advance(task)
                    continue
                
                # Download
                success = download_season_data(league_code, season, output_dir)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                progress.advance(task)
                
                # Be nice to the server
                time.sleep(0.5)
    
    # Summary
    console.print(f"\n[bold green]Download Complete![/bold green]")
    console.print(f"Successful: {successful}")
    console.print(f"Skipped (already exist): {skipped}")
    console.print(f"Failed: {failed}")
    console.print(f"Total files: {successful + skipped}")
    
    # Count total matches
    total_matches = 0
    for csv_file in output_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            total_matches += len(df)
        except:
            pass
    
    console.print(f"\n[bold cyan]Total matches in database: {total_matches:,}[/bold cyan]")
    console.print(f"[dim]Ready for training![/dim]\n")


if __name__ == "__main__":
    main()
