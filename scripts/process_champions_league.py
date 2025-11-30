"""
Process Champions League Data

Merge odds and scores files, convert to Football-Data.co.uk format.
"""

import pandas as pd
from pathlib import Path
from rich.console import Console

console = Console()


def process_champions_league():
    """Process and merge Champions League data"""
    
    console.print("[bold cyan]Processing Champions League Data[/bold cyan]\n")
    
    # Load all 4 files
    odds_cs = pd.read_csv('Database - Odds - Europe Champions League - CS.csv')
    scores_cs = pd.read_csv('Database - Scores - Europe Champions League - CS.csv')
    odds_ls = pd.read_csv('Database - Odds - Europe Champions League - LS.csv')
    scores_ls = pd.read_csv('Database - Scores - Europe Champions League - LS.csv')
    
    console.print(f"Loaded CS: {len(odds_cs)} matches")
    console.print(f"Loaded LS: {len(odds_ls)} matches")
    
    # Merge odds and scores for each dataset
    cs_merged = pd.merge(scores_cs, odds_cs, on='id', suffixes=('', '_odds'))
    ls_merged = pd.merge(scores_ls, odds_ls, on='id', suffixes=('', '_odds'))
    
    # Combine both
    all_cl = pd.concat([cs_merged, ls_merged], ignore_index=True)
    
    console.print(f"\nTotal Champions League matches: {len(all_cl)}")
    
    # Convert to Football-Data.co.uk format
    cl_formatted = pd.DataFrame({
        'Date': pd.to_datetime(all_cl['matchDate']).dt.strftime('%d/%m/%Y'),
        'HomeTeam': all_cl['homeTeam'],
        'AwayTeam': all_cl['awayTeam'],
        'FTHG': all_cl['FTHG'],
        'FTAG': all_cl['FTAG'],
        'FTR': all_cl['FTR'],
        'B365H': all_cl['H'],  # Home odds
        'B365D': all_cl['D'],  # Draw odds
        'B365A': all_cl['A'],  # Away odds
        'League': 'CL'  # Champions League code
    })
    
    # Remove any rows with missing data
    cl_formatted = cl_formatted.dropna()
    
    console.print(f"After cleaning: {len(cl_formatted)} matches with complete data")
    
    # Save by season
    output_dir = Path("data/historical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dates to extract seasons
    cl_formatted['DateParsed'] = pd.to_datetime(cl_formatted['Date'], format='%d/%m/%Y')
    
    # Group by season (Aug-May)
    seasons_saved = 0
    for year in range(2015, 2026):
        # Season runs from Aug year to May year+1
        season_start = pd.Timestamp(f'{year}-08-01')
        season_end = pd.Timestamp(f'{year+1}-07-31')
        
        season_data = cl_formatted[
            (cl_formatted['DateParsed'] >= season_start) & 
            (cl_formatted['DateParsed'] <= season_end)
        ].copy()
        
        if len(season_data) > 0:
            # Drop the DateParsed column
            season_data = season_data.drop('DateParsed', axis=1)
            
            # Save as CL_YYZZ.csv format
            season_code = f"{str(year)[2:]}{str(year+1)[2:]}"
            output_file = output_dir / f"CL_{season_code}.csv"
            season_data.to_csv(output_file, index=False)
            
            console.print(f"  Saved {output_file.name}: {len(season_data)} matches")
            seasons_saved += 1
    
    console.print(f"\n[bold green]Success![/bold green]")
    console.print(f"Saved {seasons_saved} Champions League season files")
    console.print(f"Total CL matches: {len(cl_formatted)}")
    
    # Calculate new total
    all_files = list(output_dir.glob("*.csv"))
    total_matches = 0
    cl_matches = 0
    
    for f in all_files:
        try:
            df = pd.read_csv(f)
            total_matches += len(df)
            if 'CL_' in f.name:
                cl_matches += len(df)
        except:
            pass
    
    console.print(f"\n[bold cyan]Complete Dataset:[/bold cyan]")
    console.print(f"  League matches: {total_matches - cl_matches:,}")
    console.print(f"  Champions League: {cl_matches:,}")
    console.print(f"  [bold]TOTAL: {total_matches:,} matches[/bold]")
    console.print(f"\n[dim]Ready for re-training![/dim]\n")


if __name__ == "__main__":
    process_champions_league()
