"""
Import Historical Data to PostgreSQL

Reads CSV files from data/historical/ and imports them into PostgreSQL.
Creates tables and indexes for efficient querying.
"""

import glob
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def create_schema():
    """Create database schema for historical matches"""
    console.print("\n[bold cyan]Creating database schema...[/bold cyan]")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Create historical_matches table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_matches (
            id SERIAL PRIMARY KEY,
            league_code VARCHAR(10) NOT NULL,
            season VARCHAR(20) NOT NULL,
            match_date DATE NOT NULL,
            home_team VARCHAR(100) NOT NULL,
            away_team VARCHAR(100) NOT NULL,
            
            -- Full-time result
            fthg INTEGER,  -- Full-time home goals
            ftag INTEGER,  -- Full-time away goals
            ftr VARCHAR(1),  -- Full-time result (H/D/A)
            
            -- Half-time result
            hthg INTEGER,
            htag INTEGER,
            htr VARCHAR(1),
            
            -- Bet365 odds
            b365h DECIMAL(10,2),  -- Home win
            b365d DECIMAL(10,2),  -- Draw
            b365a DECIMAL(10,2),  -- Away win
            b365_over_2_5 DECIMAL(10,2),
            b365_under_2_5 DECIMAL(10,2),
            
            -- Pinnacle odds
            psh DECIMAL(10,2),
            psd DECIMAL(10,2),
            psa DECIMAL(10,2),
            
            -- Match statistics
            hs INTEGER,  -- Home shots
            as_ INTEGER,  -- Away shots
            hst INTEGER,  -- Home shots on target
            ast INTEGER,  -- Away shots on target
            hc INTEGER,  -- Home corners
            ac INTEGER,  -- Away corners
            hf INTEGER,  -- Home fouls
            af INTEGER,  -- Away fouls
            hy INTEGER,  -- Home yellow cards
            ay INTEGER,  -- Away yellow cards
            hr INTEGER,  -- Home red cards
            ar INTEGER,  -- Away red cards
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    console.print("[dim]Creating indexes...[/dim]")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_league_season ON historical_matches(league_code, season)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON historical_matches(match_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_home_team ON historical_matches(home_team)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_away_team ON historical_matches(away_team)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_teams ON historical_matches(home_team, away_team)")
    
    conn.commit()
    cur.close()
    conn.close()
    
    console.print("[green]OK[/green] Schema created successfully\n")


def parse_date(date_str):
    """Parse date from various formats"""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str)
    
    # Try different date formats
    for fmt in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d']:
        try:
            return pd.to_datetime(date_str, format=fmt).date()
        except:
            continue
    
    return None


def import_csv(filepath):
    """Import a single CSV file"""
    df = pd.read_csv(filepath)
    
    # Extract league code and season from filename
    filename = filepath.split('\\')[-1].replace('.csv', '')
    league_code, season_code = filename.split('_')
    
    # Convert season code to readable format (e.g., "2324" -> "2023-2024")
    year1 = 2000 + int(season_code[:2])
    year2 = year1 + 1
    season = f"{year1}-{year2}"
    
    records = []
    
    for _, row in df.iterrows():
        match_date = parse_date(row.get('Date'))
        if not match_date:
            continue
        
        record = (
            league_code,
            season,
            match_date,
            str(row.get('HomeTeam', '')),
            str(row.get('AwayTeam', '')),
            
            # Full-time
            int(row['FTHG']) if pd.notna(row.get('FTHG')) else None,
            int(row['FTAG']) if pd.notna(row.get('FTAG')) else None,
            str(row['FTR']) if pd.notna(row.get('FTR')) else None,
            
            # Half-time
            int(row['HTHG']) if pd.notna(row.get('HTHG')) else None,
            int(row['HTAG']) if pd.notna(row.get('HTAG')) else None,
            str(row['HTR']) if pd.notna(row.get('HTR')) else None,
            
            # Bet365 odds
            float(row['B365H']) if pd.notna(row.get('B365H')) else None,
            float(row['B365D']) if pd.notna(row.get('B365D')) else None,
            float(row['B365A']) if pd.notna(row.get('B365A')) else None,
            float(row['B365>2.5']) if pd.notna(row.get('B365>2.5')) else None,
            float(row['B365<2.5']) if pd.notna(row.get('B365<2.5')) else None,
            
            # Pinnacle odds
            float(row['PSH']) if pd.notna(row.get('PSH')) else None,
            float(row['PSD']) if pd.notna(row.get('PSD')) else None,
            float(row['PSA']) if pd.notna(row.get('PSA')) else None,
            
            # Statistics
            int(row['HS']) if pd.notna(row.get('HS')) else None,
            int(row['AS']) if pd.notna(row.get('AS')) else None,
            int(row['HST']) if pd.notna(row.get('HST')) else None,
            int(row['AST']) if pd.notna(row.get('AST')) else None,
            int(row['HC']) if pd.notna(row.get('HC')) else None,
            int(row['AC']) if pd.notna(row.get('AC')) else None,
            int(row['HF']) if pd.notna(row.get('HF')) else None,
            int(row['AF']) if pd.notna(row.get('AF')) else None,
            int(row['HY']) if pd.notna(row.get('HY')) else None,
            int(row['AY']) if pd.notna(row.get('AY')) else None,
            int(row['HR']) if pd.notna(row.get('HR')) else None,
            int(row['AR']) if pd.notna(row.get('AR')) else None,
        )
        
        records.append(record)
    
    return records, league_code, season


def import_all_csvs():
    """Import all CSV files from data/historical/"""
    console.print("[bold cyan]Importing historical data...[/bold cyan]\n")
    
    # Get all CSV files
    csv_files = glob.glob('data/historical/*.csv')
    
    if not csv_files:
        console.print("[red]ERROR[/red] No CSV files found in data/historical/")
        return
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    total_matches = 0
    league_stats = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Importing...", total=len(csv_files))
        
        for csv_file in csv_files:
            records, league_code, season = import_csv(csv_file)
            
            if records:
                # Insert records
                execute_values(
                    cur,
                    """
                    INSERT INTO historical_matches (
                        league_code, season, match_date, home_team, away_team,
                        fthg, ftag, ftr, hthg, htag, htr,
                        b365h, b365d, b365a, b365_over_2_5, b365_under_2_5,
                        psh, psd, psa,
                        hs, as_, hst, ast, hc, ac, hf, af, hy, ay, hr, ar
                    ) VALUES %s
                    """,
                    records
                )
                
                total_matches += len(records)
                
                if league_code not in league_stats:
                    league_stats[league_code] = 0
                league_stats[league_code] += len(records)
            
            progress.advance(task)
    
    conn.commit()
    cur.close()
    conn.close()
    
    # Display summary
    console.print("\n[bold green]Import complete![/bold green]\n")
    
    table = Table(title="Imported Data Summary")
    table.add_column("League", style="cyan")
    table.add_column("Matches", style="green")
    
    league_names = {
        'E0': 'Premier League',
        'SP1': 'La Liga',
        'I1': 'Serie A',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1'
    }
    
    for code, count in sorted(league_stats.items()):
        table.add_row(league_names.get(code, code), str(count))
    
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_matches}[/bold]")
    
    console.print(table)
    console.print(f"\n[dim]Data imported to PostgreSQL (localhost:5432)[/dim]\n")


if __name__ == "__main__":
    try:
        create_schema()
        import_all_csvs()
        console.print("[bold green]SUCCESS - All data imported to PostgreSQL![/bold green]")
    except Exception as e:
        console.print(f"[red]ERROR[/red] {e}")
        import traceback
        traceback.print_exc()
