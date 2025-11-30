"""
Import Historical Data to Database

Reads all CSV files from data/historical and imports them into PostgreSQL.
"""

import pandas as pd
import psycopg2
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import numpy as np

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}

def clean_value(val):
    """Clean value for database insertion"""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and np.isinf(val):
        return None
    return val

def import_data():
    """Import all historical CSVs to database"""
    
    console.print("[bold cyan]Importing Historical Data to Database[/bold cyan]\n")
    
    data_dir = Path("data/historical")
    csv_files = list(data_dir.glob("*.csv"))
    
    console.print(f"Found {len(csv_files)} CSV files")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        total_imported = 0
        total_skipped = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Importing files...", total=len(csv_files))
            
            for csv_file in csv_files:
                progress.update(task, description=f"Importing {csv_file.name}")
                
                # Extract league and season from filename (e.g., E0_2324.csv)
                parts = csv_file.stem.split('_')
                if len(parts) >= 2:
                    league_code = parts[0]
                    season_code = parts[1]
                else:
                    league_code = 'UNKNOWN'
                    season_code = 'UNKNOWN'
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Filter rows with missing essential data
                    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
                    
                    imported_count = 0
                    
                    for _, row in df.iterrows():
                        # Map CSV columns to DB columns
                        match_date = pd.to_datetime(row['Date'], dayfirst=True).date()
                        
                        # Prepare values
                        values = {
                            'league_code': league_code,
                            'season': season_code,
                            'match_date': match_date,
                            'home_team': row['HomeTeam'],
                            'away_team': row['AwayTeam'],
                            'fthg': clean_value(row.get('FTHG')),
                            'ftag': clean_value(row.get('FTAG')),
                            'ftr': row.get('FTR'),
                            'hthg': clean_value(row.get('HTHG')),
                            'htag': clean_value(row.get('HTAG')),
                            'htr': row.get('HTR'),
                            'b365h': clean_value(row.get('B365H')),
                            'b365d': clean_value(row.get('B365D')),
                            'b365a': clean_value(row.get('B365A')),
                            'hs': clean_value(row.get('HS')),
                            'as_': clean_value(row.get('AS')),
                            'hst': clean_value(row.get('HST')),
                            'ast': clean_value(row.get('AST')),
                            'hc': clean_value(row.get('HC')),
                            'ac': clean_value(row.get('AC')),
                            'hf': clean_value(row.get('HF')),
                            'af': clean_value(row.get('AF')),
                            'hy': clean_value(row.get('HY')),
                            'ay': clean_value(row.get('AY')),
                            'hr': clean_value(row.get('HR')),
                            'ar': clean_value(row.get('AR'))
                        }
                        
                        # Insert query
                        query = """
                            INSERT INTO historical_matches (
                                league_code, season, match_date, home_team, away_team,
                                fthg, ftag, ftr, hthg, htag, htr,
                                b365h, b365d, b365a,
                                hs, as_, hst, ast, hc, ac, hf, af, hy, ay, hr, ar
                            ) VALUES (
                                %(league_code)s, %(season)s, %(match_date)s, %(home_team)s, %(away_team)s,
                                %(fthg)s, %(ftag)s, %(ftr)s, %(hthg)s, %(htag)s, %(htr)s,
                                %(b365h)s, %(b365d)s, %(b365a)s,
                                %(hs)s, %(as_)s, %(hst)s, %(ast)s, %(hc)s, %(ac)s, %(hf)s, %(af)s, %(hy)s, %(ay)s, %(hr)s, %(ar)s
                            )
                            ON CONFLICT (match_date, home_team, away_team) DO UPDATE SET
                                b365h = EXCLUDED.b365h,
                                b365d = EXCLUDED.b365d,
                                b365a = EXCLUDED.b365a
                        """
                        
                        try:
                            cur.execute(query, values)
                            imported_count += 1
                        except Exception as e:
                            # console.print(f"[red]Error inserting match: {e}[/red]")
                            conn.rollback()
                            continue
                    
                    conn.commit()
                    total_imported += imported_count
                    
                except Exception as e:
                    console.print(f"[red]Error processing file {csv_file.name}: {e}[/red]")
                    conn.rollback()
                
                progress.advance(task)
        
        console.print(f"\n[bold green]Import Complete![/bold green]")
        console.print(f"Total matches processed: {total_imported}")
        
        # Verify final count
        cur.execute("SELECT COUNT(*) FROM historical_matches")
        final_count = cur.fetchone()[0]
        console.print(f"[bold cyan]Total matches in database: {final_count}[/bold cyan]")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Database connection error: {e}[/red]")

if __name__ == "__main__":
    import_data()
