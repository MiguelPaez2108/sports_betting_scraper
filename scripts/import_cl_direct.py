"""
Import Champions League Data Directly

Reads the 4 original Database files, merges them, and imports to PostgreSQL.
"""

import pandas as pd
import psycopg2
from rich.console import Console
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

def import_cl_direct():
    """Import CL data directly from source files"""
    
    console.print("[bold cyan]Importing Champions League Data Directly[/bold cyan]\n")
    
    try:
        # Load files
        console.print("Loading CSV files...")
        base_path = "data/historical/"
        odds_cs = pd.read_csv(f'{base_path}Database - Odds - Europe Champions League - CS.csv')
        scores_cs = pd.read_csv(f'{base_path}Database - Scores - Europe Champions League - CS.csv')
        odds_ls = pd.read_csv(f'{base_path}Database - Odds - Europe Champions League - LS.csv')
        scores_ls = pd.read_csv(f'{base_path}Database - Scores - Europe Champions League - LS.csv')
        
        # Merge
        console.print("Merging Odds and Scores...")
        cs_merged = pd.merge(scores_cs, odds_cs, on='id', suffixes=('', '_odds'))
        ls_merged = pd.merge(scores_ls, odds_ls, on='id', suffixes=('', '_odds'))
        all_cl = pd.concat([cs_merged, ls_merged], ignore_index=True)
        
        console.print(f"Total matches to import: {len(all_cl)}")
        
        # Connect to DB
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        imported = 0
        total = len(all_cl)
        
        console.print(f"Starting import of {total} matches...")
        
        for idx, (_, row) in enumerate(all_cl.iterrows(), 1):
            if idx % 100 == 0:
                console.print(f"Processing match {idx}/{total}")
                
            try:
                # Parse date (Format: 26-11-25 21:00)
                # Assuming DD-MM-YY
                match_date = pd.to_datetime(row['matchDate'], dayfirst=True).date()
                
                # Extract season (e.g. "2024/2025" -> "2425")
                season_str = str(row['Season'])
                if '/' in season_str:
                    p = season_str.split('/')
                    season_code = f"{p[0][-2:]}{p[1][-2:]}"
                else:
                    season_code = 'UNKNOWN'

                # Map values
                values = {
                    'league_code': 'CL',
                    'season': season_code,
                    'match_date': match_date,
                    'home_team': row['homeTeam'],
                    'away_team': row['awayTeam'],
                    'fthg': clean_value(row.get('FTHG')),
                    'ftag': clean_value(row.get('FTAG')),
                    'ftr': row.get('FTR'),
                    # CL files don't have Half Time stats usually, or named differently
                    'hthg': clean_value(row.get('1HHG')),
                    'htag': clean_value(row.get('1HAG')),
                    'htr': row.get('1HR'),
                    'b365h': clean_value(row.get('H')),
                    'b365d': clean_value(row.get('D')),
                    'b365a': clean_value(row.get('A')),
                    # Stats might not be present or named differently
                    'hs': None, 'as_': None, 'hst': None, 'ast': None,
                    'hc': None, 'ac': None, 'hf': None, 'af': None,
                    'hy': None, 'ay': None, 'hr': None, 'ar': None
                }
                
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
                
                cur.execute(query, values)
                imported += 1
                
            except Exception as e:
                console.print(f"[red]Error row: {e}[/red]")
                conn.rollback()
                break # Stop on first error to debug
        
        conn.commit()
            
        console.print(f"\n[bold green]Successfully imported {imported} Champions League matches![/bold green]")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    import_cl_direct()
