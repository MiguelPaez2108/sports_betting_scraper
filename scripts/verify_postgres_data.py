"""
Verify PostgreSQL Import

Quick script to verify data was imported successfully.
"""

import psycopg2
from rich.console import Console
from rich.table import Table

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}

try:
    console.print("\n[bold cyan]Connecting to PostgreSQL...[/bold cyan]")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Check total matches
    cur.execute("SELECT COUNT(*) FROM historical_matches")
    total = cur.fetchone()[0]
    
    console.print(f"[green]OK[/green] Connected successfully\n")
    console.print(f"[bold]Total matches in database: {total}[/bold]\n")
    
    # Check by league
    cur.execute("""
        SELECT league_code, COUNT(*) as count
        FROM historical_matches
        GROUP BY league_code
        ORDER BY league_code
    """)
    
    table = Table(title="Matches by League")
    table.add_column("League Code", style="cyan")
    table.add_column("League Name", style="yellow")
    table.add_column("Matches", style="green")
    
    league_names = {
        'D1': 'Bundesliga',
        'E0': 'Premier League',
        'F1': 'Ligue 1',
        'I1': 'Serie A',
        'SP1': 'La Liga'
    }
    
    for row in cur.fetchall():
        code, count = row
        table.add_row(code, league_names.get(code, code), str(count))
    
    console.print(table)
    
    # Show sample data
    console.print("\n[bold]Sample matches (first 5):[/bold]\n")
    cur.execute("""
        SELECT match_date, home_team, away_team, fthg, ftag, b365h, b365d, b365a
        FROM historical_matches
        ORDER BY match_date DESC
        LIMIT 5
    """)
    
    sample_table = Table()
    sample_table.add_column("Date", style="cyan")
    sample_table.add_column("Match", style="yellow")
    sample_table.add_column("Score", style="green")
    sample_table.add_column("Odds (H/D/A)", style="magenta")
    
    for row in cur.fetchall():
        date, home, away, fthg, ftag, h, d, a = row
        match = f"{home} vs {away}"
        score = f"{fthg}-{ftag}" if fthg is not None else "N/A"
        odds = f"{h}/{d}/{a}" if h else "N/A"
        sample_table.add_row(str(date), match, score, odds)
    
    console.print(sample_table)
    
    # Check date range
    console.print("\n[bold]Date range:[/bold]")
    cur.execute("SELECT MIN(match_date), MAX(match_date) FROM historical_matches")
    min_date, max_date = cur.fetchone()
    console.print(f"  From: {min_date}")
    console.print(f"  To: {max_date}")
    
    # Check seasons
    console.print("\n[bold]Seasons available:[/bold]")
    cur.execute("SELECT DISTINCT season FROM historical_matches ORDER BY season")
    seasons = [row[0] for row in cur.fetchall()]
    console.print(f"  {', '.join(seasons)}")
    
    cur.close()
    conn.close()
    
    console.print("\n[bold green]SUCCESS - Data verified in PostgreSQL![/bold green]\n")
    
except Exception as e:
    console.print(f"\n[red]ERROR[/red] {e}\n")
    import traceback
    traceback.print_exc()
