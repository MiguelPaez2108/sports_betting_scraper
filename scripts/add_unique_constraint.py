import psycopg2
from rich.console import Console

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}

def add_constraint():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        console.print("Adding unique constraint on (match_date, home_team, away_team)...")
        
        # Check if constraint exists first (optional, but good practice)
        # But simpler to just try adding it, if it exists it will fail or we can use IF NOT EXISTS logic if supported for constraints (Postgres doesn't support IF NOT EXISTS for constraints directly in ALTER TABLE easily without a function, but we can catch the error)
        
        try:
            cur.execute("""
                ALTER TABLE historical_matches
                ADD CONSTRAINT unique_match_constraint UNIQUE (match_date, home_team, away_team);
            """)
            conn.commit()
            console.print("[green]Constraint added successfully![/green]")
        except psycopg2.errors.DuplicateTable:
            console.print("[yellow]Constraint already exists (or duplicate table error?)[/yellow]")
            conn.rollback()
        except Exception as e:
            if "already exists" in str(e):
                console.print("[yellow]Constraint already exists[/yellow]")
                conn.rollback()
            else:
                console.print(f"[red]Error adding constraint: {e}[/red]")
                conn.rollback()
        
        cur.close()
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Connection error: {e}[/red]")

if __name__ == "__main__":
    add_constraint()
