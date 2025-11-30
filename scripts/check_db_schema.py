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

def check_schema():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'historical_matches'
        """)
        
        columns = cur.fetchall()
        
        table = Table(title="historical_matches Schema")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="green")
        
        for col, dtype in columns:
            table.add_row(col, dtype)
            
        console.print(table)
        
        # Check count
        cur.execute("SELECT COUNT(*) FROM historical_matches")
        count = cur.fetchone()[0]
        console.print(f"\n[bold]Current row count: {count}[/bold]")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    check_schema()
