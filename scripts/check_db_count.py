import psycopg2
import pandas as pd

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}

conn = psycopg2.connect(**DB_CONFIG)

# Count by league
df = pd.read_sql('SELECT league_code, COUNT(*) as count FROM historical_matches GROUP BY league_code ORDER BY league_code', conn)
print("Partidos por liga:")
print(df)
print(f"\nTotal en DB: {df['count'].sum()}")

# Check date range
df_dates = pd.read_sql('SELECT MIN(match_date) as min_date, MAX(match_date) as max_date FROM historical_matches', conn)
print(f"\nRango de fechas: {df_dates['min_date'][0]} a {df_dates['max_date'][0]}")

conn.close()
