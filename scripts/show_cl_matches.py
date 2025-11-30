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

# Get a sample Champions League match
query = """
    SELECT 
        match_date,
        home_team,
        away_team,
        fthg,
        ftag,
        ftr,
        b365h,
        b365d,
        b365a
    FROM historical_matches
    WHERE league_code = 'CL'
    ORDER BY match_date DESC
    LIMIT 5
"""

df = pd.read_sql(query, conn)
print("Últimos 5 partidos de Champions League en la DB:\n")
print(df.to_string(index=False))

# Count by season
query2 = """
    SELECT 
        season,
        COUNT(*) as matches
    FROM historical_matches
    WHERE league_code = 'CL'
    GROUP BY season
    ORDER BY season DESC
"""

df2 = pd.read_sql(query2, conn)
print("\n\nPartidos de Champions League por temporada:\n")
print(df2.to_string(index=False))

conn.close()
