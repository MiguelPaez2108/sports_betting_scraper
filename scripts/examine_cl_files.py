import pandas as pd

files = [
    'Database - Odds - Europe Champions League - CS.csv',
    'Database - Scores - Europe Champions League - CS.csv',
    'Database - Odds - Europe Champions League - LS.csv',
    'Database - Scores - Europe Champions League - LS.csv'
]

for filename in files:
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print('='*60)
    
    try:
        df = pd.read_csv(filename)
        print(f"Rows: {len(df)}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # Check for odds columns
        odds_cols = [col for col in df.columns if any(x in col.lower() for x in ['odd', 'b365', 'bet'])]
        if odds_cols:
            print(f"\nOdds columns found: {odds_cols}")
        
    except Exception as e:
        print(f"Error: {e}")
