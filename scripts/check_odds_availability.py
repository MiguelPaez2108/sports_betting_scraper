import pandas as pd
import os

# Verificar odds en diferentes años
files_to_check = [
    'data/historical/E0_1516.csv',  # 2015-2016
    'data/historical/E0_1920.csv',  # 2019-2020
    'data/historical/E0_2324.csv',  # 2023-2024
]

print("Verificando disponibilidad de odds históricos:\n")

for filepath in files_to_check:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        # Verificar si tiene columna de odds
        has_odds_col = 'B365H' in df.columns
        
        if has_odds_col:
            # Contar cuántos tienen odds
            total = len(df)
            with_odds = df['B365H'].notna().sum()
            pct = (with_odds / total * 100)
            
            print(f"{os.path.basename(filepath)}:")
            print(f"  Total partidos: {total}")
            print(f"  Con odds: {with_odds} ({pct:.1f}%)")
            print(f"  Sin odds: {total - with_odds}")
            
            # Mostrar ejemplo
            if with_odds > 0:
                sample = df[df['B365H'].notna()][['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']].head(2)
                print(f"  Ejemplo:\n{sample.to_string(index=False)}")
        else:
            print(f"{os.path.basename(filepath)}: NO tiene columna de odds")
        
        print()
    else:
        print(f"{filepath}: Archivo no encontrado\n")
