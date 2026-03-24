#!/bin/bash
# Fix aime_past.parquet: regenerate with correct column mapping
set -e
cd /code/nanww-sandbox/code/PettingLLMs

echo "=== Fixing aime_past.parquet ==="
echo "Old file:"
python3 -c "
import pandas as pd
df = pd.read_parquet('data/math/train/aime_past.parquet')
non_empty = (df['question'].str.len() > 0).sum()
print(f'  {len(df)} rows, {non_empty} non-empty questions')
"

echo ""
echo "Regenerating..."
python3 scripts/dataprocess/load_math.py

echo ""
echo "New file:"
python3 -c "
import pandas as pd
df = pd.read_parquet('data/math/train/aime_past.parquet')
non_empty = (df['question'].str.len() > 0).sum()
print(f'  {len(df)} rows, {non_empty} non-empty questions')
if non_empty > 0:
    print(f'  Sample: {df.iloc[0][\"question\"][:80]}...')
    print('  SUCCESS')
else:
    print('  STILL BROKEN')
"
