#!/usr/bin/env python3
"""Fix timestamps in metadata.csv: extract years from filenames (or title),
store as plain year integers (e.g. 1450, 1922) instead of Unix timestamps."""
import csv, re

CSV_PATH = 'image_space/public/data/metadata.csv'

with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

year_pat = re.compile(r'(1[4-9]\d{2}|20[0-2]\d)')  # 1400-2029
fixed = 0

for row in rows:
    # Always re-extract year from filename (most reliable source)
    match = year_pat.search(row['filename'])
    if match:
        row['timestamp'] = match.group(1)
        fixed += 1
    else:
        row['timestamp'] = ''  # no year found

has_ts = sum(1 for r in rows if r['timestamp'])
print(f'Extracted years: {fixed}')
print(f'Rows with timestamp: {has_ts}/{len(rows)}')
print(f'Still missing: {len(rows) - has_ts}')

# Year distribution
years = [int(r['timestamp']) for r in rows if r['timestamp']]
if years:
    print(f'Year range: {min(years)} – {max(years)}')

# Samples
print('\nSample (with year):')
for r in [r for r in rows if r['timestamp']][:5]:
    print(f'  {r["filename"][:55]:55s} → {r["timestamp"]}')
print('Sample (no year):')
for r in [r for r in rows if not r['timestamp']][:5]:
    print(f'  {r["filename"][:55]:55s} → (empty)')

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'\nWrote {len(rows)} rows to {CSV_PATH}')
