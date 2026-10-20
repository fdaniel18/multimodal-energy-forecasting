"""
quality_audit.py — Audit calitate date GENOME + selectie ~100 cladiri

Rulare: python quality_audit.py
Output:
  - building_quality_report.csv  (toate cladirile cu metrici)
  - TARGET_BUILDINGS afisat in consola, gata de copiat in notebook
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Configurare
# ---------------------------------------------------------------------------
DATA_DIR        = "data_set_genome"
ELEC_PATH       = os.path.join(DATA_DIR, "electricity.csv")
META_PATH       = os.path.join(DATA_DIR, "metadata.csv")
REPORT_PATH     = "building_quality_report.csv"

TARGET_TOTAL    = 100
TARGET_EDUCATION = 40
TARGET_OFFICE    = 35
# restul ~25 se completeaza cu alte tipuri in ordinea scorului

MIN_FLAT_RUN    = 4    # ore consecutive identice → artefact
FLAT_THRESH     = 0.001  # toleranta pentru "identic" (evita floating-point noise)

# Filtre minime de calitate
MIN_COMPLETENESS    = 0.80
MAX_FLAT_RATIO      = 0.15
MAX_NEAR_ZERO_RATIO = 0.10
MIN_CV              = 0.10

# ---------------------------------------------------------------------------
# Incarcare date
# ---------------------------------------------------------------------------
print("Incarcare electricity.csv...")
df_elec = pd.read_csv(ELEC_PATH)
df_elec['timestamp'] = pd.to_datetime(df_elec['timestamp'])
df_elec.set_index('timestamp', inplace=True)
df_elec.sort_index(inplace=True)

building_cols = [c for c in df_elec.columns if c != 'timestamp']
total_hours   = len(df_elec)
print(f"  {total_hours} ore, {len(building_cols)} cladiri disponibile")

print("Incarcare metadata.csv...")
df_meta = pd.read_csv(META_PATH)
# Index building_id -> row pentru lookup rapid
meta_idx = df_meta.set_index('building_id') if 'building_id' in df_meta.columns else df_meta

# ---------------------------------------------------------------------------
# Calcul metrici per cladire
# ---------------------------------------------------------------------------
print("Calcul metrici de calitate...")

records = []
for i, col in enumerate(building_cols):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(building_cols)}...")

    series = df_elec[col].copy()

    # --- Metrica 1: Completeness ---
    non_null = series.notna()
    non_zero = series > 0
    completeness = float((non_null & non_zero).sum()) / total_hours

    # --- Metrica 2: Flat line ratio ---
    # Diferenta absoluta fata de ora precedenta < toleranta = "identic"
    diff_abs = series.diff().abs()
    is_flat  = (diff_abs < FLAT_THRESH) & series.notna()
    # Grupare in run-uri consecutive
    run_id   = (is_flat != is_flat.shift()).cumsum()
    run_len  = is_flat.groupby(run_id).transform('sum')
    flat_mask = is_flat & (run_len >= MIN_FLAT_RUN)
    flat_ratio = float(flat_mask.sum()) / total_hours

    # --- Metrica 3: Near-zero ratio ---
    med = series.median()
    if pd.isna(med) or med <= 0:
        near_zero_ratio = 1.0
    else:
        near_zero_ratio = float((series < 0.05 * med).sum()) / total_hours

    # --- Metrica 4: Coefficient of variation ---
    mean_val = series.mean()
    std_val  = series.std()
    if pd.isna(mean_val) or mean_val <= 0:
        cv = 0.0
    else:
        cv = float(std_val / mean_val)

    # --- Scor compozit ---
    cv_norm = min(cv / 0.3, 1.0)
    quality_score = (
        completeness
        * (1.0 - flat_ratio)
        * (1.0 - near_zero_ratio)
        * cv_norm
    )

    # --- Metadate ---
    b_type = "Unknown"
    site   = "Unknown"
    if col in meta_idx.index:
        row    = meta_idx.loc[col]
        b_type = str(row.get('primaryspaceusage', 'Unknown')) if hasattr(row, 'get') else str(row['primaryspaceusage']) if 'primaryspaceusage' in row else 'Unknown'
        site   = str(row.get('site_id', 'Unknown')) if hasattr(row, 'get') else 'Unknown'

    records.append({
        'building_id':      col,
        'type':             b_type,
        'site':             site,
        'completeness':     round(completeness, 4),
        'flat_ratio':       round(flat_ratio, 4),
        'near_zero_ratio':  round(near_zero_ratio, 4),
        'cv':               round(cv, 4),
        'quality_score':    round(quality_score, 4),
        'passes_filter':    (
            completeness    >= MIN_COMPLETENESS    and
            flat_ratio      <= MAX_FLAT_RATIO      and
            near_zero_ratio <= MAX_NEAR_ZERO_RATIO and
            cv              >= MIN_CV
        )
    })

df_report = pd.DataFrame(records).sort_values('quality_score', ascending=False)
df_report.to_csv(REPORT_PATH, index=False)
print(f"\nRaport salvat in: {REPORT_PATH}")

# ---------------------------------------------------------------------------
# Statistici eliminare
# ---------------------------------------------------------------------------
total        = len(df_report)
passed       = df_report['passes_filter'].sum()
failed_comp  = (df_report['completeness']    < MIN_COMPLETENESS).sum()
failed_flat  = (df_report['flat_ratio']      > MAX_FLAT_RATIO).sum()
failed_zero  = (df_report['near_zero_ratio'] > MAX_NEAR_ZERO_RATIO).sum()
failed_cv    = (df_report['cv']              < MIN_CV).sum()

print(f"\n{'='*55}")
print(f"SUMAR AUDIT")
print(f"{'='*55}")
print(f"Total cladiri analizate : {total}")
print(f"Trec filtrele           : {passed}  ({100*passed/total:.1f}%)")
print(f"Eliminate completeness  : {failed_comp}")
print(f"Eliminate flat_ratio    : {failed_flat}")
print(f"Eliminate near_zero     : {failed_zero}")
print(f"Eliminate CV mic        : {failed_cv}")

df_ok = df_report[df_report['passes_filter']].copy()
print(f"\nCladiri eligibile per tip:")
print(df_ok['type'].value_counts().to_string())

# ---------------------------------------------------------------------------
# Selectie finala ~100 cladiri
# ---------------------------------------------------------------------------
PRIORITY_TYPES = [
    ('Education',                    TARGET_EDUCATION),
    ('Office',                       TARGET_OFFICE),
    ('Lodging/residential',          8),
    ('Healthcare',                   6),
    ('Entertainment/publicassembly', 5),
    ('Public services',              4),
    ('Warehouse/storage',            2),
]

selected = []
remaining_slots = TARGET_TOTAL

for b_type, quota in PRIORITY_TYPES:
    if remaining_slots <= 0:
        break
    subset = df_ok[df_ok['type'] == b_type].head(quota)
    selected.append(subset)
    remaining_slots -= len(subset)

# Completeaza cu orice alte tipuri daca mai sunt locuri
if remaining_slots > 0:
    already = pd.concat(selected)['building_id'].tolist() if selected else []
    others  = df_ok[~df_ok['building_id'].isin(already)].head(remaining_slots)
    selected.append(others)

df_selected = pd.concat(selected).drop_duplicates(subset='building_id')
df_selected = df_selected.sort_values(['type', 'quality_score'], ascending=[True, False])

print(f"\n{'='*55}")
print(f"SELECTIE FINALA: {len(df_selected)} cladiri")
print(f"{'='*55}")
print(df_selected['type'].value_counts().to_string())
print(f"\nScor mediu selectate: {df_selected['quality_score'].mean():.4f}")
print(f"Scor minim selectate: {df_selected['quality_score'].min():.4f}")

# ---------------------------------------------------------------------------
# Output TARGET_BUILDINGS
# ---------------------------------------------------------------------------
building_list = df_selected['building_id'].tolist()

print(f"\n{'='*55}")
print("TARGET_BUILDINGS = [")
for i, b in enumerate(building_list):
    comma = "," if i < len(building_list) - 1 else ""
    print(f"    '{b}'{comma}")
print("]")
print(f"{'='*55}")
print(f"\nCopiaza lista de mai sus in celula 65cee0cef4a906ce din notebook.")
