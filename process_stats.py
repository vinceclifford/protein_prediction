"""
Step 1 — Protein Dataset Statistics Extraction
=================================================
Processes DeepLoc and Meltome CSVs to extract sequence properties
and label information into a unified stats.csv.

Usage:
    python process_stats.py \
        --data_dir ./data \
        --output_csv ./protein_stats.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import hashlib

# ── Amino Acid Group Mapping ──────────────────────────────────────────────────
AA_GROUPS = {
    'hydrophobic': 'AILMFPWV',
    'polar': 'NQCSTGY',
    'charged_pos': 'KR',
    'charged_neg': 'DE',
    'special': 'GPH'
}

def compute_seq_stats(seq):
    if not isinstance(seq, str): return {}
    L = len(seq)
    stats = {"seq_length": L}
    for group, residues in AA_GROUPS.items():
        count = sum(seq.count(r) for r in residues)
        stats[f"pct_{group}"] = round(100 * count / L, 2) if L > 0 else 0
    return stats

def process_deeploc(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    label_cols = ['Membrane', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 
                  'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 
                  'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome']
    
    results = []
    for _, row in df.iterrows():
        stats = compute_seq_stats(row['Sequence'])
        stats.update({
            "subject_id": row['ACC'],
            "dataset": "DeepLoc",
            "task": "classification",
            "target": None,
            "num_labels": int(row[label_cols].sum()),
            "primary_label": row[label_cols].idxmax() if row[label_cols].sum() > 0 else "Unknown"
        })
        results.append(stats)
    return results

def process_meltome(file_path, name):
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    results = []
    for i, row in df.iterrows():
        stats = compute_seq_stats(row['sequence'])
        # Generate a stable ID from sequence hash
        subj_id = f"{name}_{hashlib.md5(row['sequence'].encode()).hexdigest()[:8]}"
        stats.update({
            "subject_id": subj_id,
            "dataset": name,
            "task": "regression",
            "target": row['target'],
            "num_labels": 1,
            "primary_label": "Thermostability"
        })
        results.append(stats)
    return results

def main():
    parser = argparse.ArgumentParser(description="Protein stats extraction")
    parser.add_argument("--data_dir", default="./data", help="Root data directory")
    parser.add_argument("--output_csv", default="protein_stats.csv", help="Output path")
    args = parser.parse_args()

    all_results = []
    
    print("Processing DeepLoc ...")
    deeploc_path = os.path.join(args.data_dir, "DeepLoc/Swissprot_Train_Validation_dataset.csv")
    res = process_deeploc(deeploc_path)
    if res: all_results.extend(res)

    print("Processing Meltome ...")
    meltome_files = [
        ("Meltome/human.csv", "Meltome_Human"),
        ("Meltome/human_cell.csv", "Meltome_HumanCell"),
        ("Meltome/mixed_split.csv", "Meltome_Mixed")
    ]
    for rel_path, name in meltome_files:
        res = process_meltome(os.path.join(args.data_dir, rel_path), name)
        if res: all_results.extend(res)

    if not all_results:
        print("No data found!")
        return

    df_stats = pd.DataFrame(all_results)
    df_stats.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved {len(df_stats)} sequences → {args.output_csv}")

if __name__ == "__main__":
    main()
