# txt_to_csv_fog.py
"""
Convert raw Daphnet FOG .txt files to .csv with proper column names.
NO preprocessing, NO filtering, NO normalization.

Usage:
    python txt_to_csv_fog.py --input ../data --output ../data_csv
"""

import argparse
import os
import pandas as pd

COLUMNS = [
    "Time",
    "Shank_H_Fwd", "Shank_Vert", "Shank_H_Lat",
    "Thigh_H_Fwd", "Thigh_Vert", "Thigh_H_Lat",
    "Trunk_H_Fwd", "Trunk_Vert", "Trunk_H_Lat",
    "Annot"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Folder containing raw .txt files")
    parser.add_argument("--output", type=str, required=True,
                        help="Folder to save converted .csv files")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    for fname in os.listdir(args.input):
        if not fname.endswith(".txt"):
            continue

        in_path = os.path.join(args.input, fname)
        out_path = os.path.join(args.output, fname.replace(".txt", ".csv"))

        # Read raw txt (space-separated)
        df = pd.read_csv(in_path, sep=r"\s+", header=None)
        df.columns = COLUMNS

        df.to_csv(out_path, index=False)
        print(f"Converted: {fname} -> {out_path}")

    print("All TXT files converted to CSV.")

if __name__ == "__main__":
    main()
