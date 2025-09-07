# compute_indices.py
"""
Main CLI script.

Usage:
    python compute_indices.py                 # uses default seed_samples.csv
    python compute_indices.py myfile.csv      # uses myfile.csv
"""

import sys
import os
import pandas as pd
from hmpi_utils import load_config, compute_indices_for_df, detect_metals_in_df

DEFAULT_CSV = "seed_samples.csv"
OUT_CSV_SUFFIX = "_with_indices.csv"

def main():
    # determine input csv path
    input_csv = DEFAULT_CSV
    if len(sys.argv) >= 2:
        input_csv = sys.argv[1]

    if not os.path.exists(input_csv):
        print(f"Error: input CSV '{input_csv}' not found in current folder ({os.getcwd()}).")
        sys.exit(1)

    # load config
    cfg = load_config("config.json")

    # read csv
    df = pd.read_csv(input_csv)

    # info: which metals are present
    metals_present, metals_with_bg = detect_metals_in_df(df, cfg)
    print("Metals present in CSV:", metals_present)
    print("Metals with background values (will be used for CF/PLI):", metals_with_bg)
    if metals_with_bg:
        print("(CF/Igeo will be computed for the metals listed above.)")
    else:
        print("Warning: No metals with background values found. PLI will be Unknown.")

    # compute
    df_out = compute_indices_for_df(df, cfg)

    # write output file
    base, ext = os.path.splitext(input_csv)
    out_file = f"{base}{OUT_CSV_SUFFIX}"
    df_out.to_csv(out_file, index=False)
    print(f"\nDone. Results written to: {out_file}")

    # print quick preview
    preview_cols = ["sample_id", "site_name", "date", "PLI", "PLI_category"]
    # add a few CF columns if present
    cf_candidates = [c for c in df_out.columns if c.startswith("CF_")]
    preview_cols += cf_candidates[:5]
    preview_cols = [c for c in preview_cols if c in df_out.columns]
    print("\n--- Sample of computed results ---")
    print(df_out[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

