# hmpi_utils.py
"""
HMPI utility functions (reusable).

Provides:
- load_config(path)
- detect_metals(df, cfg)
- compute_indices_for_df(df, cfg)
"""

import json
import math
from typing import List, Dict, Tuple
import pandas as pd


def load_config(path: str = "config.json") -> Dict:
    """Load config JSON (throws FileNotFoundError if missing)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def detect_metals_in_df(df: pd.DataFrame, cfg: Dict) -> Tuple[List[str], List[str]]:
    """
    Return two lists:
    - metals_present: metals from config (configured + optional) that exist as columns in df
    - metals_with_bg: subset of metals_present that have numeric background values in cfg
    """
    configured = cfg.get("metal_columns", [])
    optional = cfg.get("optional_metal_columns", [])
    bg_values = cfg.get("background_values", {})

    metals_present = [m for m in configured if m in df.columns]
    metals_present += [m for m in optional if m in df.columns and m not in metals_present]

    metals_with_bg = [m for m in metals_present if bg_values.get(m) is not None]

    return metals_present, metals_with_bg


def compute_cf(concentration, background):
    """Contamination Factor CF = Ci / Bi. Returns None if invalid."""
    try:
        if background is None:
            return None
        if pd.isna(concentration):
            return None
        if float(background) == 0:
            return None
        return float(concentration) / float(background)
    except Exception:
        return None


def compute_igeo(concentration, background):
    """Geoaccumulation index: Igeo = log2( Ci / (1.5 * Bi) )."""
    try:
        if background is None:
            return None
        if pd.isna(concentration) or float(concentration) <= 0 or float(background) <= 0:
            return None
        return math.log2(float(concentration) / (1.5 * float(background)))
    except Exception:
        return None


def geometric_mean(nums):
    """Geometric mean of positive numbers in nums; ignores None/NaN/<=0."""
    vals = [float(x) for x in nums if x is not None and not pd.isna(x) and float(x) > 0]
    if not vals:
        return None
    prod = 1.0
    for v in vals:
        prod *= v
    return prod ** (1.0 / len(vals))


def pli_category_from_value(v, thresholds: Dict):
    """Return textual PLI category based on thresholds dict."""
    try:
        if v is None:
            return "Unknown"
        v = float(v)
        if v <= thresholds.get("unpolluted", 1.0):
            return "Unpolluted (PLI<=1)"
        elif v <= thresholds.get("moderate", 2.0):
            return "Moderately Polluted (1<PLI<=2)"
        elif v <= thresholds.get("high", 3.0):
            return "Highly Polluted (2<PLI<=3)"
        else:
            return "Very Highly Polluted (PLI>3)"
    except Exception:
        return "Unknown"


def compute_indices_for_df(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Main function: compute CF, Igeo for metals with background in cfg,
    compute PLI as geometric mean of available CFs, and PLI_category.
    Also compute HMPI (Heavy Metal Pollution Index) and HMPI_category
    using standard_values and ideal_values from cfg.

    Returns a new DataFrame (copy) with new columns added.
    """
    # copy so we don't mutate original accidentally
    df = df.copy()

    # detect metals
    metals_present, metals_with_bg = detect_metals_in_df(df, cfg)
    bg_values = cfg.get("background_values", {})
    thresholds = cfg.get("pli_thresholds", {"unpolluted": 1.0, "moderate": 2.0, "high": 3.0})

    # --- compute CF and Igeo for metals that have background values ---
    cf_cols = []
    for m in metals_with_bg:
        cf_col = f"CF_{m}"
        igeo_col = f"Igeo_{m}"
        df[cf_col] = df[m].apply(lambda x: compute_cf(x, bg_values.get(m)))
        df[igeo_col] = df[m].apply(lambda x: compute_igeo(x, bg_values.get(m)))
        cf_cols.append(cf_col)

    # --- compute PLI — geometric mean of available CF columns ---
    if cf_cols:
        def pli_row(row):
            vals = [row[c] for c in cf_cols if not pd.isna(row[c])]
            return geometric_mean(vals)
        df["PLI"] = df.apply(pli_row, axis=1)
        df["PLI_category"] = df["PLI"].apply(lambda v: pli_category_from_value(v, thresholds))
    else:
        df["PLI"] = None
        df["PLI_category"] = "Unknown"

    # --- HMPI calculation (Heavy Metal Pollution Index) ---
    # HMPI formula used:
    #   Wi = 1 / Si  (unit weight, Si = standard permissible value for metal i)
    #   Qi = ((Mi - Ii) / (Si - Ii)) * 100   (Mi = measured conc, Ii = ideal conc, often 0)
    #   HMPI = sum(Wi * Qi) / sum(Wi)
    # Config requirements:
    #   cfg["standard_values"] -> dict mapping metal column -> Si (permissible)
    #   cfg["ideal_values"] -> dict mapping metal column -> Ii (default 0 if missing)
    metal_cols = cfg.get("metal_columns", [])  # full list configured
    standard_values = cfg.get("standard_values", {})  # map metal->Si
    ideal_values = cfg.get("ideal_values", {})  # map metal->Ii (optional)

    hmpi_vals = []
    hmpi_categories = []

    for _, row in df.iterrows():
        num = 0.0
        den = 0.0
        valid = False

        for m in metal_cols:
            Mi = row.get(m)
            Si = standard_values.get(m)
            Ii = ideal_values.get(m, 0.0)

            # skip if Si or Mi missing or Mi NaN
            if Si is None or Mi is None or pd.isna(Mi):
                continue

            try:
                Si_f = float(Si)
                Mi_f = float(Mi)
                Ii_f = float(Ii)
            except Exception:
                continue

            # avoid invalid denominator
            if Si_f == Ii_f:
                continue

            # Wi = 1/Si
            try:
                Wi = 1.0 / Si_f
            except Exception:
                continue

            # Qi = ((Mi - Ii) / (Si - Ii)) * 100
            Qi = ((Mi_f - Ii_f) / (Si_f - Ii_f)) * 100.0

            num += Wi * Qi
            den += Wi
            valid = True

        if not valid or den == 0:
            hmpi = None
            cat = "Unknown"
        else:
            hmpi = num / den
            # threshold: HMPI < 100 => Safe; else Polluted (common interpretation)
            try:
                if float(hmpi) < 100.0:
                    cat = "Safe (HMPI < 100)"
                else:
                    cat = "Polluted (HMPI ≥ 100)"
            except Exception:
                cat = "Unknown"

        hmpi_vals.append(hmpi)
        hmpi_categories.append(cat)

    # attach HMPI columns (align length with df)
    df["HMPI"] = hmpi_vals
    df["HMPI_category"] = hmpi_categories

    # return dataframe with new columns
    return df

