"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.3.1
"""

"""
Basic cleaning operation before splitting data into train and val sets
"""

import logging

import numpy as np
import pandas as pd

from kedro_carprice_prediction.fuzzy_matching import FuzzyMatching, findModel

logger = logging.getLogger(__name__)

COLUMN_RENAME_MAP = {
    "carID": "car_id",
    "Brand": "brand",
    "fuelType": "fuel_type",
    "previousOwners": "previous_owners",
    "engineSize": "engine_size",
    "paintQuality%": "paint_quality_pct",
    "hasDamage": "has_damage",
}

BRANDS_LIST = ["Volkswagen", "Toyota", "Audi", "Ford", "BMW", "Skoda", "Opel", "Mercedes", "Hyundai"]
BRANDS_ALIASES = {"vw": "Volkswagen", "v": "Volkswagen", "merc": "Mercedes", "bm": "BMW"}
TRANSMISSIONS_LIST = ["Manual", "Automatic", "Semi-Automatic", "Other", "Unkown"]
TRANSMISSIONS_ALIASES = {"mt": "Manual", "man": "Manual", "auto": "Automatic", "at": "Automatic", "dsg": "Automatic"}
FUELTYPE_LIST = ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]

MODEL_BY_BRAND = {
    "Audi": ["A", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "Q", "Q2", "Q3", "Q5", "Q7", "Q8", "RS", "RS3", "RS4", "RS5", "RS6", "S", "S1", "S3", "S4", "S5", "S6", "SQ2", "SQ5", "TT", "e-tron"],
    "BMW": ["1 Series", "2 Series", "3 Serie", "3 SERIES", "3 Series", "4 Series", "5 Serie", "5 Series", "7 Series", "8 Series", "M2", "M3", "M4", "M5", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "i3", "i4"],
    "Ford": ["B-MAX", "C-MAX", "Ecosport", "Edge", "Fiesta", "Focus", "Galaxy", "Ka", "Kuga", "Mondeo", "Mustang", "Puma", "S-MAX", "Tourneo", "Transit"],
    "Hyundai": ["Accent", "Getz", "I1", "I10", "I2", "I20", "I3", "I30", "I40", "I80", "I800", "IONIQ", "Kona", "Santa Fe", "Tucson"],
    "Mercedes": ["A-Class", "AMG GT", "B-Class", "C-Class", "CLA", "CLK", "CLS", "Citan", "E-Class", "GLA", "GLB", "GLC", "GLE", "GLK", "GLS", "M-Class", "S-Class", "GLC-Class", "SL", "SLC", "SLK", "Sprinter", "V-Class", "Vaneo", "Viano"],
    "Opel": ["Adam", "Astra", "Corsa", "Crossland", "Grandland", "Insignia", "Karl", "Meriva", "Mokka", "Vectra", "Zafira"],
    "Skoda": ["Citigo", "Enyaq", "Fabia", "Kamiq", "Karoq", "Kodiaq", "Octavia", "Rapid", "Roomster", "Scala", "Superb", "Yeti"],
    "Toyota": ["Auris", "Avensis", "Aygo", "C-HR", "Camry", "Corolla", "GT86", "RAV4", "Verso", "Yaris"],
    "Volkswagen": ["Arteon", "Beetle", "Bora", "Caddy", "CC", "Crafter", "Eos", "Fox", "Golf", "Golf Plus", "ID.3", "ID.4", "Jetta", "Lupo", "Passat", "Phaeton", "Polo", "Scirocco", "Sharan", "T-Cross", "T-Roc", "Tiguan", "Touareg", "Touran", "Transporter", "Up"],
}

NUMERIC_COLS = ["year", "mileage", "tax", "mpg", "engine_size",
                "previous_owners", "paint_quality_pct", "has_damage"]
NON_NEGATIVE_COLS = ["mileage", "tax", "mpg", "previous_owners", "engine_size"]

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw camelCase/symbol column names to snake_case, Hopsworks-safe
    names. Must run first -- every other step assumes the new names."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def normalize_na_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Replace literal 'NA'/'N/A'/'' string placeholders with real NaN."""
    return df.replace({"NA": None, "N/A": None, "na": None, "": None})


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to numeric dtype, categoricals to string."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    for col in ["brand", "model", "transmission", "fuel_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def fuzzy_match_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-match brand/transmission/fuel_type/model against canonical
    lists, then backfill brand from model where possible. No fitting --
    safe to run per-file, before train/test combining."""
    cleaner = FuzzyMatching(
        df=df, brand_list=BRANDS_LIST, model_by_brand=MODEL_BY_BRAND,
        brand_aliases=BRANDS_ALIASES, transmission_list=TRANSMISSIONS_LIST,
        transmission_aliases=TRANSMISSIONS_ALIASES, fueltype_list=FUELTYPE_LIST,
    )
    cleaner.clean_column("brand")
    cleaner.clean_column("transmission")
    # fuel_type bypasses clean_column() -- its hardcoded "fueltype" check
    # (no underscore) won't recognize our snake_case column name
    cleaner.df["fuel_type"] = cleaner.df["fuel_type"].apply(
        lambda x: cleaner.fuzzy_fix(raw_value=x, valid_values=FUELTYPE_LIST)
    )
    cleaner.clean_modelCol("model", "brand")
    return findModel(cleaner.df, MODEL_BY_BRAND)


def null_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Null out values that are PHYSICALLY impossible regardless of business
    context (negative counts/measurements, out-of-range percentages/flags,
    pre-automobile years). Business-rule bounds (e.g. max engine size) live
    later in standardize_types/remove_outliers, not here."""
    df = df.copy()
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    if "paint_quality_pct" in df.columns:
        df.loc[(df["paint_quality_pct"] < 0) | (df["paint_quality_pct"] > 100), "paint_quality_pct"] = np.nan
    if "has_damage" in df.columns:
        df.loc[~df["has_damage"].isin([0, 1]), "has_damage"] = np.nan
    if "year" in df.columns:
        df.loc[df["year"] < 1900, "year"] = np.nan
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, then duplicate car_ids (keep first)."""
    before = len(df)
    df = df.drop_duplicates()
    if "car_id" in df.columns:
        df = df.drop_duplicates(subset=["car_id"], keep="first")
    logger.info(f"Removed {before - len(df)} duplicate rows.")
    return df


def drop_broken_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are structurally broken, not just noisy -- a missing
    target or missing entity ID can't be fixed by imputation."""
    before = len(df)
    if "price" in df.columns:
        df = df[df["price"] > 0]  # also drops NaN price
    if "car_id" in df.columns:
        df = df.dropna(subset=["car_id"])
    logger.info(f"Dropped {before - len(df)} structurally broken rows.")
    return df


# Orchestrator

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning that needs no fitted statistics and is safe to run on train
    and test independently, before any combining/splitting. Runs each step
    in order -- see individual functions above for what each does and why
    it's safe pre-split.
    """
    df = rename_columns(df)
    df = normalize_na_placeholders(df)
    df = fix_data_types(df)
    df = fuzzy_match_categoricals(df)
    df = null_impossible_values(df)
    df = remove_duplicate_rows(df)
    df = drop_broken_rows(df)
    return df
