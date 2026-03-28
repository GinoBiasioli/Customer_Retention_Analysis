# -*- coding: utf-8 -*-

import pandas as pd


def load_raw_data(data_path):
    """
    Load raw Online Retail dataset.
    """
    df = pd.read_csv(
        data_path,
        sep=";",
        encoding="latin1"
    )
    return df


def basic_inspection(df):
    """
    Print basic information about the raw dataset.
    """
    print("=" * 80)
    print("RAW DATA INSPECTION")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
