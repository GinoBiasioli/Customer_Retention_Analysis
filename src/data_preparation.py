# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def prepare_raw_columns(df):
    """
    Standardize raw columns and create basic derived variables.
    """
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

    df["UnitPrice"] = (
        df["UnitPrice"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    return df


def print_data_quality_checks(df):
    """
    Print useful raw-data quality checks before cleaning.
    """
    print("=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)

    print("\nSummary stats:")
    print(df[["Quantity", "UnitPrice", "Revenue"]].describe())

    missing_customer_pct = df["CustomerID"].isna().mean() * 100
    print(f"\nMissing CustomerID: {missing_customer_pct:.2f}%")

    print("\nCancellation rows:", df["IsCancellation"].sum())
    print("Rows with negative quantity:", (df["Quantity"] < 0).sum())
    print("Rows with zero quantity:", (df["Quantity"] == 0).sum())
    print("Rows with zero unit price:", (df["UnitPrice"] == 0).sum())
    print("Rows with negative revenue:", (df["Revenue"] < 0).sum())


def clean_transactions(df):
    """
    Keep valid customer transactions only.
    """
    df_clean = df.copy()

    df_clean = df_clean.dropna(subset=["CustomerID"])
    df_clean = df_clean[~df_clean["IsCancellation"]]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean = df_clean.drop_duplicates()

    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

    print("=" * 80)
    print("CLEAN DATA")
    print("=" * 80)
    print(f"Clean shape: {df_clean.shape}")
    print("Unique customers:", df_clean["CustomerID"].nunique())
    print("Unique invoices:", df_clean["InvoiceNo"].nunique())

    return df_clean


def build_customer_table(df_clean):
    """
    Build customer-level dataset for LTV / RFM analysis.
    """
    snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_df = (
        df_clean.groupby("CustomerID")
        .agg(
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            n_invoices=("InvoiceNo", "nunique"),
            total_quantity=("Quantity", "sum"),
            total_revenue=("Revenue", "sum"),
            country=("Country", "first")
        )
        .reset_index()
    )

    customer_df["recency_days"] = (
        snapshot_date - customer_df["last_purchase"]
    ).dt.days

    customer_df["active_span_days"] = (
        customer_df["last_purchase"] - customer_df["first_purchase"]
    ).dt.days

    customer_df["tenure_days"] = (
        snapshot_date - customer_df["first_purchase"]
    ).dt.days

    customer_df["avg_order_value"] = (
        customer_df["total_revenue"] / customer_df["n_invoices"]
    )

    customer_df["avg_items_per_order"] = (
        customer_df["total_quantity"] / customer_df["n_invoices"]
    )

    customer_df["purchase_frequency"] = np.where(
        customer_df["tenure_days"] > 0,
        customer_df["n_invoices"] / customer_df["tenure_days"],
        customer_df["n_invoices"]
    )

    customer_df["country_group"] = np.where(
        customer_df["country"] == "United Kingdom",
        "UK",
        "Other"
    )

    customer_df["log_total_revenue"] = np.log1p(customer_df["total_revenue"])
    customer_df["log_avg_order_value"] = np.log1p(customer_df["avg_order_value"])
    customer_df["log_total_quantity"] = np.log1p(customer_df["total_quantity"])
    customer_df["log_frequency"] = np.log1p(customer_df["n_invoices"])
    customer_df["log_monetary"] = np.log1p(customer_df["total_revenue"])

    print("=" * 80)
    print("CUSTOMER TABLE")
    print("=" * 80)
    print(customer_df.head())

    return customer_df


def build_invoice_table(df_clean):
    """
    Build invoice-level dataset.
    """
    invoice_df = (
        df_clean.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            invoice_date=("InvoiceDate", "min"),
            invoice_quantity=("Quantity", "sum"),
            invoice_revenue=("Revenue", "sum"),
            n_products=("StockCode", "nunique"),
            n_lines=("StockCode", "count")
        )
    )

    print("=" * 80)
    print("INVOICE TABLE")
    print("=" * 80)
    print(invoice_df.head())

    return invoice_df


def build_customer_behavior_table(invoice_df):
    """
    Build customer-level invoice behavior features.
    """
    customer_behavior = (
        invoice_df.groupby("CustomerID", as_index=False)
        .agg(
            n_invoices=("InvoiceNo", "nunique"),
            avg_qty_per_invoice=("invoice_quantity", "mean"),
            median_qty_per_invoice=("invoice_quantity", "median"),
            max_qty_in_invoice=("invoice_quantity", "max"),
            avg_invoice_value=("invoice_revenue", "mean"),
            median_invoice_value=("invoice_revenue", "median"),
            max_invoice_value=("invoice_revenue", "max"),
            avg_distinct_products=("n_products", "mean"),
            max_distinct_products=("n_products", "max")
        )
    )

    print("=" * 80)
    print("CUSTOMER BEHAVIOR TABLE")
    print("=" * 80)
    print(customer_behavior.head())

    return customer_behavior

import numpy as np
import pandas as pd


def add_advanced_customer_features(invoice_df, customer_behavior):
    """
    Add richer behavioral features:
    - average/median days between purchases
    - product diversity ratio
    - basket value per product
    """
    invoice_sorted = invoice_df.sort_values(["CustomerID", "invoice_date"]).copy()

    invoice_sorted["prev_invoice_date"] = (
        invoice_sorted.groupby("CustomerID")["invoice_date"].shift(1)
    )

    invoice_sorted["days_since_prev_invoice"] = (
        invoice_sorted["invoice_date"] - invoice_sorted["prev_invoice_date"]
    ).dt.days

    gap_features = (
        invoice_sorted.groupby("CustomerID", as_index=False)
        .agg(
            avg_days_between_purchases=("days_since_prev_invoice", "mean"),
            median_days_between_purchases=("days_since_prev_invoice", "median")
        )
    )

    behavior = customer_behavior.merge(gap_features, on="CustomerID", how="left")

    behavior["product_diversity_ratio"] = np.where(
        behavior["avg_qty_per_invoice"] > 0,
        behavior["avg_distinct_products"] / behavior["avg_qty_per_invoice"],
        np.nan
    )

    behavior["basket_value_per_product"] = np.where(
        behavior["avg_distinct_products"] > 0,
        behavior["avg_invoice_value"] / behavior["avg_distinct_products"],
        np.nan
    )

    print("=" * 80)
    print("CUSTOMER BEHAVIOR TABLE - ADVANCED FEATURES")
    print("=" * 80)
    print(
        behavior[
            [
                "CustomerID",
                "avg_days_between_purchases",
                "median_days_between_purchases",
                "product_diversity_ratio",
                "basket_value_per_product"
            ]
        ].head()
    )

    return behavior


def build_monthly_cohort_table(df_clean):
    """
    Create customer-month activity table for cohort analysis.
    """
    df = df_clean.copy()
    df["invoice_month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    first_month = (
        df.groupby("CustomerID", as_index=False)["invoice_month"]
        .min()
        .rename(columns={"invoice_month": "cohort_month"})
    )

    monthly_activity = (
        df.groupby(["CustomerID", "invoice_month"], as_index=False)
        .agg(n_invoices=("InvoiceNo", "nunique"))
    )

    cohort_df = monthly_activity.merge(first_month, on="CustomerID", how="left")

    cohort_df["cohort_index"] = (
        (cohort_df["invoice_month"].dt.year - cohort_df["cohort_month"].dt.year) * 12
        + (cohort_df["invoice_month"].dt.month - cohort_df["cohort_month"].dt.month)
        + 1
    )

    return cohort_df


def build_monthly_cohort_retention(cohort_df):
    """
    Build cohort retention matrix in %.
    """
    cohort_counts = (
        cohort_df.groupby(["cohort_month", "cohort_index"])["CustomerID"]
        .nunique()
        .reset_index()
        .rename(columns={"CustomerID": "n_customers"})
    )

    cohort_sizes = (
        cohort_counts[cohort_counts["cohort_index"] == 1]
        [["cohort_month", "n_customers"]]
        .rename(columns={"n_customers": "cohort_size"})
    )

    retention = cohort_counts.merge(cohort_sizes, on="cohort_month", how="left")
    retention["retention_pct"] = 100 * retention["n_customers"] / retention["cohort_size"]

    retention_matrix = retention.pivot(
        index="cohort_month",
        columns="cohort_index",
        values="retention_pct"
    ).round(1)

    return retention_matrix


def build_customer_period_rfm_input(df_subset):
    """
    Split a subset of transactions into early and late periods,
    then create customer-level RFM-ready inputs for each period.
    """
    df = df_subset.copy()

    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()
    midpoint = min_date + (max_date - min_date) / 2

    early_df = df[df["InvoiceDate"] <= midpoint].copy()
    late_df = df[df["InvoiceDate"] > midpoint].copy()

    def _customer_period_table(period_df):
        if period_df.empty:
            return pd.DataFrame(
                columns=["CustomerID", "buyer_type", "recency_days", "n_invoices", "total_revenue"]
            )

        snapshot_date = period_df["InvoiceDate"].max() + pd.Timedelta(days=1)

        out = (
            period_df.groupby("CustomerID", as_index=False)
            .agg(
                last_purchase=("InvoiceDate", "max"),
                n_invoices=("InvoiceNo", "nunique"),
                total_revenue=("Revenue", "sum")
            )
        )

        out["recency_days"] = (snapshot_date - out["last_purchase"]).dt.days
        out["buyer_type"] = "Bulk / commercial-like"
        return out[["CustomerID", "buyer_type", "recency_days", "n_invoices", "total_revenue"]]

    return {
        "early": _customer_period_table(early_df),
        "late": _customer_period_table(late_df)
    }
