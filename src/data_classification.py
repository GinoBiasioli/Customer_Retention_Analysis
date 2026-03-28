# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# RFM BASE
# =============================================================================
def build_rfm_table(customer_df):
    """
    Create classic RFM base table.
    """
    rfm = customer_df[
        ["CustomerID", "buyer_type", "recency_days", "n_invoices", "total_revenue"]
    ].copy()

    rfm = rfm.rename(columns={
        "recency_days": "Recency",
        "n_invoices": "Frequency",
        "total_revenue": "Monetary"
    })

    rfm["log_frequency"] = np.log1p(rfm["Frequency"])
    rfm["log_monetary"] = np.log1p(rfm["Monetary"])

    return rfm


# =============================================================================
# GLOBAL RFM SCORING
# =============================================================================
def assign_rfm_scores(rfm):
    """
    Assign global quantile-based RFM scores.
    """
    rfm = rfm.copy()

    rfm["R_score"] = pd.qcut(
        rfm["Recency"],
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop"
    ).astype(int)

    rfm["F_score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    ).astype(int)

    rfm["M_score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop"
    ).astype(int)

    rfm["RFM_score"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )

    rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    return rfm


# =============================================================================
# SEGMENT LABELS
# =============================================================================
def assign_rfm_segment(row, r_col="R_score", f_col="F_score", m_col="M_score"):
    r = row[r_col]
    f = row[f_col]
    m = row[m_col]

    if r >= 4 and f >= 4 and m >= 4:
        return "Best customers"
    elif r >= 4 and f <= 2:
        return "Recent customers"
    elif r <= 2 and f >= 4 and m >= 4:
        return "At risk high value"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Low value / inactive"
    elif f >= 4 and m >= 4:
        return "Loyal high value"
    elif f >= 4:
        return "Loyal customers"
    elif m >= 4:
        return "Big spenders"
    else:
        return "Mid-value customers"


def build_rfm_segments(rfm):
    """
    Add segment labels using global RFM scores.
    """
    rfm = rfm.copy()
    rfm["segment_global"] = rfm.apply(
        assign_rfm_segment,
        axis=1,
        r_col="R_score",
        f_col="F_score",
        m_col="M_score"
    )

    segment_summary = (
        rfm.groupby("segment_global")
        .agg(
            customers=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            total_monetary=("Monetary", "sum")
        )
        .sort_values("total_monetary", ascending=False)
    )

    segment_summary["customer_share_pct"] = (
        100 * segment_summary["customers"] / segment_summary["customers"].sum()
    ).round(2)

    segment_summary["revenue_share_pct"] = (
        100 * segment_summary["total_monetary"] / segment_summary["total_monetary"].sum()
    ).round(2)

    return rfm, segment_summary


# =============================================================================
# RFM WITHIN BUYER TYPE
# =============================================================================
def assign_rfm_scores_by_buyer_type(rfm):
    """
    Assign RFM scores separately within each buyer_type.
    This makes customers comparable inside their own behavioral segment.
    """
    rfm = rfm.copy()

    def score_group(group):
        group = group.copy()

        group["R_score_bt"] = pd.qcut(
            group["Recency"],
            q=5,
            labels=[5, 4, 3, 2, 1],
            duplicates="drop"
        ).astype(int)

        group["F_score_bt"] = pd.qcut(
            group["Frequency"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        ).astype(int)

        group["M_score_bt"] = pd.qcut(
            group["Monetary"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        ).astype(int)

        group["RFM_score_bt"] = (
            group["R_score_bt"].astype(str)
            + group["F_score_bt"].astype(str)
            + group["M_score_bt"].astype(str)
        )

        group["RFM_total_bt"] = (
            group["R_score_bt"] + group["F_score_bt"] + group["M_score_bt"]
        )

        group["segment_buyer_type"] = group.apply(
            assign_rfm_segment,
            axis=1,
            r_col="R_score_bt",
            f_col="F_score_bt",
            m_col="M_score_bt"
        )

        return group

    rfm = (
        rfm.groupby("buyer_type", group_keys=False)
        .apply(score_group)
        .reset_index(drop=True)
    )

    return rfm


def build_rfm_summary_by_buyer_type(rfm):
    """
    Summary of within-buyer-type RFM segments.
    """
    summary = (
        rfm.groupby(["buyer_type", "segment_buyer_type"])
        .agg(
            customers=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            total_monetary=("Monetary", "sum")
        )
        .sort_values(["buyer_type", "total_monetary"], ascending=[True, False])
    )

    summary["customer_share_within_type_pct"] = (
        100 * summary["customers"]
        / summary.groupby(level=0)["customers"].transform("sum")
    ).round(2)

    summary["revenue_share_within_type_pct"] = (
        100 * summary["total_monetary"]
        / summary.groupby(level=0)["total_monetary"].transform("sum")
    ).round(2)

    return summary


# =============================================================================
# PARETO
# =============================================================================
def build_pareto_table(customer_df, metric_col):
    """
    Generic Pareto table for a customer-level metric.
    """
    pareto_df = (
        customer_df[["CustomerID", metric_col]]
        .sort_values(metric_col, ascending=False)
        .reset_index(drop=True)
    )

    pareto_df["customer_rank"] = np.arange(1, len(pareto_df) + 1)
    pareto_df["customer_share"] = pareto_df["customer_rank"] / len(pareto_df)

    cum_col = f"cum_{metric_col}"
    cum_share_col = f"cum_share_{metric_col}"

    pareto_df[cum_col] = pareto_df[metric_col].cumsum()
    pareto_df[cum_share_col] = pareto_df[cum_col] / pareto_df[metric_col].sum()

    return pareto_df


def pareto_top_share_summary(pareto_df, metric_col, top_groups=None):
    """
    Revenue / invoice concentration in top customer groups.
    """
    if top_groups is None:
        top_groups = [1, 5, 10, 20, 30]

    total_metric = pareto_df[metric_col].sum()
    rows = []

    for pct in top_groups:
        n_top = int(np.ceil(len(pareto_df) * (pct / 100)))
        share = pareto_df.iloc[:n_top][metric_col].sum() / total_metric
        rows.append({
            "top_customer_pct": pct,
            "metric_share_pct": round(share * 100, 2)
        })

    return pd.DataFrame(rows)


# =============================================================================
# BUYER TYPE CLASSIFICATION
# =============================================================================
def classify_buyer_type(customer_behavior, quantile=0.85):
    """
    Classify customers into Retail-like vs Bulk / commercial-like.
    """
    cb = customer_behavior.copy()

    qty_threshold = cb["avg_qty_per_invoice"].quantile(quantile)
    value_threshold = cb["avg_invoice_value"].quantile(quantile)
    max_qty_threshold = cb["max_qty_in_invoice"].quantile(quantile)

    print("=" * 80)
    print("BUYER TYPE THRESHOLDS")
    print("=" * 80)
    print(f"avg_qty_per_invoice {int(quantile * 100)}th percentile:", round(qty_threshold, 2))
    print(f"avg_invoice_value {int(quantile * 100)}th percentile:", round(value_threshold, 2))
    print(f"max_qty_in_invoice {int(quantile * 100)}th percentile:", round(max_qty_threshold, 2))

    cb["high_avg_qty"] = (cb["avg_qty_per_invoice"] >= qty_threshold).astype(int)
    cb["high_avg_value"] = (cb["avg_invoice_value"] >= value_threshold).astype(int)
    cb["high_max_qty"] = (cb["max_qty_in_invoice"] >= max_qty_threshold).astype(int)

    cb["bulk_score"] = (
        cb["high_avg_qty"]
        + cb["high_avg_value"]
        + cb["high_max_qty"]
    )

    cb["buyer_type"] = np.where(
        cb["bulk_score"] >= 2,
        "Bulk / commercial-like",
        "Retail-like"
    )

    return cb


def build_buyer_summary(customer_df):
    """
    Summary table by buyer type.
    """
    buyer_summary = (
        customer_df.groupby("buyer_type")
        .agg(
            customers=("CustomerID", "count"),
            total_revenue=("total_revenue", "sum"),
            avg_revenue=("total_revenue", "mean"),
            median_revenue=("total_revenue", "median"),
            avg_invoices=("n_invoices", "mean"),
            median_invoices=("n_invoices", "median"),
            avg_order_value=("avg_order_value", "mean"),
            median_order_value=("avg_order_value", "median"),
            avg_qty_per_invoice=("avg_qty_per_invoice", "mean"),
            median_qty_per_invoice=("avg_qty_per_invoice", "median"),
            avg_recency=("recency_days", "mean")
        )
    )

    buyer_summary["customer_share_pct"] = (
        100 * buyer_summary["customers"] / buyer_summary["customers"].sum()
    ).round(2)

    buyer_summary["revenue_share_pct"] = (
        100 * buyer_summary["total_revenue"] / buyer_summary["total_revenue"].sum()
    ).round(2)

    return buyer_summary


# =============================================================================
# PLOTS: GLOBAL
# =============================================================================
def plot_rfm_global(rfm):
    """
    Global distributions of R, F, M.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(rfm["Recency"], bins=40, ax=axes[0])
    axes[0].set_title("Global Recency")
    axes[0].set_xlabel("Recency (days)")

    sns.histplot(rfm["Frequency"], bins=40, ax=axes[1])
    axes[1].set_title("Global Frequency")
    axes[1].set_xlabel("Number of invoices")

    sns.histplot(rfm["Monetary"], bins=40, ax=axes[2])
    axes[2].set_title("Global Monetary")
    axes[2].set_xlabel("Total revenue")

    plt.tight_layout()
    plt.show()


def plot_rfm_global_log(rfm):
    """
    Global distributions with log scale for skewed variables.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(np.log1p(rfm["Recency"]), bins=40, ax=axes[0])
    axes[0].set_title("Global log(1 + Recency)")
    axes[0].set_xlabel("log(1 + Recency)")

    sns.histplot(np.log1p(rfm["Frequency"]), bins=40, ax=axes[1])
    axes[1].set_title("Global log(1 + Frequency)")
    axes[1].set_xlabel("log(1 + Frequency)")

    sns.histplot(np.log1p(rfm["Monetary"]), bins=40, ax=axes[2])
    axes[2].set_title("Global log(1 + Monetary)")
    axes[2].set_xlabel("log(1 + Monetary)")

    plt.tight_layout()
    plt.show()


# =============================================================================
# PLOTS: BY BUYER TYPE
# =============================================================================
def plot_rfm_by_buyer_type_boxplots(rfm):
    """
    Compare R, F, M across buyer types using boxplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=rfm, x="buyer_type", y="Recency", ax=axes[0])
    axes[0].set_title("Recency by Buyer Type")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=rfm, x="buyer_type", y="Frequency", ax=axes[1])
    axes[1].set_title("Frequency by Buyer Type")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)

    sns.boxplot(data=rfm, x="buyer_type", y="Monetary", ax=axes[2])
    axes[2].set_title("Monetary by Buyer Type")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()


def plot_rfm_by_buyer_type_boxplots_log(rfm):
    """
    Compare R, F, M across buyer types using log-transformed boxplots.
    Better when distributions are highly skewed.
    """
    plot_df = rfm.copy()
    plot_df["log_recency"] = np.log1p(plot_df["Recency"])
    plot_df["log_frequency"] = np.log1p(plot_df["Frequency"])
    plot_df["log_monetary"] = np.log1p(plot_df["Monetary"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=plot_df, x="buyer_type", y="log_recency", ax=axes[0])
    axes[0].set_title("log(1 + Recency) by Buyer Type")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=plot_df, x="buyer_type", y="log_frequency", ax=axes[1])
    axes[1].set_title("log(1 + Frequency) by Buyer Type")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)

    sns.boxplot(data=plot_df, x="buyer_type", y="log_monetary", ax=axes[2])
    axes[2].set_title("log(1 + Monetary) by Buyer Type")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()


def plot_rfm_scores_by_buyer_type(rfm):
    """
    Show average global vs within-buyer-type RFM scores by buyer_type.
    """
    score_summary = (
        rfm.groupby("buyer_type")
        .agg(
            avg_R_global=("R_score", "mean"),
            avg_F_global=("F_score", "mean"),
            avg_M_global=("M_score", "mean"),
            avg_R_within_type=("R_score_bt", "mean"),
            avg_F_within_type=("F_score_bt", "mean"),
            avg_M_within_type=("M_score_bt", "mean")
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    score_summary[["avg_R_global", "avg_F_global", "avg_M_global"]].plot(
        kind="bar", ax=axes[0]
    )
    axes[0].set_title("Average Global RFM Scores by Buyer Type")
    axes[0].set_ylabel("Average score")
    axes[0].tick_params(axis="x", rotation=15)

    score_summary[["avg_R_within_type", "avg_F_within_type", "avg_M_within_type"]].plot(
        kind="bar", ax=axes[1]
    )
    axes[1].set_title("Average Within-Type RFM Scores by Buyer Type")
    axes[1].set_ylabel("Average score")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()


# =============================================================================
# CROSS TABLES
# =============================================================================
def build_segment_crosstab(rfm):
    """
    Cross-tab of buyer type vs global RFM segment.
    """
    ctab = pd.crosstab(
        rfm["buyer_type"],
        rfm["segment_global"],
        normalize="index"
    ) * 100

    return ctab.round(2)


def build_segment_crosstab_within_type(rfm):
    """
    Cross-tab of buyer type vs within-buyer-type RFM segment.
    """
    ctab = pd.crosstab(
        rfm["buyer_type"],
        rfm["segment_buyer_type"],
        normalize="index"
    ) * 100

    return ctab.round(2)

def build_transition_matrix(rfm_early, rfm_late):
    """
    Transition matrix from early-period segment to late-period segment.
    Rows sum to 100.
    """
    transition_df = rfm_early.merge(rfm_late, on="CustomerID", how="inner")

    transition = pd.crosstab(
        transition_df["segment_early"],
        transition_df["segment_late"],
        normalize="index"
    ) * 100

    return transition.round(2)