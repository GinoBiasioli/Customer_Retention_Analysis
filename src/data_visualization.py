# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_pareto_curve(pareto_df, metric_col, title, ylabel):
    """
    Plot Pareto curve for a metric.
    """
    cum_share_col = f"cum_share_{metric_col}"

    plt.figure(figsize=(8, 6))
    plt.plot(
        pareto_df["customer_share"] * 100,
        pareto_df[cum_share_col] * 100
    )
    plt.axhline(80, linestyle="--")
    plt.axvline(20, linestyle="--")
    plt.title(title)
    plt.xlabel("Top % of customers")
    plt.ylabel(ylabel)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def plot_top_share_bar(summary_df, title, ylabel):
    """
    Plot how much of the metric is captured by top customer groups.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(
        [f"Top {x}%" for x in summary_df["top_customer_pct"]],
        summary_df["metric_share_pct"]
    )
    plt.title(title)
    plt.xlabel("Customer group")
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def plot_buyer_type_counts(customer_df):
    """
    Bar chart of buyer type counts.
    """
    counts = customer_df["buyer_type"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Customer Count by Buyer Type")
    plt.xlabel("Buyer type")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.show()




def plot_rfm_segment_counts(rfm, segment_col="segment_global", title="RFM Segment Counts"):
    """
    Plot customer counts by RFM segment.

    Parameters
    ----------
    rfm : pd.DataFrame
        RFM table containing a segment column.
    segment_col : str
        Column name with the segment labels
        (e.g. 'segment_global' or 'segment_buyer_type').
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, 6))

    order = rfm[segment_col].value_counts().index

    sns.countplot(
        data=rfm,
        y=segment_col,
        order=order
    )

    plt.title(title)
    plt.xlabel("Number of customers")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_buyer_behavior_scatter(customer_behavior):
    """
    Scatter plot of log avg quantity vs log avg invoice value by buyer type.
    """
    plt.figure(figsize=(8, 6))

    for group in customer_behavior["buyer_type"].unique():
        temp = customer_behavior[customer_behavior["buyer_type"] == group]
        plt.scatter(
            np.log1p(temp["avg_qty_per_invoice"]),
            np.log1p(temp["avg_invoice_value"]),
            alpha=0.5,
            label=group
        )

    plt.title("Customer Behavior by Buyer Type")
    plt.xlabel("log(1 + avg quantity per invoice)")
    plt.ylabel("log(1 + avg invoice value)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_buyer_type_boxplots(customer_df):
    """
    Boxplots comparing the two buyer groups in key metrics.
    """
    metrics = [
        ("avg_qty_per_invoice", "Average Quantity per Invoice"),
        ("avg_invoice_value", "Average Invoice Value"),
        ("total_revenue", "Total Revenue"),
        ("n_invoices", "Number of Invoices")
    ]

    groups = ["Retail-like", "Bulk / commercial-like"]

    for col, label in metrics:
        data_to_plot = [
            customer_df.loc[customer_df["buyer_type"] == group, col].dropna()
            for group in groups
        ]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data_to_plot, tick_labels=groups, showmeans=True)
        plt.title(f"{label} by Buyer Type")
        plt.xlabel("Buyer type")
        plt.ylabel(label)
        plt.tight_layout()
        plt.show()


def plot_buyer_type_boxplots_log(customer_df):
    """
    Log-scale boxplots for highly skewed variables.
    """
    metrics = [
        ("avg_qty_per_invoice", "log(1 + avg quantity per invoice)"),
        ("avg_invoice_value", "log(1 + avg invoice value)"),
        ("total_revenue", "log(1 + total revenue)"),
        ("n_invoices", "log(1 + number of invoices)")
    ]

    groups = ["Retail-like", "Bulk / commercial-like"]

    for col, label in metrics:
        data_to_plot = [
            np.log1p(customer_df.loc[customer_df["buyer_type"] == group, col].dropna())
            for group in groups
        ]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data_to_plot, tick_labels=groups, showmeans=True)
        plt.title(f"{label} by Buyer Type")
        plt.xlabel("Buyer type")
        plt.ylabel(label)
        plt.tight_layout()
        plt.show()

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



def plot_cohort_heatmap(
    retention_matrix,
    title="Cohort Retention Heatmap",
    figsize=(12, 6),
    cmap="Blues",
    cohort_label_format="%Y-%m"
):
    """
    Plot a cohort retention heatmap with cleaner formatting.

    Parameters
    ----------
    retention_matrix : pd.DataFrame
        Cohort retention table where rows = cohort months
        and columns = months since first purchase.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for the heatmap.
    cohort_label_format : str
        Datetime format for cohort labels.
    """

    # Work on a copy so the original matrix is not modified
    plot_data = retention_matrix.copy()

    # Clean row labels (cohort month)
    if isinstance(plot_data.index, pd.DatetimeIndex):
        plot_data.index = plot_data.index.strftime(cohort_label_format)
    else:
        # In case index is period-like or string-like, try to standardize it
        try:
            plot_data.index = pd.to_datetime(plot_data.index).strftime(cohort_label_format)
        except Exception:
            plot_data.index = plot_data.index.astype(str)

    # Clean column labels
    # Optional: ensure they appear as integers 1, 2, 3...
    try:
        plot_data.columns = [int(col) for col in plot_data.columns]
    except Exception:
        pass

    # Mask missing values so the empty triangle looks cleaner
    mask = plot_data.isna()

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        plot_data,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Retention %"}
    )

    ax.set_title(title, fontsize=17, pad=10)
    ax.set_xlabel("Months since first purchase", fontsize=14)
    ax.set_ylabel("Cohort month", fontsize=14)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_transition_heatmap(transition_matrix, title="Segment Transition Matrix"):
    """
    Plot transition matrix heatmap.
    """
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".1f",
        cmap="Oranges",
        linewidths=0.5
    )
    plt.title(title)
    plt.xlabel("Late-period segment")
    plt.ylabel("Early-period segment")
    plt.tight_layout()
    plt.show()


def plot_bulk_feature_boxplots(bulk_customer_df):
    """
    Deep-dive boxplots for bulk/commercial-like customers.
    """
    metrics = [
        ("avg_days_between_purchases", "Avg days between purchases"),
        ("product_diversity_ratio", "Product diversity ratio"),
        ("basket_value_per_product", "Basket value per product")
    ]

    for col, label in metrics:
        plot_df = bulk_customer_df[[col, "segment_global"]].dropna()

        plt.figure(figsize=(10, 5))
        sns.boxplot(data=plot_df, x="segment_global", y=col)
        plt.title(f"{label} across Global RFM Segments (Bulk customers)")
        plt.xlabel("")
        plt.ylabel(label)
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.show()


def plot_bulk_interpurchase_hist(bulk_customer_df):
    """
    Distribution of inter-purchase time for bulk/commercial-like customers.
    """
    plot_df = bulk_customer_df["avg_days_between_purchases"].dropna()

    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df, bins=30)
    plt.title("Distribution of Avg Days Between Purchases - Bulk Customers")
    plt.xlabel("Average days between purchases")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()