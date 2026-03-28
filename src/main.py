from config import DATA_PATH
from data_import import load_raw_data, basic_inspection
from data_preparation import (
    prepare_raw_columns,
    print_data_quality_checks,
    clean_transactions,
    build_customer_table,
    build_invoice_table,
    build_customer_behavior_table,
    add_advanced_customer_features,
    build_monthly_cohort_table,
    build_monthly_cohort_retention,
    build_customer_period_rfm_input
)
from data_classification import (
    build_rfm_table,
    assign_rfm_scores,
    build_rfm_segments,
    assign_rfm_scores_by_buyer_type,
    build_rfm_summary_by_buyer_type,
    build_segment_crosstab,
    build_segment_crosstab_within_type,
    build_pareto_table,
    pareto_top_share_summary,
    classify_buyer_type,
    build_buyer_summary,
    build_transition_matrix
)
from data_visualization import (
    plot_pareto_curve,
    plot_top_share_bar,
    plot_buyer_type_counts,
    plot_rfm_segment_counts,
    plot_buyer_behavior_scatter,
    plot_buyer_type_boxplots_log,
    plot_rfm_global_log,
    plot_rfm_by_buyer_type_boxplots_log,
    plot_cohort_heatmap,
    plot_transition_heatmap,
    plot_bulk_feature_boxplots,
    plot_bulk_interpurchase_hist
)


def main():
    # -------------------------------------------------------------------------
    # 1) LOAD + INSPECT
    # -------------------------------------------------------------------------
    df = load_raw_data(DATA_PATH)
    basic_inspection(df)

    # -------------------------------------------------------------------------
    # 2) PREPARE + CLEAN
    # -------------------------------------------------------------------------
    df = prepare_raw_columns(df)
    print_data_quality_checks(df)
    df_clean = clean_transactions(df)

    # -------------------------------------------------------------------------
    # 3) BUILD CORE TABLES
    # -------------------------------------------------------------------------
    customer_df = build_customer_table(df_clean)
    invoice_df = build_invoice_table(df_clean)
    customer_behavior = build_customer_behavior_table(invoice_df)

    # -------------------------------------------------------------------------
    # 4) ADVANCED BEHAVIOR FEATURES
    # -------------------------------------------------------------------------
    customer_behavior = add_advanced_customer_features(invoice_df, customer_behavior)

    # -------------------------------------------------------------------------
    # 5) BUYER TYPE CLASSIFICATION
    # -------------------------------------------------------------------------
    customer_behavior = classify_buyer_type(customer_behavior, quantile=0.85)

    customer_df = customer_df.merge(
        customer_behavior[
            [
                "CustomerID",
                "avg_qty_per_invoice",
                "median_qty_per_invoice",
                "max_qty_in_invoice",
                "avg_invoice_value",
                "median_invoice_value",
                "max_invoice_value",
                "avg_distinct_products",
                "max_distinct_products",
                "avg_days_between_purchases",
                "median_days_between_purchases",
                "product_diversity_ratio",
                "basket_value_per_product",
                "bulk_score",
                "buyer_type"
            ]
        ],
        on="CustomerID",
        how="left"
    )

    buyer_summary = build_buyer_summary(customer_df)

    # -------------------------------------------------------------------------
    # 6) GLOBAL RFM
    # -------------------------------------------------------------------------
    rfm = build_rfm_table(customer_df)
    rfm = assign_rfm_scores(rfm)
    rfm, segment_summary_global = build_rfm_segments(rfm)

    # -------------------------------------------------------------------------
    # 7) RFM WITHIN BUYER TYPE
    # -------------------------------------------------------------------------
    rfm = assign_rfm_scores_by_buyer_type(rfm)
    segment_summary_by_type = build_rfm_summary_by_buyer_type(rfm)

    segment_crosstab_global = build_segment_crosstab(rfm)
    segment_crosstab_within_type = build_segment_crosstab_within_type(rfm)

    # -------------------------------------------------------------------------
    # 8) MERGE RFM BACK INTO CUSTOMER TABLE
    # -------------------------------------------------------------------------
    customer_df = customer_df.merge(
        rfm[
            [
                "CustomerID",
                "R_score",
                "F_score",
                "M_score",
                "RFM_score",
                "RFM_total",
                "segment_global",
                "R_score_bt",
                "F_score_bt",
                "M_score_bt",
                "RFM_score_bt",
                "RFM_total_bt",
                "segment_buyer_type"
            ]
        ],
        on="CustomerID",
        how="left"
    )

    # -------------------------------------------------------------------------
    # 9) PARETO
    # -------------------------------------------------------------------------
    pareto_revenue = build_pareto_table(customer_df, metric_col="total_revenue")
    pareto_revenue_summary = pareto_top_share_summary(
        pareto_revenue,
        metric_col="total_revenue"
    )

    pareto_invoices = build_pareto_table(customer_df, metric_col="n_invoices")
    pareto_invoices_summary = pareto_top_share_summary(
        pareto_invoices,
        metric_col="n_invoices"
    )

    # -------------------------------------------------------------------------
    # 10) COHORT ANALYSIS - ALL CUSTOMERS
    # -------------------------------------------------------------------------
    cohort_base = build_monthly_cohort_table(df_clean)
    cohort_retention = build_monthly_cohort_retention(cohort_base)

    # -------------------------------------------------------------------------
    # 11) BULK / COMMERCIAL-LIKE DEEP DIVE
    # -------------------------------------------------------------------------
    bulk_customers = customer_df.loc[
        customer_df["buyer_type"] == "Bulk / commercial-like",
        "CustomerID"
    ].unique()

    df_bulk = df_clean[df_clean["CustomerID"].isin(bulk_customers)].copy()
    bulk_behavior = customer_behavior[
        customer_behavior["CustomerID"].isin(bulk_customers)
    ].copy()
    bulk_customer_df = customer_df[
        customer_df["CustomerID"].isin(bulk_customers)
    ].copy()

    bulk_cohort_base = build_monthly_cohort_table(df_bulk)
    bulk_cohort_retention = build_monthly_cohort_retention(bulk_cohort_base)

    # -------------------------------------------------------------------------
    # 11B) RETAIL / NON-BULK COHORT
    # -------------------------------------------------------------------------
    retail_customers = customer_df.loc[
        customer_df["buyer_type"] != "Bulk / commercial-like",
        "CustomerID"
    ].unique()
    
    df_retail = df_clean[df_clean["CustomerID"].isin(retail_customers)].copy()
    
    retail_cohort_base = build_monthly_cohort_table(df_retail)
    retail_cohort_retention = build_monthly_cohort_retention(retail_cohort_base)

    # -------------------------------------------------------------------------
    # 12) SEGMENT TRANSITIONS (EARLY vs LATE PERIOD) - BULK ONLY
    # -------------------------------------------------------------------------
    period_rfm = build_customer_period_rfm_input(df_bulk)

    rfm_early = build_rfm_table(period_rfm["early"])
    rfm_early = assign_rfm_scores(rfm_early)
    rfm_early, _ = build_rfm_segments(rfm_early)
    rfm_early = rfm_early[["CustomerID", "segment_global"]].rename(
        columns={"segment_global": "segment_early"}
    )

    rfm_late = build_rfm_table(period_rfm["late"])
    rfm_late = assign_rfm_scores(rfm_late)
    rfm_late, _ = build_rfm_segments(rfm_late)
    rfm_late = rfm_late[["CustomerID", "segment_global"]].rename(
        columns={"segment_global": "segment_late"}
    )

    bulk_transition = build_transition_matrix(rfm_early, rfm_late)

    # -------------------------------------------------------------------------
    # 13) EXTRA SUMMARY TABLES
    # -------------------------------------------------------------------------
    bulk_feature_summary = (
        bulk_customer_df[
            [
                "total_revenue",
                "n_invoices",
                "avg_order_value",
                "avg_qty_per_invoice",
                "avg_days_between_purchases",
                "product_diversity_ratio",
                "basket_value_per_product",
                "segment_global"
            ]
        ]
        .groupby("segment_global")
        .agg(
            customers=("total_revenue", "count"),
            avg_revenue=("total_revenue", "mean"),
            median_revenue=("total_revenue", "median"),
            avg_invoices=("n_invoices", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_qty_per_invoice=("avg_qty_per_invoice", "mean"),
            avg_days_between_purchases=("avg_days_between_purchases", "mean"),
            avg_product_diversity_ratio=("product_diversity_ratio", "mean"),
            avg_basket_value_per_product=("basket_value_per_product", "mean")
        )
        .sort_values("avg_revenue", ascending=False)
    )

    # -------------------------------------------------------------------------
    # 14) PRINT SUMMARIES
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("GLOBAL RFM SEGMENT SUMMARY")
    print("=" * 80)
    print(segment_summary_global)

    print("=" * 80)
    print("RFM SEGMENT SUMMARY WITHIN BUYER TYPE")
    print("=" * 80)
    print(segment_summary_by_type)

    print("=" * 80)
    print("BUYER TYPE x GLOBAL RFM SEGMENT (%)")
    print("=" * 80)
    print(segment_crosstab_global)

    print("=" * 80)
    print("BUYER TYPE x WITHIN-TYPE RFM SEGMENT (%)")
    print("=" * 80)
    print(segment_crosstab_within_type)

    print("=" * 80)
    print("PARETO SUMMARY - REVENUE")
    print("=" * 80)
    print(pareto_revenue_summary)

    print("=" * 80)
    print("PARETO SUMMARY - INVOICES")
    print("=" * 80)
    print(pareto_invoices_summary)

    print("=" * 80)
    print("BUYER SUMMARY")
    print("=" * 80)
    print(buyer_summary)

    print("=" * 80)
    print("BULK FEATURE SUMMARY BY GLOBAL SEGMENT")
    print("=" * 80)
    print(bulk_feature_summary)

    print("=" * 80)
    print("BULK SEGMENT TRANSITION MATRIX (%)")
    print("=" * 80)
    print(bulk_transition)

    # -------------------------------------------------------------------------
    # 15) PLOTS
    # -------------------------------------------------------------------------
    # Core portfolio plots
    plot_pareto_curve(
        pareto_revenue,
        metric_col="total_revenue",
        title="Pareto Curve: Cumulative Revenue by Customer Share",
        ylabel="Cumulative % of revenue"
    )

    plot_top_share_bar(
        pareto_revenue_summary,
        title="Revenue Share Captured by Top Customers",
        ylabel="% of total revenue"
    )

    plot_pareto_curve(
        pareto_invoices,
        metric_col="n_invoices",
        title="Pareto Curve: Cumulative Invoices by Customer Share",
        ylabel="Cumulative % of invoices"
    )

    plot_top_share_bar(
        pareto_invoices_summary,
        title="Invoice Share Captured by Top Customers",
        ylabel="% of total invoices"
    )

    plot_buyer_type_counts(customer_df)
    plot_buyer_behavior_scatter(customer_behavior)
    plot_buyer_type_boxplots_log(customer_df)

    plot_rfm_global_log(rfm)
    plot_rfm_by_buyer_type_boxplots_log(rfm)

    plot_rfm_segment_counts(
        rfm,
        segment_col="segment_global",
        title="Global RFM Segment Counts"
    )

    # Cohort: all customers
    plot_cohort_heatmap(
        cohort_retention,
        title="Monthly Cohort Retention - All Customers"
    )

    # Bulk deep dive
    plot_cohort_heatmap(
        bulk_cohort_retention,
        title="Monthly Cohort Retention - Bulk / Commercial-like Customers"
    )


    plot_cohort_heatmap(
        retail_cohort_retention,
        title="Monthly Cohort Retention - Retail Customers"
)
    plot_bulk_feature_boxplots(bulk_customer_df)
    plot_bulk_interpurchase_hist(bulk_customer_df)

    plot_transition_heatmap(
        bulk_transition,
        title="Bulk / Commercial-like Segment Transitions"
    )

    return {
        "df_clean": df_clean,
        "customer_df": customer_df,
        "invoice_df": invoice_df,
        "customer_behavior": customer_behavior,
        "rfm": rfm,
        "segment_summary_global": segment_summary_global,
        "segment_summary_by_type": segment_summary_by_type,
        "segment_crosstab_global": segment_crosstab_global,
        "segment_crosstab_within_type": segment_crosstab_within_type,
        "pareto_revenue": pareto_revenue,
        "pareto_revenue_summary": pareto_revenue_summary,
        "pareto_invoices": pareto_invoices,
        "pareto_invoices_summary": pareto_invoices_summary,
        "buyer_summary": buyer_summary,
        "cohort_retention": cohort_retention,
        "bulk_cohort_retention": bulk_cohort_retention,
        "bulk_transition": bulk_transition,
        "bulk_feature_summary": bulk_feature_summary
    }


if __name__ == "__main__":
    results = main()

