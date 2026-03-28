# Customer Retention Analysis

A project exploring customer value, retention, and purchasing behavior in an e-commerce setting using the Online Retail dataset.

## Overview

This project builds a customer analytics pipeline from raw transaction data to actionable segmentation insights. The analysis combines data cleaning, customer-level feature engineering, buyer-type classification, RFM segmentation, Pareto analysis, and cohort retention.

The main idea that not all customers behave the same way. Instead of treating every buyer as part of one single group, the project first separates **Retail-like** and **Bulk / commercial-like** customers, then analyzes value and retention patterns across those segments.

## What this project covers

* Data cleaning and preparation from raw transaction data
* Customer-level and invoice-level feature engineering
* Buyer-type classification based on purchasing behavior
* Global and within-segment RFM analysis
* Pareto analysis of revenue and invoice concentration
* Monthly cohort retention analysis
* Deeper retention and transition analysis for bulk customers


## Key highlights

* Built from **541,909 raw transactions**
* Final cleaned dataset includes **392,692 transactions**
* Analysis covers **4,338 customers** and **18,532 invoices**
* Bulk / commercial-like buyers represent a relatively small share of customers but a disproportionately large share of revenue
* Cohort analysis shows stronger repeat-purchase behavior among bulk customers than retail customers

## Main insights

### 1. Revenue is highly concentrated

A relatively small group of customers drives a large share of total revenue, making customer concentration a central part of the business story.

### 2. Buyer behavior is structurally different

Retail-like and bulk/commercial-like customers do not follow the same purchase patterns. Bulk buyers tend to place larger and higher-value orders, making a single segmentation approach less informative.

### 3. Retention differs by segment

Monthly cohort retention suggests that bulk customers behave more like repeat accounts, while retail customers show a lower and less persistent return pattern.

### 4. RFM is more useful when layered

Applying RFM after classifying customer type creates a more interpretable view of loyalty, value, and churn risk than applying one universal scoring approach.

## Tech stack

* **Python**
* **pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **HTML / CSS** for the portfolio presentation layer

## Project structure

```text
Customer_Retention_Analysis/
├── data/
│   └── Online Retail.csv
├── src/
│   ├── config.py
│   ├── data_import.py
│   ├── data_preparation.py
│   ├── data_classification.py
│   ├── data_visualization.py
│   └── main.py
└── customer value segmentation.html
```

## How to run

1. Clone the repository
2. Place the dataset in the `data/` folder if needed
3. Update the path in `config.py` if necessary
4. Run:

```bash
python src/main.py
```



**Gino Biasioli**
Data / Business Analytics portfolio project
