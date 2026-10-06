# Project 1 - Customer Segmentation using Python

## Overview

This project implements customer segmentation using Python and machine learning techniques.

The project uses a customer dataset and follows a complete data science workflow, including data inspection, data cleaning, feature engineering, feature selection, feature scaling and K-Means clustering.

The main objective is to identify meaningful customer groups based on customer characteristics, financial information and purchasing behavior, and generate insights that can support targeted marketing strategies.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenPyXL

---

## Dataset

The project uses an Excel dataset containing **53,503 customer records** and **20 original features**.

The dataset includes information such as:

- Customer ID
- Age
- Gender
- Marital Status
- Education Level
- Geographic Information
- Occupation
- Income Level
- Purchase History
- Coverage Amount
- Premium Amount
- Policy Type
- Customer Preferences
- Preferred Communication Channel
- Preferred Contact Time
- Preferred Language

The dataset is included in this project folder because this version of the project is designed to work with a specific dataset.

---

## Project Workflow

### 1. Data Loading

The customer dataset is loaded from an Excel file using Pandas.

### 2. Data Inspection

The dataset is examined to understand:

- Number of rows and columns
- Data types
- Missing values
- Duplicate records
- Numerical features
- Categorical features

### 3. Data Cleaning

The `Purchase History` column is converted into a datetime format.

The dataset is also checked for invalid dates, missing values and duplicate records.

### 4. Feature Engineering

Additional features are created from the purchase date, including:

- Purchase Year
- Purchase Month
- Purchase Day
- Purchase Day of Week
- Recency
- Purchase Quarter
- Purchase Season

Additional customer-related features include:

- Age Group
- Premium Coverage Ratio

### 5. Feature Selection

The following features are selected for customer segmentation:

- Age
- Income Level
- Coverage Amount
- Premium Amount
- Purchase Year
- Purchase Month
- Purchase Day of Week
- Recency

`Customer ID` is excluded because it is an identifier and does not provide meaningful information for clustering.

The existing `Segmentation Group` column is also not used as a clustering feature.

### 6. Feature Scaling

The selected numerical features are standardized using `StandardScaler`.

Scaling is important because the features have different numerical ranges.

### 7. Selecting the Number of Clusters

Two methods are used to evaluate different numbers of clusters:

- Elbow Method
- Silhouette Analysis

The final analysis uses:

**K = 5**

The five-cluster solution was selected mainly for meaningful customer segmentation and business interpretation.

### 8. K-Means Clustering

K-Means clustering is applied to the scaled customer features.

Each customer is assigned to one of five clusters.

### 9. Customer Segments

The resulting clusters are interpreted as:

1. Older Medium-Premium Customers
2. Recent High-Premium Customers
3. Recent Low-Premium Customers
4. Recent Very High-Premium Customers
5. Inactive Medium-Premium Customers

### 10. Customer Insights

The customer segments are analyzed using characteristics such as:

- Premium Amount
- Recency
- Income Level
- Coverage Amount
- Purchase Timing

### 11. Marketing Recommendations

Different customer groups can be targeted using different strategies, such as:

- VIP retention
- Premium service offers
- Upselling
- Affordable upgrade offers
- Win-back campaigns
- Re-engagement campaigns
- Cross-selling

---

## Clustering Evaluation

The Elbow Method showed a gradual reduction in inertia as the number of clusters increased.

Silhouette analysis showed relatively weak cluster separation, with the highest tested silhouette score occurring at K = 2.

However, K = 5 was selected for the final analysis because it provided more meaningful and interpretable customer segments for exploratory and marketing analysis.

This indicates that the dataset contains considerable overlap between customer groups.

---

## Results

The K-Means model divides the **53,503 customers into five segments**.

The resulting segments show differences particularly in:

- Purchase recency
- Premium amount
- Purchase timing

These differences can be used to understand customer behavior and design more targeted marketing strategies.

---

## How to Run

### 1. Install the required libraries

Open a terminal inside the `Project 1` folder and run:

```bash
pip install -r requirements.txt