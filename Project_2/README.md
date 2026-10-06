# Project 2 - Automated Customer Segmentation using Streamlit

## Overview

This project is an automated customer segmentation application built using Python and Streamlit.

Unlike the dataset-specific Python implementation, this version allows users to upload their own CSV or Excel dataset.

The application automatically performs data cleaning, feature engineering, feature selection, scaling, K-Means clustering and customer segmentation.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- OpenPyXL

## Key Features

- CSV and Excel dataset upload
- Automatic dataset inspection
- Automatic data cleaning
- Automatic date detection
- Automatic feature engineering
- Automatic feature selection
- Feature scaling
- Elbow Method
- Silhouette Analysis
- K-Means clustering
- Customer segmentation
- Customer insights
- Marketing recommendations
- Downloadable results

## Application Workflow

Upload Dataset
        ↓
Dataset Detection
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Automatic Feature Selection
        ↓
Feature Scaling
        ↓
Find Suitable K
        ↓
K-Means Clustering
        ↓
Customer Segmentation
        ↓
Customer Insights
        ↓
Marketing Recommendations
        ↓
Download Results

## How It Works

### 1. Upload Dataset

Users can upload a CSV or Excel dataset through the Streamlit interface.

### 2. Automatic Dataset Detection

The application examines the uploaded dataset and identifies suitable columns for analysis.

### 3. Data Cleaning

The application handles common data-quality issues such as:

- Missing numerical values
- Missing categorical values
- Duplicate records
- Infinite numerical values

### 4. Feature Engineering

The application automatically derives useful features from detected date columns where possible.

### 5. Automatic Feature Selection

The application selects suitable numerical features for customer segmentation while excluding identifier-like columns and unsuitable variables.

### 6. Feature Scaling

Selected features are standardized before clustering.

### 7. Cluster Selection

The application evaluates multiple K values using:

- Elbow Method
- Silhouette Score

### 8. Customer Segmentation

K-Means clustering is applied to divide customers into groups with similar characteristics.

### 9. Insights and Recommendations

The application generates customer-segment insights and marketing recommendations based on the resulting groups.

### 10. Download Results

Users can download the analyzed dataset containing the generated customer segments.

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt