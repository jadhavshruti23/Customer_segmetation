# Customer Segmentation Projects

This repository contains two separate implementations of a customer segmentation system developed using Python, machine learning and Streamlit.

The projects demonstrate the progression from a dataset-specific machine learning implementation to an automated and reusable application.

---

## Project 1 - Customer Segmentation using Python

This project is a dataset-specific Python and machine learning implementation.

It focuses on implementing the complete customer segmentation workflow using a fixed customer dataset.

### Main Components

- Data loading
- Data inspection
- Data cleaning
- Feature engineering
- Feature selection
- Feature scaling
- Elbow Method
- Silhouette Analysis
- K-Means clustering
- Customer segmentation
- Customer insights
- Marketing recommendations

### Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenPyXL

[View Project 1](./Project%201)

---

## Project 2 - Automated Customer Segmentation using Streamlit

This project is an automated and reusable Streamlit application.

Instead of using one fixed dataset, users can upload their own CSV or Excel dataset.

The application automatically performs the major steps of the customer segmentation workflow.

### Main Components

- Dataset upload
- Automatic dataset detection
- Data cleaning
- Feature engineering
- Automatic feature selection
- Feature scaling
- Elbow Method
- Silhouette Analysis
- K-Means clustering
- Customer segmentation
- Automated insights
- Marketing recommendations
- Downloadable results

### Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- OpenPyXL

[View Project 2](./Project%202)

---

## Project Comparison

| Feature | Project 1 | Project 2 |
|---|---|---|
| Implementation | Python script | Streamlit application |
| Dataset | Pre-existing | User uploaded |
| Data Cleaning | Dataset-specific | Automated |
| Feature Engineering | Dataset-specific | Automated |
| Feature Selection | Selected features | Automatic |
| Feature Scaling | Yes | Yes |
| K-Means Clustering | Yes | Yes |
| Elbow Method | Yes | Yes |
| Silhouette Analysis | Yes | Yes |
| Customer Segmentation | Yes | Yes |
| Customer Insights | Yes | Yes |
| Marketing Recommendations | Yes | Yes |
| Download Results | No | Yes |

---

## Overall Workflow

```text
Customer Dataset
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Feature Selection
       |
       v
Feature Scaling
       |
       v
K-Means Clustering
       |
       v
Customer Segments
       |
       v
Customer Insights
       |
       v
Marketing Recommendations