# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# LOAD DATASET

df = pd.read_excel("customer_segmentation_data.xlsx")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# BASIC DATASET INFORMATION

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# NUMERICAL DATA ANALYSIS

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)

print("\nNumerical Data Summary:")
print(df[numerical_columns].describe())


# CATEGORICAL DATA ANALYSIS

categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nCategorical Data Summary:")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts().head(10))


# CHECK PURCHASE HISTORY

print("\nPurchase History:")
print(df["Purchase History"].head(20).to_string())

print("\nPurchase History Data Type:")
print(df["Purchase History"].dtype)

print("\nSample Purchase History Values:")
print(df["Purchase History"].sample(20).to_string(index=False))


# CONVERT PURCHASE HISTORY TO DATETIME

df["Purchase History"] = pd.to_datetime(
    df["Purchase History"],
    errors="coerce"
)

print("\nAfter Date Conversion:")
print(df["Purchase History"].head(10))

print("\nData Type:")
print(df["Purchase History"].dtype)

print("\nInvalid Dates:")
print(df["Purchase History"].isnull().sum())


# CREATE DATE FEATURES

df["Purchase Year"] = df["Purchase History"].dt.year
df["Purchase Month"] = df["Purchase History"].dt.month
df["Purchase Day"] = df["Purchase History"].dt.day
df["Purchase Day of Week"] = df["Purchase History"].dt.dayofweek

print("\nNew Date Features:")

print(
    df[
        [
            "Purchase History",
            "Purchase Year",
            "Purchase Month",
            "Purchase Day",
            "Purchase Day of Week"
        ]
    ].head()
)


# CREATE CUSTOMER RECENCY

latest_purchase_date = df["Purchase History"].max()

df["Recency"] = (
    latest_purchase_date - df["Purchase History"]
).dt.days

print("\nLatest Purchase Date:")
print(latest_purchase_date)

print("\nRecency:")
print(
    df[
        [
            "Customer ID",
            "Purchase History",
            "Recency"
        ]
    ].head()
)


# FEATURE ENGINEERING

df["Premium Coverage Ratio"] = (
    df["Premium Amount"] / df["Coverage Amount"]
)


df["Purchase Quarter"] = (
    df["Purchase History"].dt.quarter
)


def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Autumn"


df["Purchase Season"] = (
    df["Purchase History"].dt.month.apply(get_season)
)


def get_age_group(age):

    if age < 30:
        return "Young"

    elif age < 50:
        return "Adult"

    elif age < 65:
        return "Mature"

    else:
        return "Senior"


df["Age Group"] = df["Age"].apply(get_age_group)


print("\n========== NEW ENGINEERED FEATURES ==========")

print("\nPremium Coverage Ratio:")
print(df["Premium Coverage Ratio"].head())

print("\nPurchase Quarter:")
print(df["Purchase Quarter"].head())

print("\nPurchase Season:")
print(df["Purchase Season"].value_counts())

print("\nAge Group:")
print(df["Age Group"].value_counts())


# SELECT FEATURES FOR CUSTOMER SEGMENTATION

segmentation_features = [
    "Age",
    "Income Level",
    "Coverage Amount",
    "Premium Amount",
    "Purchase Year",
    "Purchase Month",
    "Purchase Day of Week",
    "Recency"
]

X = df[segmentation_features]

print("\nFeatures Selected for Customer Segmentation:")
print(X.head())

print("\nShape of Feature Dataset:")
print(X.shape)


# FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Features:")
print(X_scaled[:5])

print("\nScaled Dataset Shape:")
print(X_scaled.shape)


# ELBOW METHOD

inertia = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


print("\nInertia Values:")

for k, value in zip(range(2, 11), inertia):

    print(
        f"K = {k} : Inertia = {value:.2f}"
    )


# ELBOW GRAPH

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xticks(range(2, 11))

plt.grid(True)

plt.show()


# SILHOUETTE SCORE

print("\nSilhouette Scores:")

for k in range(2, 7):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans_test.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K = {k} : Silhouette Score = {score:.3f}"
    )


# FINAL K-MEANS MODEL

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

print("\nK-Means Model Created Successfully!")


# ASSIGN CLUSTERS TO CUSTOMERS

df["Cluster"] = kmeans.labels_

print("\nCustomer Cluster Assignment:")

print(
    df[
        [
            "Customer ID",
            "Cluster"
        ]
    ].head(10)
)


# NUMBER OF CUSTOMERS IN EACH CLUSTER

print("\nNumber of Customers in Each Cluster:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)


# CLUSTER SUMMARY

cluster_summary = (
    df.groupby("Cluster")[segmentation_features]
    .mean()
)

print("\nCustomer Cluster Summary:")

print(
    cluster_summary.round(2)
)


# VISUALIZE CUSTOMER CLUSTERS

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Recency",
    y="Premium Amount",
    hue="Cluster",
    palette="Set2",
    alpha=0.6
)

plt.title(
    "Customer Segments: Premium Amount vs Recency"
)

plt.xlabel("Recency (Days)")
plt.ylabel("Premium Amount")

plt.legend(title="Cluster")

plt.show()


# COMPARE CLUSTER CENTERS

cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=segmentation_features
)

print("\nCluster Centers (Scaled Values):")

print(
    cluster_centers.round(2)
)


# FEATURE VARIATION ACROSS CLUSTERS

feature_variation = (
    cluster_centers.max()
    - cluster_centers.min()
).sort_values(ascending=False)

print("\nFeature Variation Across Clusters:")

print(
    feature_variation.round(2)
)


# COMPARE IMPORTANT CLUSTER FEATURES

comparison = (
    df.groupby("Cluster")[
        [
            "Premium Amount",
            "Recency",
            "Purchase Year",
            "Purchase Month",
            "Purchase Day of Week"
        ]
    ].mean()
)

print("\nImportant Cluster Comparison:")

print(
    comparison.round(2)
)


# NAME CUSTOMER SEGMENTS

cluster_names = {

    0: "Older Medium-Premium Customers",

    1: "Recent High-Premium Customers",

    2: "Recent Low-Premium Customers",

    3: "Recent Very High-Premium Customers",

    4: "Inactive Medium-Premium Customers"
}

df["Segment Name"] = (
    df["Cluster"].map(cluster_names)
)

print("\nCustomer Segment Names:")

print(
    df[
        [
            "Customer ID",
            "Cluster",
            "Segment Name"
        ]
    ].head(10)
)


# CUSTOMER INSIGHT SYSTEM
# CUSTOMER INSIGHT SYSTEM

print(
    "\n========== CUSTOMER INSIGHT SYSTEM =========="
)

while True:

    print("\nWhat would you like to check?")

    print("1. General Customer Analysis")
    print("2. Individual Customer Insights")
    print("3. Exit")

    choice = input(
        "\nEnter your choice (1/2/3): "
    )


    # ==========================================
    # GENERAL CUSTOMER ANALYSIS
    # ==========================================

    if choice == "1":

        print(
            "\n========== GENERAL CUSTOMER ANALYSIS =========="
        )

        print(
            "Total Customers:",
            len(df)
        )

        print(
            "Number of Customer Segments:",
            df["Cluster"].nunique()
        )

        print("\nCustomers in Each Segment:")

        print(
            df["Segment Name"].value_counts()
        )

        print("\nAverage Premium by Segment:")

        print(
            df.groupby("Segment Name")["Premium Amount"]
            .mean()
            .round(2)
        )

        print("\nAverage Recency by Segment:")

        print(
            df.groupby("Segment Name")["Recency"]
            .mean()
            .round(2)
        )

        premium_median = (
            df["Premium Amount"].median()
        )

        recency_median = (
            df["Recency"].median()
        )

        print(
            "\nOverall Median Premium:",
            round(premium_median, 2)
        )

        print(
            "Overall Median Recency:",
            round(recency_median, 2)
        )

        print(
            "\nReturning to main menu..."
        )


    # ==========================================
    # INDIVIDUAL CUSTOMER ANALYSIS
    # ==========================================

    elif choice == "2":

        print(
            "\n========== INDIVIDUAL CUSTOMER INSIGHTS =========="
        )

        try:

            customer_id = int(
                input(
                    "\nEnter Customer ID to view individual insights: "
                )
            )

        except ValueError:

            print(
                "\nInvalid Customer ID. Please enter a number."
            )

            continue


        customer = df[
            df["Customer ID"] == customer_id
        ]


        if customer.empty:

            print(
                "\nCustomer ID not found."
            )

        else:

            customer = customer.iloc[0]


            print(
                "\n---------- CUSTOMER DETAILS ----------"
            )

            print(
                "Customer ID:",
                customer["Customer ID"]
            )

            print(
                "Age:",
                customer["Age"]
            )

            print(
                "Income Level:",
                customer["Income Level"]
            )

            print(
                "Coverage Amount:",
                customer["Coverage Amount"]
            )

            print(
                "Premium Amount:",
                customer["Premium Amount"]
            )


            print(
                "\n---------- PURCHASE INFORMATION ----------"
            )

            print(
                "Purchase Date:",
                customer["Purchase History"]
            )

            print(
                "Recency:",
                customer["Recency"],
                "days"
            )


            print(
                "\n---------- CUSTOMER SEGMENT ----------"
            )

            print(
                "Cluster:",
                customer["Cluster"]
            )

            print(
                "Segment:",
                customer["Segment Name"]
            )


            # ==========================================
            # AUTOMATIC RECOMMENDATION
            # ==========================================

            premium_median = (
                df["Premium Amount"].median()
            )

            recency_median = (
                df["Recency"].median()
            )


            if (
                customer["Recency"] <= recency_median
                and
                customer["Premium Amount"] >= premium_median
            ):

                strategy = (
                    "VIP retention and premium service offers"
                )


            elif (
                customer["Recency"] <= recency_median
                and
                customer["Premium Amount"] < premium_median
            ):

                strategy = (
                    "Upselling and affordable upgrade offers"
                )


            elif (
                customer["Recency"] > recency_median
                and
                customer["Premium Amount"] >= premium_median
            ):

                strategy = (
                    "Win-back campaigns and renewal offers"
                )


            else:

                strategy = (
                    "Re-engagement and cross-selling campaigns"
                )


            print(
                "\n---------- RECOMMENDED MARKETING STRATEGY ----------"
            )

            print(strategy)


        print(
            "\nReturning to main menu..."
        )


    # ==========================================
    # EXIT
    # ==========================================

    elif choice == "3":

        print(
            "\nThank you for using the Customer Insight System!"
        )

        print(
            "Exiting program..."
        )

        break


    # ==========================================
    # INVALID CHOICE
    # ==========================================

    else:

        print(
            "\nInvalid choice."
        )

        print(
            "Please enter 1, 2, or 3."
        )