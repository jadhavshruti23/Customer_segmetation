# AUTOMATED CUSTOMER ANALYTICS & SEGMENTATION SYSTEM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Automated Customer Analytics System",
    layout="wide"
)

# PROFESSIONAL DASHBOARD STYLING
st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

[data-testid="stAppViewContainer"] {
    background-color: #f5f7fb;
}

[data-testid="stMainBlockContainer"] {
    background-color: #f5f7fb;
}


/* ============================================================
   MAIN TEXT
   ============================================================ */

[data-testid="stMarkdownContainer"] p {
    color: #344054 !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: #1d2939 !important;
}

[data-testid="stMarkdownContainer"] li {
    color: #344054 !important;
}

[data-testid="stMarkdownContainer"] a {
    color: #175cd3 !important;
}


/* ============================================================
   HEADINGS
   ============================================================ */

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #1d2939 !important;
}

[data-testid="stHeader"] {
    color: #1d2939 !important;
}

.stTitle {
    color: #1d2939 !important;
}


/* ============================================================
   CUSTOM TITLE
   ============================================================ */

.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
    color: #1d2939 !important;
}

.subtitle {
    font-size: 16px;
    color: #667085 !important;
    margin-bottom: 25px;
}

.main-content-text {
    color: #344054 !important;
}

.workflow-title {
    color: #1d2939 !important;
    font-size: 24px;
    font-weight: 700;
}

.workflow-step {
    color: #344054 !important;
    font-size: 16px;
    margin: 8px 0;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

[data-testid="stMetricLabel"] {
    color: #667085 !important;
}

[data-testid="stMetricLabel"] p {
    color: #667085 !important;
}

[data-testid="stMetricValue"] {
    color: #1d2939 !important;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #667085 !important;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #1d2939 !important;
}


/* ============================================================
   SECTION CARDS
   ============================================================ */

.section-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}


/* ============================================================
   SUCCESS / INFO / WARNING
   ============================================================ */

[data-testid="stAlert"] {
    color: #344054 !important;
}

[data-testid="stAlert"] p {
    color: #344054 !important;
}

.success-box {
    background: #ecfdf3;
    border: 1px solid #abefc6;
    color: #067647 !important;
    padding: 14px;
    border-radius: 10px;
    margin: 15px 0;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    color: #344054 !important;
}


/* ============================================================
   INPUT LABELS
   ============================================================ */

[data-testid="stWidgetLabel"] {
    color: #344054 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #344054 !important;
}


/* ============================================================
   RADIO BUTTONS
   ============================================================ */

[data-testid="stRadio"] label {
    color: #344054 !important;
}

[data-testid="stRadio"] label p {
    color: #344054 !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    color: #344054 !important;
}

[data-testid="stFileUploader"] label {
    color: #344054 !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"]
[data-testid="stRadio"]
label {
    color: white !important;
}

section[data-testid="stSidebar"]
[data-testid="stRadio"]
label p {
    color: white !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p {
    color: white !important;
}


/* ============================================================
   PROGRESS TEXT
   ============================================================ */

.progress-text {
    font-size: 14px;
    font-weight: 600;
    margin-top: 8px;
    color: #344054 !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton button {
    color: white !important;
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: #e5e7eb !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">Customer Analytics & Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Automated customer intelligence, segmentation and analytics system'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROCESSING STATUS
# ============================================================

progress_bar = st.progress(0)

status_text = st.empty()

status_text.info(
    "Initializing automated analysis..."
)


# ============================================================
# HELPER FUNCTION
# DATE DETECTION
# ============================================================

def detect_date_columns(data):

    """
    Automatically detect date columns.

    Only a sample of rows is checked first
    to improve processing speed.
    """

    date_columns = []

    sample_size = min(
        1000,
        len(data)
    )

    if len(data) > sample_size:

        sample = data.sample(
            n=sample_size,
            random_state=42
        )

    else:

        sample = data

    for column in data.columns:

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(
            data[column]
        ):

            date_columns.append(column)

            continue

        # Only inspect object/string columns
        if data[column].dtype != "object":
            continue

        try:

            converted_sample = pd.to_datetime(
                sample[column],
                errors="coerce",
                format="mixed"
            )

            valid_ratio = (
                converted_sample.notna().mean()
            )

            if valid_ratio >= 0.70:

                data[column] = pd.to_datetime(
                    data[column],
                    errors="coerce",
                    format="mixed"
                )

                date_columns.append(column)

        except Exception:

            continue

    return date_columns


# ============================================================
# HELPER FUNCTION
# ID DETECTION
# ============================================================

def detect_id_columns(data):

    """
    Automatically detect identifier-like columns.
    """

    id_columns = []

    data_length = len(data)

    if data_length == 0:
        return id_columns

    for column in data.columns:

        name = column.lower()

        unique_ratio = (
            data[column].nunique(
                dropna=False
            )
            / data_length
        )

        if (
            "id" in name
            or "identifier" in name
            or "account number" in name
            or "customer number" in name
            or "user number" in name
        ):

            id_columns.append(column)

        elif unique_ratio >= 0.95:

            id_columns.append(column)

    return list(
        dict.fromkeys(id_columns)
    )


# ============================================================
# HELPER FUNCTION
# DATA CLEANING
# ============================================================

def clean_dataset(data):

    """
    Automatically clean the dataset.
    """

    cleaned = data.copy()

    duplicates_before = int(
        cleaned.duplicated().sum()
    )

    missing_before = int(
        cleaned.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    cleaned = cleaned.drop_duplicates()

    # --------------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------------

    for column in cleaned.columns:

        if pd.api.types.is_numeric_dtype(
            cleaned[column]
        ):

            median_value = (
                cleaned[column].median()
            )

            cleaned[column] = (
                cleaned[column]
                .fillna(median_value)
            )

        else:

            mode_values = (
                cleaned[column].mode()
            )

            if not mode_values.empty:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
                        mode_values.iloc[0]
                    )
                )

            else:

                cleaned[column] = (
                    cleaned[column]
                    .fillna("Unknown")
                )

    duplicates_after = int(
        cleaned.duplicated().sum()
    )

    missing_after = int(
        cleaned.isnull().sum().sum()
    )

    cleaning_report = {

        "Missing Values Before":
            missing_before,

        "Missing Values After":
            missing_after,

        "Duplicates Before":
            duplicates_before,

        "Duplicates After":
            duplicates_after
    }

    return (
        cleaned,
        cleaning_report
    )


# ============================================================
# HELPER FUNCTION
# FEATURE ENGINEERING
# ============================================================

def engineer_features(
    data,
    date_columns
):

    """
    Automatically create useful features.
    """

    engineered = data.copy()

    created_features = []

    # ========================================================
    # DATE FEATURES
    # ========================================================

    for date_column in date_columns:

        safe_name = (
            date_column
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        # ----------------------------------------------------
        # YEAR
        # ----------------------------------------------------

        year_column = (
            f"{safe_name}_Year"
        )

        engineered[year_column] = (
            engineered[date_column].dt.year
        )

        created_features.append(
            year_column
        )

        # ----------------------------------------------------
        # MONTH
        # ----------------------------------------------------

        month_column = (
            f"{safe_name}_Month"
        )

        engineered[month_column] = (
            engineered[date_column].dt.month
        )

        created_features.append(
            month_column
        )

        # ----------------------------------------------------
        # DAY OF WEEK
        # ----------------------------------------------------

        weekday_column = (
            f"{safe_name}_Day_of_Week"
        )

        engineered[weekday_column] = (
            engineered[date_column]
            .dt.dayofweek
        )

        created_features.append(
            weekday_column
        )

        # ----------------------------------------------------
        # QUARTER
        # ----------------------------------------------------

        quarter_column = (
            f"{safe_name}_Quarter"
        )

        engineered[quarter_column] = (
            engineered[date_column]
            .dt.quarter
        )

        created_features.append(
            quarter_column
        )

        # ----------------------------------------------------
        # SEASON
        # ----------------------------------------------------

        season_column = (
            f"{safe_name}_Season"
        )

        month_values = (
            engineered[date_column]
            .dt.month
        )

        engineered[season_column] = np.select(
            [
                month_values.isin([12, 1, 2]),
                month_values.isin([3, 4, 5]),
                month_values.isin([6, 7, 8])
            ],
            [
                "Winter",
                "Spring",
                "Summer"
            ],
            default="Autumn"
        )

        created_features.append(
            season_column
        )

        # ----------------------------------------------------
        # RECENCY
        # ----------------------------------------------------

        latest_date = (
            engineered[date_column].max()
        )

        if pd.notna(latest_date):

            recency_column = (
                f"{safe_name}_Recency"
            )

            engineered[recency_column] = (
                latest_date
                - engineered[date_column]
            ).dt.days

            created_features.append(
                recency_column
            )

    # ========================================================
    # AGE GROUP
    # ========================================================

    age_columns = []

    for column in engineered.columns:

        if "age" not in column.lower():
            continue

        if pd.api.types.is_numeric_dtype(
            engineered[column]
        ):

            age_columns.append(column)

    for age_column in age_columns:

        group_column = (
            f"{age_column}_Group"
        )

        age_values = (
            engineered[age_column]
        )

        engineered[group_column] = np.select(
            [
                age_values < 30,
                age_values < 50,
                age_values < 65
            ],
            [
                "Young",
                "Adult",
                "Mature"
            ],
            default="Senior"
        )

        engineered.loc[
            age_values.isna(),
            group_column
        ] = "Unknown"

        created_features.append(
            group_column
        )

    # ========================================================
    # AUTOMATIC NUMERIC RATIOS
    # ========================================================

    numeric_columns = (
        engineered
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    amount_columns = [
        column
        for column in numeric_columns
        if any(
            word in column.lower()
            for word in [
                "amount",
                "price",
                "cost",
                "income",
                "revenue",
                "sales",
                "value"
            ]
        )
    ]

    # --------------------------------------------------------
    # LIMIT RATIOS
    # --------------------------------------------------------

    ratio_pairs = []

    if len(amount_columns) >= 2:

        ratio_pairs.append(
            (
                amount_columns[0],
                amount_columns[1]
            )
        )

    if len(amount_columns) >= 4:

        ratio_pairs.append(
            (
                amount_columns[2],
                amount_columns[3]
            )
        )

    # --------------------------------------------------------
    # CREATE RATIOS
    # --------------------------------------------------------

    for numerator, denominator in ratio_pairs:

        ratio_column = (
            f"{numerator}_to_{denominator}_Ratio"
        )

        denominator_values = (
            engineered[denominator]
        )

        engineered[ratio_column] = np.where(
            denominator_values != 0,
            engineered[numerator]
            / denominator_values,
            np.nan
        )

        created_features.append(
            ratio_column
        )

    return (
        engineered,
        created_features
    )


# ============================================================
# HELPER FUNCTION
# FEATURE SELECTION
# ============================================================

def select_clustering_features(
    data,
    id_columns,
    date_columns
):

    """
    Automatically select suitable numeric
    features for K-Means clustering.
    """

    numeric_columns = (
        data
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    selected = []

    data_length = len(data)

    if data_length == 0:
        return selected

    for column in numeric_columns:

        # ----------------------------------------------------
        # EXCLUDE DETECTED IDS
        # ----------------------------------------------------

        if column in id_columns:
            continue

        name = column.lower()

        # ----------------------------------------------------
        # EXCLUDE IDENTIFIER-LIKE COLUMNS
        # ----------------------------------------------------

        if (
            "id" in name
            or "identifier" in name
        ):

            continue

        # ----------------------------------------------------
        # EXCLUDE CONSTANT COLUMNS
        # ----------------------------------------------------

        if data[column].nunique(
            dropna=True
        ) <= 1:

            continue

        # ----------------------------------------------------
        # EXCLUDE HIGH-CARDINALITY IDS
        # ----------------------------------------------------

        unique_ratio = (
            data[column].nunique(
                dropna=True
            )
            / data_length
        )

        if unique_ratio >= 0.98:
            continue

        selected.append(column)

    return selected


# ============================================================
# HELPER FUNCTION
# FIND BEST K
# ============================================================

def find_best_k(
    X_scaled
):

    """
    Automatically evaluate K values using
    sampled inertia and silhouette scores.

    Optimized for faster execution on
    large datasets.
    """

    max_k = min(
        8,
        len(X_scaled) - 1
    )

    if max_k < 2:

        return (
            2,
            [],
            [],
            []
        )

    k_values = list(
        range(
            2,
            max_k + 1
        )
    )

    # --------------------------------------------------------
    # SAMPLE DATA
    # --------------------------------------------------------

    evaluation_sample_size = min(
        5000,
        len(X_scaled)
    )

    silhouette_sample_size = min(
        1000,
        evaluation_sample_size
    )

    rng = np.random.RandomState(
        42
    )

    sample_indices = rng.choice(
        len(X_scaled),
        size=evaluation_sample_size,
        replace=False
    )

    X_sample = (
        X_scaled[
            sample_indices
        ]
        .astype(np.float32)
    )

    inertia_values = []

    silhouette_values = []

    # --------------------------------------------------------
    # TEST K VALUES
    # --------------------------------------------------------

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,
            max_iter=150
        )

        labels = (
            model.fit_predict(
                X_sample
            )
        )

        inertia_values.append(
            model.inertia_
        )

        score = silhouette_score(
            X_sample,
            labels,
            sample_size=silhouette_sample_size,
            random_state=42
        )

        silhouette_values.append(
            score
        )

    # --------------------------------------------------------
    # BEST K
    # --------------------------------------------------------

    best_index = int(
        np.argmax(
            silhouette_values
        )
    )

    best_k = k_values[
        best_index
    ]

    return (
        best_k,
        k_values,
        inertia_values,
        silhouette_values
    )


# ============================================================
# HELPER FUNCTION
# SEGMENT NAMES
# ============================================================

def generate_segment_names(
    data,
    cluster_column,
    numeric_features
):

    """
    Automatically generate meaningful
    cluster names.
    """

    summary = (
        data
        .groupby(
            cluster_column
        )[numeric_features]
        .mean()
    )

    recency_column = None
    value_column = None

    # --------------------------------------------------------
    # DETECT RECENCY COLUMN
    # --------------------------------------------------------

    for column in numeric_features:

        name = column.lower()

        if (
            "recency" in name
            and recency_column is None
        ):

            recency_column = column

    # --------------------------------------------------------
    # DETECT VALUE COLUMN
    # --------------------------------------------------------

    for column in numeric_features:

        name = column.lower()

        if any(
            word in name
            for word in [
                "premium",
                "revenue",
                "sales",
                "income",
                "amount",
                "value"
            ]
        ):

            if value_column is None:

                value_column = column

    # --------------------------------------------------------
    # MEDIAN VALUES
    # --------------------------------------------------------

    if recency_column:

        recency_median = (
            summary[
                recency_column
            ].median()
        )

    else:

        recency_median = None

    if value_column:

        value_median = (
            summary[
                value_column
            ].median()
        )

    else:

        value_median = None

    names = {}

    # --------------------------------------------------------
    # CREATE NAMES
    # --------------------------------------------------------

    for cluster in summary.index:

        labels = []

        # Recency
        if recency_column:

            recency = summary.loc[
                cluster,
                recency_column
            ]

            if recency <= recency_median:

                labels.append(
                    "Recent"
                )

            else:

                labels.append(
                    "Inactive"
                )

        # Value
        if value_column:

            value = summary.loc[
                cluster,
                value_column
            ]

            if value >= value_median:

                labels.append(
                    "High-Value"
                )

            else:

                labels.append(
                    "Standard-Value"
                )

        if not labels:

            labels.append(
                f"Segment {cluster}"
            )

        names[cluster] = (
            " ".join(labels)
        )

    return (
        names,
        summary
    )


# ============================================================
# CACHED DATASET LOADER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5
)
def load_dataset(
    file_bytes,
    file_name
):

    """
    Cached dataset loader.
    """

    if file_name.lower().endswith(
        ".csv"
    ):

        return pd.read_csv(
            BytesIO(file_bytes)
        )

    else:

        return pd.read_excel(
            BytesIO(file_bytes)
        )


# ============================================================
# CACHED ANALYSIS ENGINE
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3
)
def run_analysis_cached(
    file_bytes,
    file_name
):

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_dataset(
        file_bytes,
        file_name
    )

    if data.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )

    original_data = data.copy()

    # ========================================================
    # DATASET DETECTION
    # ========================================================

    date_columns = (
        detect_date_columns(
            data
        )
    )

    id_columns = (
        detect_id_columns(
            data
        )
    )

    numeric_columns = (
        data
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        data
        .select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    # ========================================================
    # DATA CLEANING
    # ========================================================

    cleaned_data, cleaning_report = (
        clean_dataset(
            data
        )
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    engineered_data, engineered_features = (
        engineer_features(
            cleaned_data,
            date_columns
        )
    )

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    segmentation_features = (
        select_clustering_features(
            engineered_data,
            id_columns,
            date_columns
        )
    )

    if len(
        segmentation_features
    ) < 2:

        raise ValueError(
            "The dataset does not contain enough "
            "numeric features for segmentation."
        )

    # ========================================================
    # PREPARE K-MEANS DATA
    # ========================================================

    X = engineered_data[
        segmentation_features
    ].copy()

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Fill missing values
    X = X.fillna(
        X.median()
    )

    # Remove completely invalid columns
    X = X.dropna(
        axis=1,
        how="all"
    )

    segmentation_features = (
        X.columns.tolist()
    )

    if len(
        segmentation_features
    ) < 2:

        raise ValueError(
            "Not enough valid numeric features "
            "remain after cleaning."
        )

    # Convert to float32
    X = X.astype(
        np.float32
    )

    # ========================================================
    # SCALING
    # ========================================================

    scaler = StandardScaler()

    X_scaled = (
        scaler
        .fit_transform(X)
        .astype(np.float32)
    )

    # ========================================================
    # FIND BEST K
    # ========================================================

    (
        best_k,
        k_values,
        inertia_values,
        silhouette_values
    ) = find_best_k(
        X_scaled
    )

    # ========================================================
    # FINAL K-MEANS
    # ========================================================

    training_sample_size = min(
        10000,
        len(X_scaled)
    )

    if (
        len(X_scaled)
        > training_sample_size
    ):

        rng = np.random.RandomState(
            42
        )

        training_indices = (
            rng.choice(
                len(X_scaled),
                size=training_sample_size,
                replace=False
            )
        )

        X_train = (
            X_scaled[
                training_indices
            ]
        )

    else:

        X_train = X_scaled

    X_train = X_train.astype(
        np.float32
    )

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=5,
        max_iter=200
    )

    kmeans.fit(
        X_train
    )

    # Predict every customer
    engineered_data["Cluster"] = (
        kmeans.predict(
            X_scaled
        )
    )

    # ========================================================
    # SEGMENT NAMES
    # ========================================================

    (
        segment_names,
        cluster_summary
    ) = generate_segment_names(
        engineered_data,
        "Cluster",
        segmentation_features
    )

    engineered_data[
        "Segment Name"
    ] = (
        engineered_data[
            "Cluster"
        ].map(
            segment_names
        )
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "original_df":
            original_data,

        "cleaned_df":
            cleaned_data,

        "engineered_df":
            engineered_data,

        "cleaning_report":
            cleaning_report,

        "date_columns":
            date_columns,

        "id_columns":
            id_columns,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "engineered_features":
            engineered_features,

        "segmentation_features":
            segmentation_features,

        "best_k":
            best_k,

        "k_values":
            k_values,

        "inertia_values":
            inertia_values,

        "silhouette_values":
            silhouette_values,

        "segment_names":
            segment_names,

        "cluster_summary":
            cluster_summary
    }


# ============================================================
# SIDEBAR - DATASET UPLOAD
# ============================================================

st.sidebar.header(
    "Dataset"
)

uploaded_file = (
    st.sidebar.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "xlsx"
        ]
    )
)


# ============================================================
# NO FILE UPLOADED
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        Automated Workflow

        1. Upload dataset
        2. Detect dataset structure
        3. Clean data
        4. Engineer features
        5. Select useful features
        6. Find optimal number of clusters
        7. Perform customer segmentation
        8. Generate automatic insights
        9. Generate recommendations
        10. Download results
        """
    )

    st.stop()


# ============================================================
# RUN AUTOMATED ANALYSIS
# ============================================================

try:

    file_bytes = (
        uploaded_file.getvalue()
    )

    results = (
        run_analysis_cached(
            file_bytes,
            uploaded_file.name
        )
    )

    progress_bar.progress(
        100
    )

    status_text.success(
        "Automated analysis completed."
    )

except Exception as error:

    st.error(
        f"Unable to analyze the dataset: {error}"
    )

    st.stop()


# ============================================================
# GET ANALYSIS RESULTS
# ============================================================

original_df = (
    results["original_df"]
)

cleaned_df = (
    results["cleaned_df"]
)

engineered_df = (
    results["engineered_df"]
)

cleaning_report = (
    results["cleaning_report"]
)

date_columns = (
    results["date_columns"]
)

id_columns = (
    results["id_columns"]
)

numeric_columns = (
    results["numeric_columns"]
)

categorical_columns = (
    results["categorical_columns"]
)

engineered_features = (
    results["engineered_features"]
)

segmentation_features = (
    results["segmentation_features"]
)

best_k = (
    results["best_k"]
)

k_values = (
    results["k_values"]
)

inertia_values = (
    results["inertia_values"]
)

silhouette_values = (
    results["silhouette_values"]
)

segment_names = (
    results["segment_names"]
)

cluster_summary = (
    results["cluster_summary"]
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

analysis_option = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Overview",
        "Data Cleaning",
        "Feature Engineering",
        "Automatic Feature Selection",
        "Customer Segmentation",
        "Customer Insights",
        "Marketing Recommendations",
        "Individual Customer",
        "Download Results"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if analysis_option == "Dashboard":

    st.header(
        "Automated Customer Analytics Dashboard"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Rows",
        f"{len(engineered_df):,}"
    )

    col2.metric(
        "Columns",
        len(original_df.columns)
    )

    col3.metric(
        "Detected Clusters",
        best_k
    )

    col4.metric(
        "Engineered Features",
        len(engineered_features)
    )

    st.divider()

    st.subheader(
        "Automatically Detected Dataset Structure"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Numeric Columns",
        len(numeric_columns)
    )

    col2.metric(
        "Categorical Columns",
        len(categorical_columns)
    )

    col3.metric(
        "Date Columns",
        len(date_columns)
    )

    st.divider()

    st.subheader(
        "Automated System Workflow"
    )

    st.success(
        "Dataset detected -> Cleaned -> "
        "Features engineered -> Features selected -> "
        "Best K identified -> Customers segmented -> "
        "Insights generated"
    )

    st.subheader(
        "Customer Segment Distribution"
    )

    segment_counts = (
        engineered_df[
            "Segment Name"
        ]
        .value_counts()
    )

    st.bar_chart(
        segment_counts
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif analysis_option == "Dataset Overview":

    st.header(
        "Dataset Overview"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Rows",
        f"{len(original_df):,}"
    )

    col2.metric(
        "Columns",
        len(original_df.columns)
    )

    col3.metric(
        "Numeric",
        len(numeric_columns)
    )

    col4.metric(
        "Categorical",
        len(categorical_columns)
    )

    st.subheader(
        "Column Information"
    )

    column_info = pd.DataFrame({

        "Column":
            original_df.columns,

        "Data Type": [
            str(
                original_df[
                    col
                ].dtype
            )
            for col in original_df.columns
        ],

        "Unique Values": [
            original_df[
                col
            ].nunique()
            for col in original_df.columns
        ],

        "Missing Values": [
            original_df[
                col
            ].isna().sum()
            for col in original_df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        original_df.head(10),
        use_container_width=True
    )


# ============================================================
# DATA CLEANING
# ============================================================

elif analysis_option == "Data Cleaning":

    st.header(
        "Automatic Data Cleaning"
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Missing Before",
        cleaning_report[
            "Missing Values Before"
        ]
    )

    col2.metric(
        "Missing After",
        cleaning_report[
            "Missing Values After"
        ]
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Duplicates Before",
        cleaning_report[
            "Duplicates Before"
        ]
    )

    col2.metric(
        "Duplicates After",
        cleaning_report[
            "Duplicates After"
        ]
    )

    st.success(
        "Automatic cleaning completed."
    )

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

elif analysis_option == "Feature Engineering":

    st.header(
        "Automatic Feature Engineering"
    )

    st.success(
        f"{len(engineered_features)} "
        "features were automatically created."
    )

    if date_columns:

        st.subheader(
            "Detected Date Columns"
        )

        st.write(
            date_columns
        )

    else:

        st.info(
            "No date columns were detected."
        )

    st.subheader(
        "Automatically Created Features"
    )

    if engineered_features:

        st.dataframe(
            pd.DataFrame({
                "Engineered Feature":
                    engineered_features
            }),
            use_container_width=True
        )

        st.subheader(
            "Preview"
        )

        st.dataframe(
            engineered_df[
                engineered_features
            ].head(10),
            use_container_width=True
        )

    else:

        st.info(
            "No additional features were created."
        )


# ============================================================
# FEATURE SELECTION
# ============================================================

elif analysis_option == "Automatic Feature Selection":

    st.header(
        "Automatic Feature Selection"
    )

    st.write(
        "The system automatically selects suitable "
        "numeric features for K-Means clustering."
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Available Numeric Features",
        len(numeric_columns)
    )

    col2.metric(
        "Selected Features",
        len(segmentation_features)
    )

    st.subheader(
        "Selected Features"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature":
                segmentation_features
        }),
        use_container_width=True
    )

    if id_columns:

        st.subheader(
            "Automatically Detected ID Columns"
        )

        st.write(
            id_columns
        )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif analysis_option == "Customer Segmentation":

    st.header(
        "Automatic Customer Segmentation"
    )

    col1, col2 = (
        st.columns(2)
    )

    col1.metric(
        "Automatically Selected K",
        best_k
    )

    col2.metric(
        "Number of Customers",
        f"{len(engineered_df):,}"
    )

    # --------------------------------------------------------
    # ELBOW GRAPH
    # --------------------------------------------------------

    st.subheader(
        "Elbow Method"
    )

    fig1, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    ax1.plot(
        k_values,
        inertia_values,
        marker="o"
    )

    ax1.set_xlabel(
        "Number of Clusters (K)"
    )

    ax1.set_ylabel(
        "Inertia"
    )

    ax1.set_title(
        "Elbow Method"
    )

    ax1.grid(True)

    st.pyplot(
        fig1,
        clear_figure=True
    )

    plt.close(fig1)

    # --------------------------------------------------------
    # SILHOUETTE
    # --------------------------------------------------------

    st.subheader(
        "Silhouette Scores"
    )

    silhouette_df = pd.DataFrame({

        "K":
            k_values,

        "Silhouette Score":
            silhouette_values
    })

    st.dataframe(
        silhouette_df.round(3),
        use_container_width=True
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    ax2.plot(
        k_values,
        silhouette_values,
        marker="o"
    )

    ax2.set_xlabel(
        "Number of Clusters (K)"
    )

    ax2.set_ylabel(
        "Silhouette Score"
    )

    ax2.set_title(
        "Silhouette Analysis"
    )

    ax2.grid(True)

    st.pyplot(
        fig2,
        clear_figure=True
    )

    plt.close(fig2)

    # --------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Customer Segments"
    )

    cluster_counts = (
        engineered_df[
            "Segment Name"
        ]
        .value_counts()
    )

    st.bar_chart(
        cluster_counts
    )

    # --------------------------------------------------------
    # CLUSTER SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Cluster Summary"
    )

    st.dataframe(
        cluster_summary.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    if len(
        segmentation_features
    ) >= 2:

        x_feature = (
            segmentation_features[0]
        )

        y_feature = (
            segmentation_features[1]
        )

        st.subheader(
            f"{x_feature} vs {y_feature}"
        )

        plot_data = (
            engineered_df[
                [
                    x_feature,
                    y_feature,
                    "Segment Name"
                ]
            ]
            .dropna()
        )

        # Limit visualization to 5,000 points
        if len(plot_data) > 5000:

            plot_data = (
                plot_data.sample(
                    5000,
                    random_state=42
                )
            )

        fig3, ax3 = plt.subplots(
            figsize=(9, 6)
        )

        sns.scatterplot(
            data=plot_data,
            x=x_feature,
            y=y_feature,
            hue="Segment Name",
            alpha=0.6,
            ax=ax3
        )

        ax3.set_title(
            "Automatically Generated Customer Segments"
        )

        st.pyplot(
            fig3,
            clear_figure=True
        )

        plt.close(fig3)


# ============================================================
# CUSTOMER INSIGHTS
# ============================================================

elif analysis_option == "Customer Insights":

    st.header(
        "Automatic Customer Insights"
    )

    st.subheader(
        "Segment Size"
    )

    segment_size = (
        engineered_df
        .groupby(
            "Segment Name"
        )
        .size()
        .reset_index(
            name="Customers"
        )
        .sort_values(
            "Customers",
            ascending=False
        )
    )

    st.dataframe(
        segment_size,
        use_container_width=True
    )

    st.subheader(
        "Segment Profiles"
    )

    st.dataframe(
        cluster_summary.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # LARGEST SEGMENT
    # --------------------------------------------------------

    largest_segment = (
        segment_size.iloc[0]
    )

    st.info(
        f"**Largest segment:** "
        f"{largest_segment['Segment Name']} "
        f"with "
        f"{largest_segment['Customers']:,} "
        f"customers."
    )

    # --------------------------------------------------------
    # FEATURE VARIATION
    # --------------------------------------------------------

    st.subheader(
        "Features Showing Most Variation Between Segments"
    )

    feature_variation = (
        cluster_summary.max()
        - cluster_summary.min()
    ).sort_values(
        ascending=False
    )

    st.dataframe(
        feature_variation
        .round(2)
        .rename(
            "Variation"
        )
        .to_frame(),
        use_container_width=True
    )


# ============================================================
# MARKETING RECOMMENDATIONS
# ============================================================

elif analysis_option == "Marketing Recommendations":

    st.header(
        "Automatic Marketing Recommendations"
    )

    st.write(
        "Recommendations are generated automatically "
        "from the characteristics of each customer segment."
    )

    # --------------------------------------------------------
    # DETECT VALUE COLUMN
    # --------------------------------------------------------

    value_column = None

    for column in segmentation_features:

        if any(
            word in column.lower()
            for word in [
                "premium",
                "revenue",
                "sales",
                "income",
                "amount",
                "value"
            ]
        ):

            value_column = column

            break

    # --------------------------------------------------------
    # DETECT RECENCY
    # --------------------------------------------------------

    recency_column = None

    for column in segmentation_features:

        if "recency" in column.lower():

            recency_column = column

            break

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = []

    for cluster in cluster_summary.index:

        segment = (
            segment_names[
                cluster
            ]
        )

        recommendation = (
            "General personalized marketing campaign"
        )

        if (
            recency_column
            and value_column
        ):

            recency = (
                cluster_summary.loc[
                    cluster,
                    recency_column
                ]
            )

            value = (
                cluster_summary.loc[
                    cluster,
                    value_column
                ]
            )

            recency_median = (
                cluster_summary[
                    recency_column
                ].median()
            )

            value_median = (
                cluster_summary[
                    value_column
                ].median()
            )

            if (
                recency <= recency_median
                and value >= value_median
            ):

                recommendation = (
                    "VIP retention, premium service "
                    "and loyalty offers"
                )

            elif (
                recency <= recency_median
                and value < value_median
            ):

                recommendation = (
                    "Upselling, product upgrades "
                    "and personalized offers"
                )

            elif (
                recency > recency_median
                and value >= value_median
            ):

                recommendation = (
                    "Win-back campaigns, renewal "
                    "offers and re-engagement"
                )

            else:

                recommendation = (
                    "Re-engagement, cross-selling "
                    "and introductory offers"
                )

        recommendations.append({

            "Cluster":
                cluster,

            "Segment":
                segment,

            "Recommendation":
                recommendation
        })

    recommendations_df = pd.DataFrame(
        recommendations
    )

    st.dataframe(
        recommendations_df,
        use_container_width=True
    )


# ============================================================
# INDIVIDUAL CUSTOMER
# ============================================================

elif analysis_option == "Individual Customer":

    st.header(
        "Individual Customer Analysis"
    )

    if id_columns:

        selected_id = st.selectbox(
            "Select Customer Identifier",
            id_columns
        )

        unique_values = (
            engineered_df[
                selected_id
            ]
            .dropna()
            .unique()
            .tolist()
        )

        selected_customer = (
            st.selectbox(
                "Select Customer",
                unique_values
            )
        )

        customer_data = (
            engineered_df[
                engineered_df[
                    selected_id
                ]
                == selected_customer
            ]
        )

        if not customer_data.empty:

            st.subheader(
                "Customer Information"
            )

            st.dataframe(
                customer_data.T,
                use_container_width=True
            )

            customer_cluster = (
                customer_data.iloc[0][
                    "Segment Name"
                ]
            )

            st.success(
                f"Customer Segment: "
                f"**{customer_cluster}**"
            )

    else:

        st.warning(
            "No suitable customer identifier "
            "was automatically detected."
        )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

elif analysis_option == "Download Results":

    st.header(
        "Download Analysis Results"
    )

    st.success(
        "Your automatically analyzed dataset is ready."
    )

    csv_data = (
        engineered_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        label="Download Complete Results",
        data=csv_data,
        file_name=(
            "customer_segmentation_results.csv"
        ),
        mime="text/csv"
    )

    st.subheader(
        "Downloaded dataset contains:"
    )

    st.write(
        """
        Original dataset columns
        Automatically engineered features
        Automatically selected clustering data
        Cluster number
        Automatically generated segment name
        """
    )


# ============================================================
# AUTOMATIC ANALYSIS SUMMARY
# ============================================================

st.markdown(
    '<div class="section-card">',
    unsafe_allow_html=True
)

st.subheader(
    "Automatic Analysis Summary"
)

st.write(
    f"""
    The system analyzed **{len(engineered_df):,} records**
    and automatically identified **{len(date_columns)} date columns**,
    **{len(numeric_columns)} numeric columns**, and
    **{len(categorical_columns)} categorical columns**.

    After automatic cleaning and feature engineering,
    **{len(segmentation_features)} features** were selected
    for customer segmentation.

    The system identified **{best_k} customer segments**
    using K-Means clustering.
    """
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# AUTOMATED PROCESSING PIPELINE
# ============================================================

st.subheader(
    "Automated Processing Pipeline"
)

steps = [

    (
        "Dataset Upload",
        "Completed"
    ),

    (
        "Dataset Detection",
        "Completed"
    ),

    (
        "Data Cleaning",
        "Completed"
    ),

    (
        "Feature Engineering",
        "Completed"
    ),

    (
        "Feature Selection",
        "Completed"
    ),

    (
        "Optimal K Detection",
        "Completed"
    ),

    (
        "Customer Segmentation",
        "Completed"
    ),

    (
        "Insight Generation",
        "Completed"
    )
]

for step, status in steps:

    col1, col2 = (
        st.columns(
            [6, 2]
        )
    )

    with col1:

        st.write(
            f"**{step}**"
        )

    with col2:

        st.success(
            status
        )