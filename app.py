import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# -----------------------------
# Load data and models
# -----------------------------

df = pd.read_csv("data/customer_segments.csv")

scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
pca = joblib.load("models/pca.pkl")


# -----------------------------
# Title
# -----------------------------

st.title("Customer Segmentation Dashboard")

st.write(
    "Explore customer groups discovered using K-Means clustering."
)


# -----------------------------
# Metrics
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    len(df)
)

col2.metric(
    "Number of Clusters",
    df["Cluster"].nunique()
)

col3.metric(
    "Average Spending",
    f"{df['Spending'].mean():.2f}"
)


# -----------------------------
# Cluster sizes
# -----------------------------

st.subheader("Cluster Sizes")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)


# -----------------------------
# Cluster characteristics
# -----------------------------

st.subheader("Cluster Characteristics")

features = [
    "Income",
    "Spending",
    "Purchase_Frequency",
    "Average_Order_Value"
]

profile = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

st.dataframe(
    profile,
    use_container_width=True
)


# -----------------------------
# PCA visualization
# -----------------------------

st.subheader("Customer Segments")

fig, ax = plt.subplots(figsize=(10, 6))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    ax.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        label=f"Cluster {cluster}",
        alpha=0.7
    )

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Customer Segmentation")

ax.legend()

st.pyplot(fig)


# -----------------------------
# New customer prediction
# -----------------------------

st.subheader("Assign a New Customer")

income = st.number_input(
    "Income",
    min_value=0.0,
    value=50000.0
)

spending = st.number_input(
    "Spending",
    min_value=0.0,
    value=2000.0
)

frequency = st.number_input(
    "Purchase Frequency",
    min_value=1,
    value=10
)

aov = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=100.0
)


if st.button("Find Customer Segment"):

    new_customer = np.array([
        [
            income,
            spending,
            frequency,
            aov
        ]
    ])

    new_customer_scaled = scaler.transform(
        new_customer
    )

    cluster = kmeans.predict(
        new_customer_scaled
    )[0]

    st.success(
        f"This customer belongs to Cluster {cluster}"
    )