import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

st.set_page_config(page_title="K-Means Clustering", page_icon="🎯", layout="wide")
st.title("🎯 K-Means Clustering")
st.markdown("""
**K-Means Clustering** is an unsupervised learning algorithm that groups unlabelled data into a pre-defined number of clusters (`k`). 
It iteratively assigns points to the nearest cluster centroid, and then moves the centroid to the center of the assigned points.
""")

st.sidebar.header("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4, 1)
random_state = st.sidebar.slider("Random Seed", 1, 100, 42)

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
model.fit(X)
y_kmeans = model.predict(X)
centers = model.cluster_centers_

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label="Centroids")
    ax.set_title(f"K-Means Clustering (k={n_clusters})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    st.metric("Inertia (Sum of Squared Distances)", f"{model.inertia_:.2f}")
    st.metric("Iterations to Converge", f"{model.n_iter_}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.cluster import KMeans

# Initialize Model
model = KMeans(n_clusters=4, n_init="auto")

# Fit the Model and Predict Clusters
predicted_clusters = model.fit_predict(X)

# Get Centroids
centroids = model.cluster_centers_
    """, language="python")