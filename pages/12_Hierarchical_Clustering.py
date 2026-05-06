import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Hierarchical Clustering", page_icon="🧬", layout="wide")
st.title("🧬 Hierarchical Clustering")
st.markdown("""
**Hierarchical Clustering** groups data by building a hierarchy of clusters. 
*Agglomerative* clustering (used here) is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
""")

st.sidebar.header("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, 1)
linkage_type = st.sidebar.selectbox("Linkage Type", ["ward", "complete", "average", "single"])

X, _ = make_blobs(n_samples=150, centers=3, random_state=42, cluster_std=1.0)

model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
y_pred = model.fit_predict(X)

col1, col2 = st.columns([2, 2])
with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', alpha=0.7, edgecolors='k')
    ax.set_title(f"Agglomerative Clustering ({linkage_type} linkage)")
    st.pyplot(fig)

with col2:
    st.subheader("Dendrogram")
    st.markdown("A dendrogram visualizes the hierarchical merging process.")
    fig_dend, ax_dend = plt.subplots(figsize=(8, 6))
    Z = linkage(X, method=linkage_type)
    dendrogram(Z, ax=ax_dend, truncate_mode='level', p=3)
    ax_dend.set_title("Hierarchical Clustering Dendrogram")
    st.pyplot(fig_dend)

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Initialize Model
model = AgglomerativeClustering(n_clusters=3, linkage="ward")

# Fit and Predict
clusters = model.fit_predict(X)

# Compute Linkage Matrix for Dendrogram
Z = linkage(X, method="ward")
dendrogram(Z)
    """, language="python")