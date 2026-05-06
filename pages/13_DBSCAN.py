import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="DBSCAN", page_icon="🌌", layout="wide")
from utils import load_css
load_css()
st.title("🌌 DBSCAN")
st.markdown("""
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed together (points with many nearby neighbors).
Points that lie alone in low-density regions (whose nearest neighbors are too far away) are marked as outliers/noise (represented as black dots). It does not require specifying the number of clusters in advance!
""")

st.sidebar.header("Model Parameters")
eps = st.sidebar.slider("Epsilon (Radius of neighborhood)", 0.05, 0.5, 0.2, 0.01)
min_samples = st.sidebar.slider("Minimum Samples to form a dense region", 2, 20, 5, 1)

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

model = DBSCAN(eps=eps, min_samples=min_samples)
y_pred = model.fit_predict(X)

# DBSCAN marks noise as -1
n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise_ = list(y_pred).count(-1)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points, noise in black
    unique_labels = set(y_pred)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1] # Black used for noise.
            
        class_member_mask = (y_pred == k)
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=8)

    ax.set_title(f"DBSCAN Clustering")
    st.pyplot(fig)

with col2:
    st.subheader("Clustering Metrics")
    st.metric("Estimated Clusters", f"{n_clusters_}")
    st.metric("Noise Points (Outliers)", f"{n_noise_}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.cluster import DBSCAN

# Initialize Model
model = DBSCAN(eps=0.2, min_samples=5)

# Fit and Predict (-1 means noise/outlier)
clusters = model.fit_predict(X)
    """, language="python")