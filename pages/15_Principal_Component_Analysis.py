import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

st.set_page_config(page_title="Principal Component Analysis", page_icon="📉", layout="wide")
from utils import load_css
load_css()
st.title("📉 Principal Component Analysis (PCA)")
st.markdown("""
**PCA** is a dimensionality reduction technique. It transforms high-dimensional data into a lower-dimensional space while retaining as much variance (information) as possible.
It works by finding the directions (Principal Components) along which the data varies the most.
""")

st.sidebar.header("Model Parameters")
st.sidebar.write("We start with a 5-Dimensional dataset and use PCA to project it down to 2 or 3 Dimensions for visualization.")
n_components = st.sidebar.slider("Number of Components to keep", 2, 3, 2)

X, y = make_classification(n_samples=300, n_features=5, n_informative=3, n_redundant=0, n_classes=3, random_state=42)

model = PCA(n_components=n_components)
X_pca = model.fit_transform(X)

col1, col2 = st.columns([3, 1])
with col1:
    fig = plt.figure(figsize=(10, 6))
    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='k')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
    else:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.7, edgecolors='k')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
    
    ax.set_title(f"PCA Projection ({n_components} Components)")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    explained_variance = model.explained_variance_ratio_
    st.metric("Total Variance Explained", f"{np.sum(explained_variance):.2%}")
    for i, var in enumerate(explained_variance):
        st.write(f"PC {i+1}: {var:.2%}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.decomposition import PCA

# Initialize Model (reduce to 2 dimensions)
model = PCA(n_components=2)

# Fit and Transform the data
X_reduced = model.fit_transform(X)

# See how much information was kept
explained_variance = model.explained_variance_ratio_
    """, language="python")