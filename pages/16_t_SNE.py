import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE

st.set_page_config(page_title="t-SNE", page_icon="🕸️", layout="wide")
from utils import load_css
load_css()
st.title("🕸️ t-SNE")
st.markdown("""
**t-SNE** (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique particularly well suited for visualizing high-dimensional datasets.
Unlike PCA, which tries to preserve global variance, t-SNE tries to preserve *local* neighborhoods, meaning points that are close in high dimensions stay close in low dimensions.
""")

st.sidebar.header("Model Parameters")
perplexity = st.sidebar.slider("Perplexity", 5, 50, 30, 5)
n_iter = st.sidebar.slider("Number of Iterations", 250, 1000, 300, 50)

st.sidebar.warning("t-SNE is computationally expensive. High iterations will take a few seconds to process.")

X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=0, n_classes=4, random_state=42)

@st.cache_data
def run_tsne(X, perplexity, n_iter):
    model = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    return model.fit_transform(X)

with st.spinner('Running t-SNE...'):
    X_tsne = run_tsne(X, perplexity, n_iter)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='plasma', alpha=0.7, edgecolors='k')
    ax.set_title(f"t-SNE Projection (Perplexity={perplexity})")
    st.pyplot(fig)

with col2:
    st.subheader("What is Perplexity?")
    st.write("Perplexity loosely models the number of close neighbors each point has. Typical values are between 5 and 50.")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.manifold import TSNE

# Initialize Model
model = TSNE(n_components=2, perplexity=30, n_iter=1000)

# Fit and Transform (t-SNE does not have a separate transform method)
X_reduced = model.fit_transform(X)
    """, language="python")