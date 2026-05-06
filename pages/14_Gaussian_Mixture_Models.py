import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title="Gaussian Mixture Models", page_icon="☁️", layout="wide")
from utils import load_css
load_css()
st.title("☁️ Gaussian Mixture Models (GMM)")
st.markdown("""
A **Gaussian Mixture Model** assumes that the data is generated from a mixture of a finite number of Gaussian (normal) distributions with unknown parameters.
Unlike K-Means (which assumes clusters are spherical), GMMs can form elliptical clusters. They also provide "soft clustering", giving the probability that a point belongs to a particular cluster.
""")

st.sidebar.header("Model Parameters")
n_components = st.sidebar.slider("Number of Components (Gaussians)", 1, 6, 3, 1)

# Generate data with elliptical blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)

model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
y_pred = model.fit_predict(X_aniso)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred, cmap='viridis', zorder=2, alpha=0.6, edgecolors='k')
    
    # Plotting the ellipses for the Gaussian distributions
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        for nsig in range(1, 4):
            ax.add_patch(plt.matplotlib.patches.Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))
            
    for pos, covar, w in zip(model.means_, model.covariances_, model.weights_):
        draw_ellipse(pos, covar, alpha=w * 0.5, color='red', fill=False, linewidth=2)
        
    ax.set_title(f"Gaussian Mixture Model ({n_components} components)")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    st.metric("Converged?", f"{model.converged_}")
    st.metric("Iterations", f"{model.n_iter_}")
    st.metric("AIC", f"{model.aic(X_aniso):.0f}")
    st.metric("BIC", f"{model.bic(X_aniso):.0f}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.mixture import GaussianMixture

# Initialize Model
model = GaussianMixture(n_components=3, covariance_type='full')

# Fit Model
model.fit(X)

# Predict clusters
clusters = model.predict(X)

# Get probabilities of belonging to each cluster
probabilities = model.predict_proba(X)
    """, language="python")