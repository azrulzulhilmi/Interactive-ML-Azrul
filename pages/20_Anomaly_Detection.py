import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

st.set_page_config(page_title="Anomaly Detection", page_icon="🕵️", layout="wide")
st.title("🕵️ Anomaly Detection (Isolation Forest)")
st.markdown("""
**Isolation Forests** are used for Anomaly Detection. They "isolate" observations by randomly selecting a feature and randomly selecting a split value between the max and min values of that feature.
Anomalies (outliers) are usually isolated very quickly (closer to the root of the tree), while normal points require more splits to be isolated.
""")

st.sidebar.header("Model Parameters")
contamination = st.sidebar.slider("Contamination (Expected % of Outliers)", 0.01, 0.20, 0.05, 0.01)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)

# Generate normal data and outliers
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0, random_state=42)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
model.fit(X)

# Predict (-1 is outlier, 1 is inlier)
y_pred = model.predict(X)
anomaly_scores = model.decision_function(X) # Lower score = more abnormal

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Inliers
    ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], color='blue', label='Normal (Inlier)', alpha=0.6, edgecolors='k')
    
    # Plot Outliers
    ax.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='red', label='Anomaly (Outlier)', alpha=0.9, marker='X', s=100)
    
    # Add contour map of anomaly scores
    xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z, cmap=plt.cm.Blues_r, alpha=0.3)
    
    ax.set_title(f"Isolation Forest (Contamination={contamination:.2f})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Detection Metrics")
    st.metric("Total Data Points", len(X))
    st.metric("Detected Anomalies", sum(y_pred == -1))
    st.metric("Detected Normal", sum(y_pred == 1))

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.ensemble import IsolationForest

# Initialize Model
model = IsolationForest(contamination=0.05, n_estimators=100)

# Fit the Model
model.fit(X)

# Predict Anomalies (-1 = Outlier, 1 = Inlier)
predictions = model.predict(X)

# Get raw anomaly scores (lower means more anomalous)
scores = model.decision_function(X)
    """, language="python")