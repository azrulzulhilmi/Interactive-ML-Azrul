import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="K-Nearest Neighbors", page_icon="📍", layout="wide")
st.title("📍 K-Nearest Neighbors (KNN)")
st.markdown("""
**K-Nearest Neighbors** is a simple, intuitive algorithm that classifies a new data point based on the majority class of its 'K' closest neighbors in the training data.
A smaller K creates a highly complex, jagged decision boundary (prone to overfitting), while a larger K creates a smoother, more generalized boundary.
""")

st.sidebar.header("Model Parameters")
n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 30, 5, 1)
weights = st.sidebar.selectbox("Weighting", ["uniform", "distance"])

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=42, n_clusters_per_class=1, flip_y=0.1)

model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.RdYlBu, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.RdYlBu)
    ax.set_title(f"KNN Decision Boundary (K={n_neighbors})")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.neighbors import KNeighborsClassifier

# Initialize Model
model = KNeighborsClassifier(n_neighbors=5, weights="uniform")

# Fit the Model
model.fit(X, y)

# Predict
predictions = model.predict(X)
    """, language="python")