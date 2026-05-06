import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Random Forests", page_icon="🌲", layout="wide")
from utils import load_css
load_css()
st.title("🌲 Random Forests")
st.markdown("""
**Random Forests** combine the predictions of multiple Decision Trees. Each tree is trained on a random subset of the data and a random subset of features. 
This "ensemble" approach vastly reduces the overfitting seen in individual Decision Trees.
""")

st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 1, 100, 10, 1)
max_depth = st.sidebar.slider("Maximum Depth per Tree", 1, 20, 5, 1)

X, y = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=42)

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.Spectral, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Spectral)
    ax.set_title(f"Random Forest Decision Boundary ({n_estimators} Trees)")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.ensemble import RandomForestClassifier

# Initialize Model
model = RandomForestClassifier(n_estimators=10, max_depth=5)

# Fit the Model
model.fit(X, y)

# Predict
predictions = model.predict(X)
    """, language="python")