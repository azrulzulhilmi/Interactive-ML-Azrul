import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Decision Trees", page_icon="🌳", layout="wide")
from utils import load_css
load_css()
st.title("🌳 Decision Trees")
st.markdown("""
A **Decision Tree** continuously splits the data into two branches based on the feature that provides the best separation, eventually ending in 'leaf' nodes that dictate the class.
They are highly prone to **overfitting** if the depth of the tree is not restricted.
""")

st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Maximum Depth", 1, 15, 4, 1)
min_samples_split = st.sidebar.slider("Minimum Samples to Split", 2, 20, 2, 1)

X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)

model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.Paired, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    ax.set_title(f"Decision Tree Boundary (Depth={max_depth})")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.metric("Actual Depth", f"{model.get_depth()}")

st.markdown("---")
st.subheader("Tree Structure")
with st.expander("View Tree Plot"):
    fig_tree, ax_tree = plt.subplots(figsize=(15, 10))
    plot_tree(model, filled=True, ax=ax_tree, rounded=True)
    st.pyplot(fig_tree)

with st.expander("View Python Code"):
    st.code("""
from sklearn.tree import DecisionTreeClassifier

# Initialize Model
model = DecisionTreeClassifier(max_depth=4)

# Fit the Model
model.fit(X, y)

# Predict
predictions = model.predict(X)
    """, language="python")