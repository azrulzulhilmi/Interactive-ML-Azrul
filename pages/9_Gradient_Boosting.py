import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Gradient Boosting", page_icon="🚀", layout="wide")
from utils import load_css
load_css()
st.title("🚀 Gradient Boosting")
st.markdown("""
Unlike Random Forests which build trees independently, **Gradient Boosting** builds trees sequentially. 
Each new tree tries to correct the errors (residuals) made by the combination of all previous trees. This leads to highly accurate models but requires careful tuning to prevent overfitting.
""")

st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Estimators", 1, 200, 50, 5)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.05)
max_depth = st.sidebar.slider("Maximum Depth per Tree", 1, 10, 3, 1)

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.PiYG, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.PiYG)
    ax.set_title(f"Gradient Boosting Boundary")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.ensemble import GradientBoostingClassifier

# Initialize Model
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)

# Fit the Model
model.fit(X, y)

# Predict
predictions = model.predict(X)
    """, language="python")