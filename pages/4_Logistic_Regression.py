import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Logistic Regression", page_icon="S", layout="wide")
from utils import load_css
load_css()
st.title("Logistic Regression")
st.markdown("""
Despite its name, **Logistic Regression** is a *classification* algorithm. It predicts the probability that an instance belongs to a given class. If the probability is > 50%, it assigns the instance to the positive class.
It creates a linear decision boundary separating the classes.
""")

st.sidebar.header("Model Parameters")
C_val = st.sidebar.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0, 0.1)
solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])

X, y = make_classification(n_samples=150, n_features=2, n_redundant=0, n_informative=2,
                           random_state=42, n_clusters_per_class=1, flip_y=0.1)

model = LogisticRegression(C=C_val, solver=solver, random_state=42)
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.RdBu, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.RdBu)
    ax.set_title("Logistic Regression Decision Boundary")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.linear_model import LogisticRegression

# Initialize Model (C is inverse of regularization strength)
model = LogisticRegression(C=1.0, solver="lbfgs")

# Fit the Model
model.fit(X, y)

# Predict probabilities and classes
probabilities = model.predict_proba(X)
predictions = model.predict(X)
    """, language="python")