import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Support Vector Machines", page_icon="⚔️", layout="wide")
from utils import load_css
load_css()
st.title("⚔️ Support Vector Machines (SVM)")
st.markdown("""
**Support Vector Machines** find the hyperplane that maximizes the margin (distance) between different classes. 
By using the **Kernel Trick**, SVMs can easily map data into higher dimensions to separate complex, non-linear data points.
""")

st.sidebar.header("Model Parameters")
kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
C_val = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)

# Only show gamma if kernel is rbf, poly, or sigmoid
if kernel in ["rbf", "poly", "sigmoid"]:
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
else:
    gamma = "scale"

X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

model = SVC(kernel=kernel, C=C_val, gamma=gamma, random_state=42)
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.coolwarm, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)
    ax.set_title(f"SVM Decision Boundary (Kernel={kernel})")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.metric("Support Vectors", f"{len(model.support_)}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.svm import SVC

# Initialize Model
model = SVC(kernel="rbf", C=1.0, gamma="scale")

# Fit the Model
model.fit(X, y)

# Predict
predictions = model.predict(X)
    """, language="python")