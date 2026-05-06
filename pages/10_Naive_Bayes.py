import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(page_title="Naive Bayes", page_icon="📊", layout="wide")
from utils import load_css
load_css()
st.title("📊 Naive Bayes")
st.markdown("""
**Naive Bayes** is a probabilistic classifier based on applying Bayes' theorem. It assumes that all features are independent of each other (the "naive" assumption).
Despite this strong assumption, it often performs surprisingly well, especially on text classification tasks. We use **Gaussian Naive Bayes** here, which assumes the features follow a normal distribution.
""")

st.sidebar.header("Model Parameters")
st.sidebar.write("Gaussian Naive Bayes has very few hyperparameters to tune. It estimates the mean and variance for each class directly from the training data.")

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

model = GaussianNB()
model.fit(X, y)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.Accent, alpha=0.5, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Accent)
    ax.set_title("Gaussian Naive Bayes Boundary")
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    accuracy = model.score(X, y)
    st.metric("Accuracy", f"{accuracy:.2%}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.naive_bayes import GaussianNB

# Initialize Model
model = GaussianNB()

# Fit the Model
model.fit(X, y)

# Predict
predictions = model.predict(X)
    """, language="python")