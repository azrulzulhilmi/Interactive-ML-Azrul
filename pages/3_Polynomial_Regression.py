import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Polynomial Regression", page_icon="📈", layout="wide")
st.title("📈 Polynomial Regression")
st.markdown("""
**Polynomial Regression** extends linear regression by adding extra predictors obtained by raising the original features to a given power (degree). This allows the model to fit non-linear, curved relationships. 
Beware of high degrees, as they easily lead to **overfitting** (the line capturing noise rather than the trend).
""")

st.sidebar.header("Model Parameters")
degree = st.sidebar.slider("Polynomial Degree", 1, 15, 3, 1)
noise = st.sidebar.slider("Noise level", 0.0, 5.0, 1.5, 0.1)

np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100) * noise
X = X[:, np.newaxis]

model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

X_plot = np.linspace(min(X), max(X), 100)
y_plot = model.predict(X_plot)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color="blue", label="Data Points", alpha=0.6)
    ax.plot(X_plot, y_plot, color="red", label=f"Degree {degree} Fit", linewidth=2)
    ax.set_title(f"Polynomial Regression (Degree {degree})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    st.metric("Mean Squared Error", f"{mse:.2f}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create a pipeline that transforms features, then fits linear model
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# Fit the model
model.fit(X, y)

# Predict
y_pred = model.predict(X_new)
    """, language="python")