import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Ridge & Lasso Regression", page_icon="🔗", layout="wide")
st.title("🔗 Ridge & Lasso Regression")
st.markdown("""
**Ridge** and **Lasso** regression are regularization techniques used to prevent overfitting in linear regression by adding a penalty to the model's complexity.
*   **Ridge (L2)** penalizes the sum of squared coefficients, shrinking them but rarely to exactly zero.
*   **Lasso (L1)** penalizes the sum of absolute coefficients, often driving some exactly to zero, performing feature selection.
""")

st.sidebar.header("Model Parameters")
n_samples = st.sidebar.slider("Number of samples", 10, 500, 50, 10)
noise = st.sidebar.slider("Noise level", 0.0, 50.0, 20.0, 1.0)
alpha_val = st.sidebar.slider("Regularization Strength (Alpha)", 0.01, 100.0, 10.0, 0.1)

# Generate Data with an outlier to show regularization effect
np.random.seed(42)
X = np.random.rand(n_samples, 1) * 10
y = 2.5 * X.squeeze() + 10.0 + np.random.randn(n_samples) * noise
# Add outlier
X = np.append(X, [[9.5]])
y = np.append(y, [150.0])
X_plot = np.linspace(0, 10, 100)[:, np.newaxis]

models = {
    "Linear (No Penalty)": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=alpha_val),
    "Lasso (L1)": Lasso(alpha=alpha_val)
}

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color="blue", label="Data Points", alpha=0.6)
    
    for name, model in models.items():
        model.fit(X, y)
        y_pred_plot = model.predict(X_plot)
        ax.plot(X_plot, y_pred_plot, label=name, linewidth=2)
        
    ax.set_title(f"Regularization Comparison (Alpha={alpha_val})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Model Coefficients")
    for name, model in models.items():
        st.write(f"**{name}**")
        st.write(f"Slope: {model.coef_[0]:.2f}")
        st.write(f"Intercept: {model.intercept_:.2f}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from sklearn.linear_model import Ridge, Lasso, LinearRegression

# Standard Linear Regression
model_linear = LinearRegression()
model_linear.fit(X, y)

# Ridge Regression (L2 Penalty)
model_ridge = Ridge(alpha=10.0)
model_ridge.fit(X, y)

# Lasso Regression (L1 Penalty)
model_lasso = Lasso(alpha=10.0)
model_lasso.fit(X, y)
    """, language="python")