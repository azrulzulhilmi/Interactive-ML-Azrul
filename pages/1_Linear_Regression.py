import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Linear Regression", page_icon="📈", layout="wide")
from utils import load_css
load_css()

st.title("📈 Linear Regression")

st.markdown("""
**Linear Regression** is one of the simplest supervised learning algorithms. It assumes a linear relationship between the input variables (X) and the single output variable (Y). 
It calculates the best fit line that minimizes the sum of the squared differences between the predicted values and the actual values (also known as the residual sum of squares).
""")

# --- Sidebar Controls ---
st.sidebar.header("Model Parameters")
n_samples = st.sidebar.slider("Number of samples", min_value=10, max_value=500, value=100, step=10)
noise = st.sidebar.slider("Noise level", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
true_slope = st.sidebar.slider("True Slope", min_value=-10.0, max_value=10.0, value=2.5, step=0.5)
true_intercept = st.sidebar.slider("True Intercept", min_value=-50.0, max_value=50.0, value=10.0, step=5.0)

# --- Data Generation ---
np.random.seed(42)
X = np.random.rand(n_samples, 1) * 100
y = true_slope * X.squeeze() + true_intercept + np.random.randn(n_samples) * noise

# --- Model Training ---
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# --- Visualization ---
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color="blue", label="Data Points", alpha=0.6)
    ax.plot(X, y_pred, color="red", label="Regression Line", linewidth=2)
    ax.set_xlabel("X (Feature)")
    ax.set_ylabel("Y (Target)")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")
    
    st.subheader("Learned Parameters")
    st.metric("Estimated Slope", f"{model.coef_[0]:.2f}")
    st.metric("Estimated Intercept", f"{model.intercept_:.2f}")

# --- Code Transparency ---
st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate Data
X = np.random.rand(100, 1) * 100
y = 2.5 * X.squeeze() + 10.0 + np.random.randn(100) * 10.0

# 2. Initialize Model
model = LinearRegression()

# 3. Fit Model
model.fit(X, y)

# 4. Predict
y_pred = model.predict(X)

# 5. Plot
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.legend()
plt.show()
    """, language="python")
