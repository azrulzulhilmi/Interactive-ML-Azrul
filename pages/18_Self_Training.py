import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Self Training", page_icon="🧑‍🏫", layout="wide")
from utils import load_css
load_css()
st.title("🧑‍🏫 Self Training (Semi-Supervised)")
st.markdown("""
**Self-Training** is a semi-supervised learning algorithm. It starts by training a model on a small amount of labeled data. 
Then, it uses that model to predict labels for the unlabeled data. The most confident predictions are added to the training set, and the model is retrained. This repeats until no more confident predictions can be made.
""")

st.sidebar.header("Model Parameters")
unlabeled_ratio = st.sidebar.slider("Percentage of Unlabeled Data", 0.1, 0.9, 0.5, 0.1)
threshold = st.sidebar.slider("Confidence Threshold for Pseudo-labels", 0.5, 0.99, 0.8, 0.05)

# Generate Data
X, y_true = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42, n_clusters_per_class=1)

# Mask labels to create unlabeled data
rng = np.random.RandomState(42)
y_train = np.copy(y_true)
unlabeled_indices = rng.rand(len(y_train)) < unlabeled_ratio
y_train[unlabeled_indices] = -1  # -1 represents unlabeled data in scikit-learn

base_classifier = SVC(probability=True, gamma="auto", random_state=42)
self_training_model = SelfTrainingClassifier(base_classifier, threshold=threshold)

self_training_model.fit(X, y_train)
y_pred = self_training_model.predict(X)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: True Labels vs Unlabeled
    ax[0].scatter(X[y_train == -1, 0], X[y_train == -1, 1], color='gray', label="Unlabeled", alpha=0.5, marker='.')
    ax[0].scatter(X[y_train == 0, 0], X[y_train == 0, 1], color='blue', label="Class 0", edgecolors='k')
    ax[0].scatter(X[y_train == 1, 0], X[y_train == 1, 1], color='red', label="Class 1", edgecolors='k')
    ax[0].set_title("Initial Training Data")
    ax[0].legend()

    # Plot 2: Final Predictions
    ax[1].scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], color='blue', label="Predicted Class 0", alpha=0.7)
    ax[1].scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], color='red', label="Predicted Class 1", alpha=0.7)
    ax[1].set_title("Self-Training Final Predictions")
    ax[1].legend()

    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    st.metric("Total Samples", len(y_train))
    st.metric("Unlabeled Samples Initially", sum(y_train == -1))
    
    # Calculate accuracy on the initially unlabeled points
    unlabeled_mask = (y_train == -1)
    if any(unlabeled_mask):
        acc = accuracy_score(y_true[unlabeled_mask], y_pred[unlabeled_mask])
        st.metric("Accuracy on Unlabeled Data", f"{acc:.2%}")
    st.metric("Iterations Taken", self_training_model.n_iter_)

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

# Create a base classifier that can output probabilities
base_model = SVC(probability=True)

# Wrap it in the SelfTrainingClassifier
self_training_model = SelfTrainingClassifier(base_model, threshold=0.8)

# Mask some labels by setting them to -1
y_train[50:150] = -1 

# Fit the model (it will iteratively pseudo-label the -1 data)
self_training_model.fit(X, y_train)

# Predict all labels
predictions = self_training_model.predict(X)
    """, language="python")