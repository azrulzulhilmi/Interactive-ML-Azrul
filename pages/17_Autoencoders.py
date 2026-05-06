import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

st.set_page_config(page_title="Autoencoders", page_icon="🪞", layout="wide")
from utils import load_css
load_css()
st.title("🪞 Autoencoders")
st.markdown("""
An **Autoencoder** is a type of artificial neural network used to learn efficient codings of unlabeled data. 
It consists of two parts: an **Encoder** that compresses the data into a lower-dimensional representation (bottleneck), and a **Decoder** that tries to reconstruct the original data from the compressed version.
""")

st.sidebar.header("Model Parameters")
st.sidebar.write("We will generate random 10D vectors and compress them to a 2D bottleneck.")
bottleneck_dim = st.sidebar.slider("Bottleneck Dimensions", 1, 5, 2, 1)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, 10)

# Generate synthetic 10D data
np.random.seed(42)
tf.random.set_seed(42)
X = np.random.randn(500, 10)

@st.cache_resource
def train_autoencoder(X, bottleneck_dim, epochs):
    input_layer = Input(shape=(10,))
    encoded = Dense(bottleneck_dim, activation='relu')(input_layer)
    decoded = Dense(10, activation='linear')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X, X, epochs=epochs, batch_size=32, shuffle=True, verbose=0)
    
    return autoencoder, encoder, history.history['loss']

with st.spinner("Training Autoencoder..."):
    autoencoder, encoder, loss_history = train_autoencoder(X, bottleneck_dim, epochs)

X_encoded = encoder.predict(X)
X_reconstructed = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - X_reconstructed))

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Training Loss
    ax[0].plot(loss_history, color="blue", label="MSE Loss")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    
    # Plot Encoded Representation
    if bottleneck_dim >= 2:
        ax[1].scatter(X_encoded[:, 0], X_encoded[:, 1], color="red", alpha=0.5)
        ax[1].set_title(f"Encoded 2D Representation")
    else:
        ax[1].hist(X_encoded[:, 0], bins=30, color="red", alpha=0.7)
        ax[1].set_title(f"Encoded 1D Representation")
        
    st.pyplot(fig)

with col2:
    st.subheader("Model Metrics")
    st.metric("Final Loss (MSE)", f"{loss_history[-1]:.4f}")
    st.metric("Reconstruction Error", f"{reconstruction_error:.4f}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define dimensions
input_dim = 10
bottleneck_dim = 2

# Build Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(bottleneck_dim, activation='relu')(input_layer)

# Build Decoder
decoded = Dense(input_dim, activation='linear')(encoded)

# Compile full autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train (Note: X is both the input and the target output)
autoencoder.fit(X, X, epochs=50)

# Compress data using just the encoder part
encoder = Model(input_layer, encoded)
compressed_data = encoder.predict(X)
    """, language="python")