import streamlit as st
import os
import matplotlib.pyplot as plt

def load_css():
    """Reads style.css and injects it into the Streamlit page."""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
    # Apply global Matplotlib styling to match the theme
    plt.rcParams['figure.facecolor'] = '#f5efe6'
    plt.rcParams['axes.facecolor'] = '#f5efe6'
    plt.rcParams['axes.edgecolor'] = '#dfd2bf'
    plt.rcParams['axes.labelcolor'] = '#5f5646'
    plt.rcParams['xtick.color'] = '#5f5646'
    plt.rcParams['ytick.color'] = '#5f5646'
    plt.rcParams['text.color'] = '#2d2a24'
    plt.rcParams['font.sans-serif'] = ['Urbanist', 'Arial', 'sans-serif']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['grid.color'] = '#dfd2bf'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
