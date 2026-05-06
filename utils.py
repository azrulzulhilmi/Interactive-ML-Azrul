import streamlit as st
import os

def load_css():
    """Reads style.css and injects it into the Streamlit page."""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
