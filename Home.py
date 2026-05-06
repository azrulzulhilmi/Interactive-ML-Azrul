import streamlit as st

st.set_page_config(
    page_title="Interactive ML Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
from utils import load_css
load_css()

st.title("Welcome to the Interactive Machine Learning Playground! 🧠")

st.markdown("""
### What is Machine Learning?
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

### About this Playground
This application is an interactive, educational tool where you can explore and understand 20 essential machine learning algorithms. 

**How to use:**
1. **Navigate** using the sidebar on the left to select an algorithm category and specific algorithm.
2. **Read** the plain English explanation at the top of each algorithm's page.
3. **Tweak** the parameters in the sidebar to see how they affect the model's performance in real-time.
4. **View** the raw Python code required to run the algorithm by expanding the "View Code" section.

Let's get started! Select an algorithm from the sidebar.
""")
