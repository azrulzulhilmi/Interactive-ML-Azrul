### Project Brief: Interactive Machine Learning Playground

**Objective**
To build an interactive, educational web application where users can explore and understand 20 essential machine learning algorithms. The platform will allow users to adjust model parameters and see real time results, bridging the gap between theoretical code and visual understanding.

**1. Tech Stack**
*   **Frontend and Backend:** Streamlit (Python framework for rapid UI development)
*   **Machine Learning Core:** scikit-learn, TensorFlow, numpy
*   **Visualizations:** matplotlib, seaborn, or Streamlit native charts
*   **Deployment:** Streamlit Community Cloud (free and beginner friendly)

**2. Key Features**
*   **Interactive Parameter Controls:** Sliders, dropdowns, and input boxes allowing the user to tweak algorithm variables (like the number of clusters in K Means or the learning rate in Neural Networks).
*   **Real Time Visualizations:** Dynamic scatter plots, line charts, and confusion matrices that update instantly when parameters are changed.
*   **Code Transparency:** A dedicated toggle or section on each page displaying the raw Python code required to run the specific algorithm.
*   **Plain English Explanations:** Short, jargon free descriptions at the top of every page explaining what the algorithm does and what the parameters mean.

**3. Website Sections**
The application will use a multipage structure with a sidebar navigation menu divided into four main zones.

*   **Home Page:** An introduction to the playground, explaining what machine learning is and how to use the interactive tools.
*   **Supervised Learning:** 
    *   Regression: Linear Regression, Ridge and Lasso Regression, Polynomial Regression.
    *   Classification: Logistic Regression, K Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, Gradient Boosting, Naive Bayes.
*   **Unsupervised Learning:**
    *   Clustering: K Means Clustering, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models.
    *   Dimensionality Reduction: Principal Component Analysis (PCA), t-SNE, Autoencoders.
*   **Specialized Algorithms:**
    *   Semi Supervised: Self Training.
    *   Reinforcement Learning: Q Learning, Deep Q Networks (DQN), Policy Gradient Methods.
    *   Anomaly Detection: One Class SVM, Isolation Forest.
    *   Deep Learning: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long Short Term Memory (LSTM), Transformers.

**4. Build Steps**
*   **Step 1: Environment Setup.** Install Python, set up a virtual environment, and install the required libraries (Streamlit, scikit-learn, TensorFlow).
*   **Step 2: Skeleton Application.** Set up the Streamlit multipage folder structure. Create a `Home.py` file and a `pages` directory filled with empty Python scripts for all 20 algorithms.
*   **Step 3: The First Prototype.** Build one complete algorithm page first (like Linear Regression). Implement the model, the interactive sliders, and the visualization. This will serve as the UI template for the rest of the site.
*   **Step 4: Scale Out.** Systematically populate the remaining 19 pages using the established template, importing the logic directly from the provided course material.
*   **Step 5: Testing and Refinement.** Test all sliders to ensure they do not crash the models with extreme values. Add error handling and clean up the educational text.
