# 🌸 Iris Flower Classification Dashboard

A professional, interactive machine learning dashboard built with **Streamlit** to explore, visualize, and model the famous Iris dataset. This dashboard combines aesthetic data storytelling with powerful machine learning capabilities.

## 🚀 Overview

The **Iris Flower Classification Dashboard** provides a comprehensive suite of tools for data analysis and machine learning. From exploring the raw dataset to visualizing complex 3D feature spaces and training predictive models in real-time, this application serves as a perfect example of a modern data science web app.

## ✨ Features

- **🏠 Overview**: High-level metrics, class distribution, and correlation heatmaps.
- **📊 Data Explorer**: View the complete dataset, statistical summaries, and filter data by species and measurement ranges.
- **🎨 3D Visualizations**:
    - Interactive 3D Scatter Plots of feature dimensions.
    - Principal Component Analysis (PCA) projections.
    - Custom feature axis selection.
- **🤖 Machine Learning Models**:
    - Train and compare **Logistic Regression**, **Support Vector Machine (SVM)**, and **Random Forest** models.
    - View confusion matrices and detailed classification reports.
    - Compare model accuracy and performance metrics.
- **🔮 Interactive Predictions**:
    - Input custom sepal and petal measurements via sliders.
    - Get real-time predictions from all three trained models with confidence scores.
- **📈 Feature Analysis**:
    - Visualize feature importance (Random Forest).
    - Compare feature distributions across species with violin plots and radar charts.

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)**: The core web framework for the dashboard.
- **[Plotly](https://plotly.com/)**: For interactive, high-quality 2D and 3D visualizations.
- **[Scikit-learn](https://scikit-learn.org/)**: For machine learning models, preprocessing, and metrics.
- **[Pandas](https://pandas.pydata.org/)**: For data manipulation and analysis.
- **Python**: The underlying programming language.

## 📦 Installation & Usage

1.  **Clone the repository** (if applicable) or download the source files.

2.  **Install dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**:
    Execute the following command in your terminal:
    ```bash
    streamlit run iris_dashboard.py
    ```

4.  **Access the App**:
    The dashboard will open in your default web browser at `http://localhost:8501`.

## 🗂️ Project Structure

```text
CodAlpha_Iris/
├── iris_dashboard.py       # Main Streamlit dashboard application
├── iris_codealpha.py       # Standalone EDA and ML analysis script
├── Iris.csv                # The dataset file
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── styles/
│   └── custom.css          # Custom CSS for UI styling (gradients, animations)
```

## 🎨 UI/UX Design

The application features a **premium dark/light theme** enhanced with custom CSS:
- **Glassmorphism** effects on cards and containers.
- **Gradient Text** headers for a modern look.
- **Responsive Layout** that adapts to different screen sizes.
- **Interactive Elements** like sliders, tabs, and hover effects.

---
*Created for the CodAlpha Iris Classification Project.*
