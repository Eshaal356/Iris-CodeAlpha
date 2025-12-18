"""
Advanced Iris Dataset Dashboard
Professional ML Classification Dashboard with Interactive Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Iris ML Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
@st.cache_data
def load_css():
    # Light theme CSS
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .explanation-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #2d3748;
        }
        .stMarkdown p, .stMarkdown li {
            color: #2d3748;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1a202c;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    if "Id" in df.columns:
        df.drop("Id", axis=1, inplace=True)
    return df

@st.cache_data
def prepare_data(df):
    X = df.drop("Species", axis=1)
    y = df["Species"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# Train models
@st.cache_resource
def train_models(X_train_scaled, y_train, X_train, X_test_scaled, y_test, X_test):
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    # SVM
    svm_model = SVC(kernel="rbf", C=10, gamma=0.1, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    return {
        'lr': (lr_model, lr_pred, lr_acc),
        'svm': (svm_model, svm_pred, svm_acc),
        'rf': (rf_model, rf_pred, rf_acc)
    }

# Load data
df = load_data()
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)
models = train_models(X_train_scaled, y_train, X_train, X_test_scaled, y_test, X_test)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 2.5rem;'>🌸 <span class='gradient-text'>Iris</span></h1>
            <p style='color: #4a5568; font-size: 0.9rem;'>Machine Learning Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation:",
        ["🏠 Overview", "📊 Data Explorer", "🎨 3D Visualizations", "🤖 ML Models", "🔮 Predictions", "📈 Feature Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Samples", len(df))
    st.metric("Features", len(df.columns) - 1)
    st.metric("Classes", df['Species'].nunique())

# Main content
if page == "🏠 Overview":
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 class='gradient-text' style='font-size: 3rem;'>Iris Flower Classification</h1>
            <p style='color: #4a5568; font-size: 1.2rem;'>Advanced Machine Learning Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("""
        <div class='explanation-box'>
            <strong>📚 About This Dashboard:</strong><br>
            This dashboard analyzes the famous Iris flower dataset, containing measurements of 150 iris flowers from 3 different species. 
            It uses machine learning to classify flowers based on their sepal and petal measurements. 
            Explore the data, visualize patterns, and see how different ML models perform!
        </div>
    """, unsafe_allow_html=True)
    
    # Class distribution
    st.markdown("### 🌸 Species Distribution")
    st.markdown("*The dataset contains equal numbers of each species, making it perfectly balanced for training machine learning models.*")
    
    col1, col2, col3 = st.columns(3)
    species_counts = df['Species'].value_counts()
    
    for idx, (col, species) in enumerate(zip([col1, col2, col3], species_counts.index)):
        with col:
            count = species_counts[species]
            colors = ['#667eea', '#4facfe', '#43e97b']
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #4a5568;'>{species.replace('Iris-', '').title()}</div>
                    <div style='font-size: 2.5rem; color: {colors[idx]}; font-weight: bold;'>{count}</div>
                    <div style='font-size: 0.85rem; color: #718096;'>{count/len(df)*100:.1f}% of dataset</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature statistics
    st.markdown("### 📊 Feature Measurements")
    st.markdown("*Each iris flower has 4 measurements: sepal length, sepal width, petal length, and petal width (all in centimeters).*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribution of All Features**")
        st.markdown("_This box plot shows the range, median, and quartiles for each measurement across all flowers._")
        
        fig = go.Figure()
        for feature in df.columns[:-1]:
            fig.add_trace(go.Box(
                y=df[feature],
                name=feature.replace('Cm', ''),
                marker=dict(color=['#667eea', '#4facfe', '#43e97b', '#fa709a'][list(df.columns[:-1]).index(feature)])
            ))
        
        fig.update_layout(
            title="Feature Distributions",
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            font={'color': "#2d3748"},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Feature Correlations**")
        st.markdown("_This heatmap shows how features relate to each other. Values close to 1 mean strong positive correlation._")
        
        # Correlation heatmap
        corr = df[df.columns[:-1]].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Feature Correlations",
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            font={'color': "#2d3748"},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Key Insight:** Petal length and petal width are highly correlated (0.96), meaning they tend to increase together!")

elif page == "📊 Data Explorer":
    st.markdown("<h1>📊 <span class='gradient-text'>Data Explorer</span></h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='explanation-box'>
            <strong>🔍 Explore the Dataset:</strong><br>
            View the complete Iris dataset, examine statistical summaries, and filter data based on your criteria. 
            This helps you understand the data before building machine learning models.
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Complete Dataset", "📊 Statistical Summary", "🔍 Custom Filters"])
    
    with tab1:
        st.markdown("### 📋 All 150 Iris Flower Measurements")
        st.markdown("*Below is the complete dataset with all measurements. You can sort by clicking column headers.*")
        st.dataframe(df, use_container_width=True, height=400)
        st.caption(f"Showing all {len(df)} samples with {len(df.columns)-1} features")
    
    with tab2:
        st.markdown("### 📊 Statistical Analysis")
        st.markdown("*This table shows the count, mean, standard deviation, min, max, and quartiles for each feature.*")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("### 🌸 Species Distribution Analysis")
        st.markdown("*The dataset is perfectly balanced with 50 samples of each species.*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(df, names='Species', title='Species Distribution',
                        color_discrete_sequence=['#667eea', '#4facfe', '#43e97b'])
            fig.update_layout(
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                font={'color': "#2d3748"},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(df['Species'].value_counts().to_frame('Count'), use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Filter by Criteria")
        st.markdown("*Use the controls below to filter flowers by species and measurement ranges.*")
        
        selected_species = st.multiselect(
            "Select Species:",
            df['Species'].unique(),
            default=df['Species'].unique()
        )
        
        col1, col2 = st.columns(2)
        with col1:
            sepal_length_range = st.slider(
                "Sepal Length (cm)",
                float(df['SepalLengthCm'].min()),
                float(df['SepalLengthCm'].max()),
                (float(df['SepalLengthCm'].min()), float(df['SepalLengthCm'].max()))
            )
        
        with col2:
            petal_length_range = st.slider(
                "Petal Length (cm)",
                float(df['PetalLengthCm'].min()),
                float(df['PetalLengthCm'].max()),
                (float(df['PetalLengthCm'].min()), float(df['PetalLengthCm'].max()))
            )
        
        filtered_df = df[
            (df['Species'].isin(selected_species)) &
            (df['SepalLengthCm'].between(*sepal_length_range)) &
            (df['PetalLengthCm'].between(*petal_length_range))
        ]
        
        st.metric("Filtered Samples", len(filtered_df))
        st.dataframe(filtered_df, use_container_width=True, height=300)

elif page == "🎨 3D Visualizations":
    st.markdown("<h1>🎨 <span class='gradient-text'>3D Visualizations</span></h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌸 Main 3D Plot", "🎯 PCA View", "📊 Feature Pairs"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("**Plot Controls**")
            x_axis = st.selectbox("X-Axis", df.columns[:-1], index=0)
            y_axis = st.selectbox("Y-Axis", df.columns[:-1], index=2)
            z_axis = st.selectbox("Z-Axis", df.columns[:-1], index=3)
            marker_size = st.slider("Marker Size", 3, 15, 8)
        
        with col1:
            species_colors = {
                'Iris-setosa': '#667eea',
                'Iris-versicolor': '#4facfe',
                'Iris-virginica': '#43e97b'
            }
            
            fig = go.Figure()
            
            for species in df['Species'].unique():
                species_data = df[df['Species'] == species]
                fig.add_trace(go.Scatter3d(
                    x=species_data[x_axis],
                    y=species_data[y_axis],
                    z=species_data[z_axis],
                    mode='markers',
                    name=species.replace('Iris-', '').title(),
                    marker=dict(
                        size=marker_size,
                        color=species_colors[species],
                        opacity=0.8,
                        line=dict(color='rgba(0, 0, 0, 0.2)', width=1)
                    ),
                    hovertemplate=f"<b>{species.replace('Iris-', '').title()}</b><br>{x_axis}: %{{x:.2f}}<br>{y_axis}: %{{y:.2f}}<br>{z_axis}: %{{z:.2f}}<extra></extra>"
                ))
            
            fig.update_layout(
                title="3D Iris Feature Space",
                scene=dict(
                    xaxis_title=x_axis.replace('Cm', ' (cm)'),
                    yaxis_title=y_axis.replace('Cm', ' (cm)'),
                    zaxis_title=z_axis.replace('Cm', ' (cm)'),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    xaxis=dict(backgroundcolor='rgba(240, 240, 240, 1)', gridcolor='rgba(200, 200, 200, 0.5)'),
                    yaxis=dict(backgroundcolor='rgba(240, 240, 240, 1)', gridcolor='rgba(200, 200, 200, 0.5)'),
                    zaxis=dict(backgroundcolor='rgba(240, 240, 240, 1)', gridcolor='rgba(200, 200, 200, 0.5)'),
                ),
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                font={'color': "#2d3748"},
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Principal Component Analysis (PCA) Visualization")
        
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(df[df.columns[:-1]])
        
        fig = go.Figure()
        
        for idx, species in enumerate(df['Species'].unique()):
            mask = df['Species'] == species
            fig.add_trace(go.Scatter3d(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                z=X_pca[mask, 2],
                mode='markers',
                name=species.replace('Iris-', '').title(),
                marker=dict(
                    size=8,
                    color=list(species_colors.values())[idx],
                    opacity=0.8
                )
            ))
        
        fig.update_layout(
            title="PCA: 3D Projection",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
                bgcolor='rgba(255, 255, 255, 0.9)'
            ),
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            font={'color': "#2d3748"},
            height=550
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"📊 **Total Variance Explained:** {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    with tab3:
        st.markdown("#### Feature Pair Scatter Matrix")
        
        fig = px.scatter_matrix(
            df,
            dimensions=df.columns[:-1],
            color='Species',
            color_discrete_map={
                'Iris-setosa': '#667eea',
                'Iris-versicolor': '#4facfe',
                'Iris-virginica': '#43e97b'
            },
            height=700
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            font={'color': "#ffffff", 'size': 10}
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 ML Models":
    st.markdown("<h1>🤖 <span class='gradient-text'>Machine Learning Models</span></h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🎯 Confusion Matrices", "📈 Performance Metrics"])
    
    with tab1:
        st.markdown("### Model Accuracy Comparison")
        
        results = pd.DataFrame({
            'Model': ['Logistic Regression', 'SVM (RBF)', 'Random Forest'],
            'Accuracy': [models['lr'][2], models['svm'][2], models['rf'][2]],
            'Type': ['Linear', 'Kernel', 'Ensemble']
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            colors = ['#667eea', '#4facfe', '#43e97b']
            
            fig.add_trace(go.Bar(
                x=results['Model'],
                y=results['Accuracy'] * 100,
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0, 0, 0, 0.2)', width=2)
                ),
                text=[f"{acc*100:.2f}%" for acc in results['Accuracy']],
                textposition='outside',
                textfont=dict(color='#2d3748', size=14),
                hovertemplate="Model: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>"
            ))
            
            fig.update_layout(
                title="Model Accuracy Comparison",
                yaxis_title="Accuracy (%)",
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                plot_bgcolor='rgba(255, 255, 255, 1)',
                font={'color': "#2d3748"},
                height=400,
                yaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Best Model")
            best_model_idx = results['Accuracy'].idxmax()
            best_model_name = results.loc[best_model_idx, 'Model']
            best_accuracy = results.loc[best_model_idx, 'Accuracy']
            
            st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <div style='color: #4a5568; font-size: 0.9rem;'>Winner</div>
                    <div style='color: #43e97b; font-size: 1.8rem; font-weight: bold;'>{best_model_name}</div>
                    <div style='color: #2d3748; font-size: 2rem; font-weight: bold; margin: 1rem 0;'>{best_accuracy*100:.2f}%</div>
                    <div style='color: #4a5568; font-size: 0.85rem;'>Test Accuracy</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(results, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Confusion Matrices")
        
        cols = st.columns(3)
        model_names = ['Logistic Regression', 'SVM (RBF)', 'Random Forest']
        model_keys = ['lr', 'svm', 'rf']
        
        for col, name, key in zip(cols, model_names, model_keys):
            with col:
                cm = confusion_matrix(y_test, models[key][1])
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Setosa', 'Versicolor', 'Virginica'],
                    y=['Setosa', 'Versicolor', 'Virginica'],
                    colorscale='Viridis',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 14}
                ))
                
                fig.update_layout(
                    title=name,
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    paper_bgcolor='rgba(255, 255, 255, 0.9)',
                    font={'color': "#2d3748"},
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Detailed Performance Metrics")
        
        selected_model = st.selectbox("Select Model:", model_names)
        model_key = model_keys[model_names.index(selected_model)]
        
        report = classification_report(y_test, models[model_key][1], output_dict=True)
        
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df = metrics_df[metrics_df.index.str.startswith('Iris')]
        
        st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)
        
        # Metrics visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precision", f"{report['weighted avg']['precision']:.3f}")
        with col2:
            st.metric("Recall", f"{report['weighted avg']['recall']:.3f}")
        with col3:
            st.metric("F1-Score", f"{report['weighted avg']['f1-score']:.3f}")

elif page == "🔮 Predictions":
    st.markdown("<h1>🔮 <span class='gradient-text'>Interactive Predictions</span></h1>", unsafe_allow_html=True)
    
    st.markdown("### Input Flower Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    
    st.markdown("---")
    st.markdown("### Predictions from All Models")
    
    cols = st.columns(3)
    
    # Logistic Regression
    with cols[0]:
        lr_pred = models['lr'][0].predict(input_scaled)[0]
        lr_proba = models['lr'][0].predict_proba(input_scaled)[0]
        
        st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div style='color: #4a5568; font-size: 0.9rem;'>Logistic Regression</div>
                <div style='color: #667eea; font-size: 1.5rem; font-weight: bold; margin: 1rem 0;'>
                    {lr_pred.replace('Iris-', '').title()}
                </div>
                <div style='color: #2d3748; font-size: 1.2rem;'>{lr_proba.max()*100:.1f}% confident</div>
            </div>
        """, unsafe_allow_html=True)
    
    # SVM
    with cols[1]:
        svm_pred = models['svm'][0].predict(input_scaled)[0]
        svm_proba = models['svm'][0].predict_proba(input_scaled)[0]
        
        st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div style='color: #4a5568; font-size: 0.9rem;'>SVM (RBF)</div>
                <div style='color: #4facfe; font-size: 1.5rem; font-weight: bold; margin: 1rem 0;'>
                    {svm_pred.replace('Iris-', '').title()}
                </div>
                <div style='color: #2d3748; font-size: 1.2rem;'>{svm_proba.max()*100:.1f}% confident</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Random Forest
    with cols[2]:
        rf_pred = models['rf'][0].predict(input_data)[0]
        rf_proba = models['rf'][0].predict_proba(input_data)[0]
        
        st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div style='color: #4a5568; font-size: 0.9rem;'>Random Forest</div>
                <div style='color: #43e97b; font-size: 1.5rem; font-weight: bold; margin: 1rem 0;'>
                    {rf_pred.replace('Iris-', '').title()}
                </div>
                <div style='color: #2d3748; font-size: 1.2rem;'>{rf_proba.max()*100:.1f}% confident</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Probability bars
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Prediction Probabilities")
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Logistic Regression', x=species, y=lr_proba*100, marker_color='#667eea'))
    fig.add_trace(go.Bar(name='SVM', x=species, y=svm_proba*100, marker_color='#4facfe'))
    fig.add_trace(go.Bar(name='Random Forest', x=species, y=rf_proba*100, marker_color='#43e97b'))
    
    fig.update_layout(
        barmode='group',
        title="Model Confidence Comparison",
        yaxis_title="Probability (%)",
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        plot_bgcolor='rgba(255, 255, 255, 1)',
        font={'color': "#2d3748"},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Feature Analysis":
    st.markdown("<h1>📈 <span class='gradient-text'>Feature Analysis</span></h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Feature Importance", "📊 Feature Ranges", "🔍 Species Comparison"])
    
    with tab1:
        st.markdown("### Random Forest Feature Importance")
        
        feature_importance = models['rf'][0].feature_importances_
        features = df.columns[:-1]
        
        imp_df = pd.DataFrame({
            'Feature': [f.replace('Cm', '') for f in features],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=imp_df['Feature'],
            x=imp_df['Importance'] * 100,
            orientation='h',
            marker=dict(
                color=imp_df['Importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f"{i*100:.1f}%" for i in imp_df['Importance']],
            textposition='outside',
            hovertemplate="Feature: %{y}<br>Importance: %{x:.2f}%<extra></extra>"
        ))
        
        fig.update_layout(
            title="Feature Importance in Classification",
            xaxis_title="Importance (%)",
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            font={'color': "#2d3748"},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("💡 **Insight:** Petal measurements (length and width) are the most important features for classification!")
    
    with tab2:
        st.markdown("### Feature Ranges by Species")
        
        selected_feature = st.selectbox("Select Feature:", df.columns[:-1])
        
        fig = go.Figure()
        
        colors = ['#667eea', '#4facfe', '#43e97b']
        
        for idx, species in enumerate(df['Species'].unique()):
            species_data = df[df['Species'] == species][selected_feature]
            
            fig.add_trace(go.Violin(
                y=species_data,
                name=species.replace('Iris-', '').title(),
                box_visible=True,
                meanline_visible=True,
                marker=dict(color=colors[idx]),
                fillcolor=colors[idx],
                opacity=0.6
            ))
        
        fig.update_layout(
            title=f"{selected_feature.replace('Cm', '')} Distribution by Species",
            yaxis_title=selected_feature.replace('Cm', ' (cm)'),
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            font={'color': "#2d3748"},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Species Feature Comparison")
        
        comparison_df = df.groupby('Species')[df.columns[:-1]].mean()
        
        fig = go.Figure()
        
        for idx, species in enumerate(comparison_df.index):
            fig.add_trace(go.Scatterpolar(
                r=comparison_df.loc[species].values,
                theta=[f.replace('Cm', '') for f in comparison_df.columns],
                fill='toself',
                name=species.replace('Iris-', '').title(),
                marker=dict(color=colors[idx])
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, comparison_df.values.max()])
            ),
            title="Average Feature Values by Species",
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            font={'color': "#2d3748"},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #4a5568; font-size: 0.85rem; padding: 1rem 0;'>
        <p>🌸 Iris ML Dashboard | Built with Streamlit & Scikit-learn</p>
    </div>
""", unsafe_allow_html=True)
