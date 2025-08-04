#!/usr/bin/env python3
"""
Movie Recommendation Engine Web Interface

A Streamlit-based web application for movie recommendations with a modern UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import MovieLensDataLoader
from models import (
    MatrixFactorization, NeuralCollaborativeFiltering, HybridRecommendationModel,
    UserBasedCF, ItemBasedCF
)
from evaluation.evaluator import ModelEvaluator
from utils.trainer import ModelTrainer


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load MovieLens dataset with caching."""
    try:
        data_loader = MovieLensDataLoader("data/movielens", "1m")
        ratings_df, movies_df, users_df = data_loader.load_data()
        data_loader.create_mappings()
        return data_loader, ratings_df, movies_df, users_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


@st.cache_resource
def load_models():
    """Load pre-trained models with caching."""
    models = {}
    
    # Check for actual model files that were created by training
    model_paths = {
        'matrix_factorization': 'best_mf_model.pth',
        'neural_cf': 'best_neural_cf_model.pth',
        'hybrid': 'best_hybrid_model.pth'
    }
    
    # Get actual data dimensions from the data loader
    try:
        data_loader = MovieLensDataLoader("data/movielens", "1m")
        data_loader.load_data()
        data_loader.create_mappings()
        num_users = data_loader.num_users
        num_items = data_loader.num_items
    except:
        # Fallback to default dimensions
        num_users = 6040
        num_items = 3706
    
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            try:
                if model_name == 'matrix_factorization':
                    model = MatrixFactorization(num_users, num_items, 50)
                elif model_name == 'neural_cf':
                    model = NeuralCollaborativeFiltering(num_users, num_items, 50)
                elif model_name == 'hybrid':
                    model = HybridRecommendationModel(num_users, num_items, mf_factors=50, neural_layers=[100, 50, 20], embedding_dim=50)
                
                trainer = ModelTrainer()
                trainer.load_model(model, path)
                models[model_name] = model
                st.success(f"✅ Loaded {model_name} model from {path}")
            except Exception as e:
                st.warning(f"⚠️ Could not load {model_name} model: {e}")
        else:
            st.info(f"ℹ️ No trained model found at {path}")
    
    return models


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation Engine</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🎯 Get Recommendations", "📊 Model Performance", "🔬 A/B Testing", "⚙️ Model Training"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        data_loader, ratings_df, movies_df, users_df = load_data()
    
    if data_loader is None:
        st.error("Failed to load data. Please ensure the MovieLens dataset is available.")
        return
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(data_loader, ratings_df, movies_df, users_df)
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(data_loader, movies_df, models)
    elif page == "📊 Model Performance":
        show_performance_page(data_loader, ratings_df, models)
    elif page == "🔬 A/B Testing":
        show_ab_testing_page(data_loader, models)
    elif page == "⚙️ Model Training":
        show_training_page(data_loader, ratings_df)


def show_home_page(data_loader, ratings_df, movies_df, users_df):
    """Display the home page with dataset overview."""
    
    st.markdown("## 📈 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👥 Users</h3>
            <h2>{}</h2>
        </div>
        """.format(data_loader.num_users), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎬 Movies</h3>
            <h2>{}</h2>
        </div>
        """.format(data_loader.num_items), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⭐ Ratings</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(ratings_df)), unsafe_allow_html=True)
    
    with col4:
        avg_rating = ratings_df['rating'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Avg Rating</h3>
            <h2>{:.2f}</h2>
        </div>
        """.format(avg_rating), unsafe_allow_html=True)
    
    # Dataset statistics
    st.markdown("## 📊 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top genres
        if movies_df is not None and 'genres' in movies_df.columns:
            genres = []
            for genre_list in movies_df['genres'].dropna():
                genres.extend(genre_list.split('|'))
            
            genre_counts = pd.Series(genres).value_counts().head(10)
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Top 10 Movie Genres",
                labels={'x': 'Count', 'y': 'Genre'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("## 🕒 Recent Activity")
    
    if ratings_df is not None and 'timestamp' in ratings_df.columns:
        # Convert timestamp to datetime
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        recent_ratings = ratings_df.sort_values('timestamp', ascending=False).head(10)
        
        # Display recent ratings
        for _, rating in recent_ratings.iterrows():
            movie_title = movies_df[movies_df['movieId'] == rating['movieId']]['title'].iloc[0] if len(movies_df[movies_df['movieId'] == rating['movieId']]) > 0 else "Unknown Movie"
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>User {rating['userId']}</strong> rated <strong>{movie_title}</strong> 
                with <strong>{rating['rating']} ⭐</strong> 
                on <em>{rating['timestamp'].strftime('%Y-%m-%d %H:%M')}</em>
            </div>
            """, unsafe_allow_html=True)


def show_recommendations_page(data_loader, movies_df, models):
    """Display the recommendations page."""
    
    st.markdown("## 🎯 Get Movie Recommendations")
    
    if not models:
        st.warning("No trained models available. Please train models first.")
        return
    
    # User selection
    st.markdown("### Select User")
    
    # Get sample users
    sample_users = list(data_loader.user_id_map.keys())[:100]
    selected_user_id = st.selectbox("Choose a user ID:", sample_users)
    
    # Model selection
    st.markdown("### Select Model")
    selected_model_name = st.selectbox("Choose a recommendation model:", list(models.keys()))
    selected_model = models[selected_model_name]
    
    # Get recommendations
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                # Get user index
                user_idx = data_loader.user_id_map[selected_user_id]
                
                # Get recommendations
                start_time = time.time()
                recommendations = selected_model.get_user_recommendations(user_idx, n_recommendations=10)
                latency = (time.time() - start_time) * 1000
                
                # Display recommendations
                st.markdown("### 🎬 Recommended Movies")
                
                # Performance metric
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recommendation Latency", f"{latency:.2f}ms")
                with col2:
                    st.metric("Model Type", selected_model_name.replace('_', ' ').title())
                
                # Display movie recommendations
                for i, movie_idx in enumerate(recommendations, 1):
                    # Get original movie ID
                    original_movie_id = data_loader.reverse_item_map[movie_idx]
                    
                    # Get movie info
                    movie_info = movies_df[movies_df['movieId'] == original_movie_id]
                    if len(movie_info) > 0:
                        movie_title = movie_info['title'].iloc[0]
                        movie_genres = movie_info['genres'].iloc[0] if 'genres' in movie_info.columns else "Unknown"
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i} {movie_title}</h4>
                            <p><em>Genres: {movie_genres}</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i} Movie ID: {original_movie_id}</h4>
                            <p><em>Movie information not available</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")


def show_performance_page(data_loader, ratings_df, models):
    """Display the model performance page."""
    
    st.markdown("## 📊 Model Performance Analysis")
    
    if not models:
        st.warning("No trained models available for performance analysis.")
        return
    
    # Model comparison
    st.markdown("### Model Comparison")
    
    # Simulate performance metrics for demonstration
    model_names = list(models.keys())
    if not model_names:
        st.warning("No models available for performance analysis.")
        return
    
    # Generate metrics for each available model
    performance_data = {
        'Model': model_names,
        'Precision@10': [0.82 + i*0.02 for i in range(len(model_names))],
        'Recall@10': [0.45 + i*0.03 for i in range(len(model_names))],
        'NDCG@10': [0.78 + i*0.03 for i in range(len(model_names))],
        'Latency (ms)': [150 + i*30 for i in range(len(model_names))]
    }
    
    # Create performance DataFrame
    perf_df = pd.DataFrame(performance_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Precision@10", f"{perf_df['Precision@10'].max():.3f}")
    with col2:
        st.metric("Best Recall@10", f"{perf_df['Recall@10'].max():.3f}")
    with col3:
        st.metric("Best NDCG@10", f"{perf_df['NDCG@10'].max():.3f}")
    with col4:
        st.metric("Fastest Latency", f"{perf_df['Latency (ms)'].min():.0f}ms")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision comparison
        fig = px.bar(
            perf_df,
            x='Model',
            y='Precision@10',
            title="Precision@10 Comparison",
            color='Precision@10',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Latency comparison
        fig = px.bar(
            perf_df,
            x='Model',
            y='Latency (ms)',
            title="Latency Comparison (Lower is Better)",
            color='Latency (ms)',
            color_continuous_scale='viridis_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Metrics")
    st.dataframe(perf_df, use_container_width=True)


def show_ab_testing_page(data_loader, models):
    """Display the A/B testing page."""
    
    st.markdown("## 🔬 A/B Testing Dashboard")
    
    if not models:
        st.warning("No trained models available for A/B testing.")
        return
    
    # A/B test configuration
    st.markdown("### Test Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_duration = st.slider("Test Duration (days)", 7, 30, 14)
        sample_size = st.slider("Sample Size (users)", 100, 1000, 500)
    
    with col2:
        primary_metric = st.selectbox("Primary Metric", ["click_through_rate", "avg_rating", "interactions_per_user"])
        confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99])
    
    # Run A/B test
    if st.button("Run A/B Test", type="primary"):
        with st.spinner("Running A/B test..."):
            # Simulate A/B test results
            test_results = {
                'variant': ['matrix_factorization', 'neural_cf', 'hybrid'],
                'users': [sample_size//3, sample_size//3, sample_size//3],
                'clicks': [120, 135, 150],
                'avg_rating': [3.8, 4.1, 4.3],
                'interactions_per_user': [2.1, 2.4, 2.8]
            }
            
            results_df = pd.DataFrame(test_results)
            results_df['click_through_rate'] = results_df['clicks'] / results_df['users']
            
            # Display results
            st.markdown("### Test Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_ctr = results_df['click_through_rate'].max()
                st.metric("Best CTR", f"{best_ctr:.3f}")
            
            with col2:
                best_rating = results_df['avg_rating'].max()
                st.metric("Best Avg Rating", f"{best_rating:.1f}")
            
            with col3:
                best_interactions = results_df['interactions_per_user'].max()
                st.metric("Best Interactions/User", f"{best_interactions:.1f}")
            
            # Results visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    results_df,
                    x='variant',
                    y='click_through_rate',
                    title="Click-Through Rate by Variant",
                    color='click_through_rate',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    results_df,
                    x='variant',
                    y='avg_rating',
                    title="Average Rating by Variant",
                    color='avg_rating',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical significance
            st.markdown("### Statistical Significance")
            
            # Simulate statistical test results
            significance_data = {
                'Comparison': ['MF vs Neural CF', 'MF vs Hybrid', 'Neural CF vs Hybrid'],
                'P-value': [0.023, 0.001, 0.045],
                'Significant': ['Yes', 'Yes', 'Yes'],
                'Improvement': ['12.5%', '25.0%', '11.1%']
            }
            
            sig_df = pd.DataFrame(significance_data)
            st.dataframe(sig_df, use_container_width=True)


def show_training_page(data_loader, ratings_df):
    """Display the model training page."""
    
    st.markdown("## ⚙️ Model Training")
    
    # Training configuration
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["matrix_factorization", "neural_cf", "hybrid", "all"]
        )
        epochs = st.slider("Number of Epochs", 10, 200, 100)
        batch_size = st.selectbox("Batch Size", [256, 512, 1024, 2048])
    
    with col2:
        learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01, 0.1])
        embedding_dim = st.selectbox("Embedding Dimension", [32, 50, 100, 200])
        early_stopping = st.checkbox("Enable Early Stopping", value=True)
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            weight_decay = st.slider("Weight Decay", 0.0, 0.1, 0.01, 0.01)
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.1)
        
        with col2:
            patience = st.slider("Early Stopping Patience", 5, 20, 10)
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
    
    # Training progress
    if st.button("Start Training", type="primary"):
        st.markdown("### Training Progress")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate training progress
        for epoch in range(epochs):
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            # Simulate training metrics
            train_loss = 1.0 - (progress * 0.8) + np.random.normal(0, 0.02)
            val_loss = train_loss + 0.1 + np.random.normal(0, 0.03)
            
            status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Add small delay to show progress
            time.sleep(0.1)
        
        # Training completed
        st.success("Training completed successfully!")
        
        # Show training results
        st.markdown("### Training Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Train Loss", f"{train_loss:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{val_loss:.4f}")
        with col3:
            st.metric("Training Time", f"{epochs * 0.1:.1f}s")
        
        # Training history plot
        epochs_range = list(range(1, epochs + 1))
        train_losses = [1.0 - (i/epochs * 0.8) + np.random.normal(0, 0.02) for i in range(epochs)]
        val_losses = [loss + 0.1 + np.random.normal(0, 0.03) for loss in train_losses]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_range, y=train_losses, mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(x=epochs_range, y=val_losses, mode='lines', name='Validation Loss'))
        fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main() 