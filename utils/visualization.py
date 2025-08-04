"""
Visualization utilities for the Movie Recommendation Engine.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import time


class VisualizationUtils:
    """Utility class for creating visualizations."""
    
    def __init__(self, style: str = "default"):
        """Initialize visualization utilities."""
        self.style = style
        self.set_style()
    
    def set_style(self):
        """Set matplotlib and seaborn style."""
        if self.style == "default":
            plt.style.use('default')
            sns.set_palette("husl")
        elif self.style == "dark":
            plt.style.use('dark_background')
            sns.set_palette("husl")
        elif self.style == "seaborn":
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None):
        """Plot training history."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Training and validation loss
        axes[0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(history['train_losses'], history['val_losses'])]
        axes[1].plot(epochs, loss_diff, 'g-', linewidth=2)
        axes[1].set_title('Training-Validation Loss Difference', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('|Train Loss - Val Loss|', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            metrics: List[str] = None,
                            save_path: Optional[str] = None):
        """Plot model comparison results."""
        
        if metrics is None:
            metrics = ['precision@10', 'recall@10', 'ndcg@10', 'rmse']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=sns.color_palette("husl", len(comparison_df)))
            ax.set_title(f'{metric.replace("@", " @").upper()}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace("@", " @").title(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_latency_distribution(self, latencies: List[float], 
                                model_name: str = "Model",
                                save_path: Optional[str] = None):
        """Plot latency distribution."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(latencies, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0].axvline(np.mean(latencies), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(latencies):.2f}ms')
        axes[0].set_xlabel('Latency (ms)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{model_name} Latency Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(latencies, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1].set_ylabel('Latency (ms)', fontsize=12)
        axes[1].set_title(f'{model_name} Latency Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_scalability_results(self, scalability_results: Dict[str, List[float]],
                               save_path: Optional[str] = None):
        """Plot scalability results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total time vs batch size
        axes[0, 0].plot(scalability_results['batch_size'], scalability_results['total_time'], 
                       'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Batch Size', fontsize=12)
        axes[0, 0].set_ylabel('Total Time (s)', fontsize=12)
        axes[0, 0].set_title('Total Time vs Batch Size', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average time per user vs batch size
        axes[0, 1].plot(scalability_results['batch_size'], scalability_results['avg_time_per_user'], 
                       'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Batch Size', fontsize=12)
        axes[0, 1].set_ylabel('Average Time per User (s)', fontsize=12)
        axes[0, 1].set_title('Average Time per User vs Batch Size', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput vs batch size
        axes[1, 0].plot(scalability_results['batch_size'], scalability_results['throughput'], 
                       'o-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Batch Size', fontsize=12)
        axes[1, 0].set_ylabel('Throughput (users/s)', fontsize=12)
        axes[1, 0].set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Log scale for throughput
        axes[1, 1].semilogy(scalability_results['batch_size'], scalability_results['throughput'], 
                           'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Batch Size', fontsize=12)
        axes[1, 1].set_ylabel('Throughput (users/s) - Log Scale', fontsize=12)
        axes[1, 1].set_title('Throughput vs Batch Size (Log Scale)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_ab_test_results(self, ab_results: Dict[str, Any], 
                           save_path: Optional[str] = None):
        """Plot A/B test results."""
        
        variants = list(ab_results['variants'].keys())
        metrics = ['click_through_rate', 'avg_rating', 'interactions_per_user']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [ab_results['variants'][v][metric] for v in variants]
            colors = sns.color_palette("husl", len(variants))
            
            bars = axes[i].bar(variants, values, color=colors)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', 
                            fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_rating_distribution(self, ratings: List[float], 
                               save_path: Optional[str] = None):
        """Plot rating distribution."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(ratings, bins=10, alpha=0.7, edgecolor='black', color='lightcoral')
        axes[0].set_xlabel('Rating', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(ratings, patch_artist=True, 
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[1].set_ylabel('Rating', fontsize=12)
        axes[1].set_title('Rating Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, data: Dict[str, Any]):
        """Create an interactive dashboard using Plotly."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'Training History', 
                          'Latency Distribution', 'Scalability Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Model performance comparison
        if 'model_comparison' in data:
            models = data['model_comparison']['Model']
            precision = data['model_comparison']['precision@10']
            
            fig.add_trace(
                go.Bar(x=models, y=precision, name='Precision@10'),
                row=1, col=1
            )
        
        # Training history
        if 'training_history' in data:
            epochs = list(range(1, len(data['training_history']['train_losses']) + 1))
            train_losses = data['training_history']['train_losses']
            val_losses = data['training_history']['val_losses']
            
            fig.add_trace(
                go.Scatter(x=epochs, y=train_losses, mode='lines', name='Training Loss'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=val_losses, mode='lines', name='Validation Loss'),
                row=1, col=2
            )
        
        # Latency distribution
        if 'latencies' in data:
            fig.add_trace(
                go.Histogram(x=data['latencies'], name='Latency Distribution'),
                row=2, col=1
            )
        
        # Scalability analysis
        if 'scalability' in data:
            batch_sizes = data['scalability']['batch_size']
            throughput = data['scalability']['throughput']
            
            fig.add_trace(
                go.Scatter(x=batch_sizes, y=throughput, mode='lines+markers', 
                          name='Throughput'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Movie Recommendation Engine Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def plot_embedding_visualization(self, user_embeddings: np.ndarray, 
                                   item_embeddings: np.ndarray,
                                   save_path: Optional[str] = None):
        """Visualize user and item embeddings using t-SNE."""
        
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("scikit-learn not available for t-SNE visualization")
            return
        
        # Combine embeddings
        combined_embeddings = np.vstack([user_embeddings, item_embeddings])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        
        # Split back into users and items
        n_users = user_embeddings.shape[0]
        user_embeddings_2d = embeddings_2d[:n_users]
        item_embeddings_2d = embeddings_2d[n_users:]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot users
        ax.scatter(user_embeddings_2d[:, 0], user_embeddings_2d[:, 1], 
                  c='blue', alpha=0.6, s=20, label='Users')
        
        # Plot items
        ax.scatter(item_embeddings_2d[:, 0], item_embeddings_2d[:, 1], 
                  c='red', alpha=0.6, s=20, label='Items')
        
        ax.set_title('t-SNE Visualization of User and Item Embeddings', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_hyperparameter_optimization(self, optimization_results: List[Dict[str, Any]],
                                       save_path: Optional[str] = None):
        """Plot hyperparameter optimization results."""
        
        # Extract data
        trials = [r['trial'] for r in optimization_results]
        val_losses = [r['val_loss'] for r in optimization_results]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Validation loss over trials
        axes[0].plot(trials, val_losses, 'o-', linewidth=2, markersize=6)
        axes[0].set_xlabel('Trial', fontsize=12)
        axes[0].set_ylabel('Validation Loss', fontsize=12)
        axes[0].set_title('Validation Loss Over Optimization Trials', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Best validation loss so far
        best_losses = []
        current_best = float('inf')
        for loss in val_losses:
            if loss < current_best:
                current_best = loss
            best_losses.append(current_best)
        
        axes[1].plot(trials, best_losses, 'o-', linewidth=2, markersize=6, color='green')
        axes[1].set_xlabel('Trial', fontsize=12)
        axes[1].set_ylabel('Best Validation Loss', fontsize=12)
        axes[1].set_title('Best Validation Loss Over Time', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_all_plots(self, plots_data: Dict[str, Any], output_dir: str = "plots"):
        """Save all plots to a directory."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        if 'training_history' in plots_data:
            self.plot_training_history(
                plots_data['training_history'],
                save_path=f"{output_dir}/training_history_{timestamp}.png"
            )
        
        # Save model comparison
        if 'model_comparison' in plots_data:
            self.plot_model_comparison(
                plots_data['model_comparison'],
                save_path=f"{output_dir}/model_comparison_{timestamp}.png"
            )
        
        # Save latency distribution
        if 'latencies' in plots_data:
            self.plot_latency_distribution(
                plots_data['latencies'],
                save_path=f"{output_dir}/latency_distribution_{timestamp}.png"
            )
        
        # Save scalability results
        if 'scalability' in plots_data:
            self.plot_scalability_results(
                plots_data['scalability'],
                save_path=f"{output_dir}/scalability_{timestamp}.png"
            )
        
        print(f"All plots saved to {output_dir}/") 