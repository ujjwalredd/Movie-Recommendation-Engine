import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .metrics import RecommendationMetrics


class ModelEvaluator:
    
    
    def __init__(self):
        self.metrics_calculator = RecommendationMetrics()
        self.evaluation_results = {}
        self.model_performances = {}
        
    def evaluate_model(self, model: Any, test_data: pd.DataFrame, 
                      data_loader: Any, n_recommendations: int = 10,
                      metrics: List[str] = None) -> Dict[str, float]:
        
        
        if metrics is None:
            metrics = ['precision@10', 'recall@10', 'ndcg@10', 'rmse', 'mae']
        
        print(f"Evaluating model: {type(model).__name__}")
        start_time = time.time()
        
        # Prepare test data
        test_users = test_data['userId'].unique()
        
        # Generate predictions and recommendations
        predictions = []
        true_ratings = []
        recommendations = []
        true_items = []
        
        for user_id in test_users[:100]:  # Limit for performance
            user_test_data = test_data[test_data['userId'] == user_id]
            
            # Get user index
            if hasattr(data_loader, 'user_id_map') and user_id in data_loader.user_id_map:
                user_idx = data_loader.user_id_map[user_id]
            else:
                continue
            
            # Generate recommendations
            try:
                if hasattr(model, 'get_user_recommendations'):
                    user_recs = model.get_user_recommendations(user_idx, n_recommendations)
                    recommendations.append(user_recs)
                    
                    # Get true items for this user
                    user_true_items = user_test_data['movieId'].tolist()
                    if hasattr(data_loader, 'item_id_map'):
                        user_true_items = [data_loader.item_id_map.get(item_id, item_id) 
                                         for item_id in user_true_items]
                    true_items.append(user_true_items)
                
                # Generate predictions for rated items
                for _, row in user_test_data.iterrows():
                    item_id = row['movieId']
                    true_rating = row['rating']
                    
                    if hasattr(data_loader, 'item_id_map') and item_id in data_loader.item_id_map:
                        item_idx = data_loader.item_id_map[item_id]
                    else:
                        continue
                    
                    # Get prediction
                    if hasattr(model, 'predict'):
                        pred_rating = model.predict(user_idx, item_idx)
                        predictions.append(pred_rating)
                        true_ratings.append(true_rating)
                        
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate metrics
        results = {}
        
        # Rating prediction metrics
        if predictions and true_ratings:
            results['rmse'] = self.metrics_calculator.calculate_rmse(true_ratings, predictions)
            results['mae'] = self.metrics_calculator.calculate_mae(true_ratings, predictions)
        
        # Recommendation metrics
        if recommendations and true_items:
            rec_metrics = self.metrics_calculator.evaluate_recommendations(
                true_items, recommendations, k_values=[5, 10, 20]
            )
            results.update(rec_metrics)
        
        # Performance metrics
        evaluation_time = time.time() - start_time
        results['evaluation_time'] = evaluation_time
        results['num_users_evaluated'] = len(recommendations)
        
        # Store results
        model_name = type(model).__name__
        self.evaluation_results[model_name] = results
        self.model_performances[model_name] = {
            'metrics': results,
            'predictions': predictions,
            'recommendations': recommendations,
            'evaluation_time': evaluation_time
        }
        
        print(f"Evaluation completed in {evaluation_time:.2f}s")
        print(f"Results: {results}")
        
        return results
    
    def compare_models(self, models: Dict[str, Any], test_data: pd.DataFrame, 
                      data_loader: Any, n_recommendations: int = 10) -> pd.DataFrame:
        
        
        print("Comparing models...")
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            self.evaluate_model(model, test_data, data_loader, n_recommendations)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            row = {'Model': model_name}
            row.update(results)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by precision@10 (or another key metric)
        if 'precision@10' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('precision@10', ascending=False)
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def evaluate_latency(self, model: Any, data_loader: Any, 
                        num_users: int = 100, n_recommendations: int = 10) -> Dict[str, float]:
        
        
        print(f"Evaluating latency for {type(model).__name__}")
        
        # Get sample users
        if hasattr(data_loader, 'reverse_user_map'):
            sample_users = list(data_loader.reverse_user_map.keys())[:num_users]
        else:
            sample_users = list(range(min(num_users, data_loader.num_users)))
        
        latencies = []
        
        for user_idx in sample_users:
            start_time = time.time()
            
            try:
                if hasattr(model, 'get_user_recommendations'):
                    model.get_user_recommendations(user_idx, n_recommendations)
                
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                
            except Exception as e:
                print(f"Error measuring latency for user {user_idx}: {e}")
                continue
        
        if latencies:
            latency_stats = {
                'mean_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'std_latency_ms': np.std(latencies)
            }
            
            print(f"Latency Statistics (ms): {latency_stats}")
            return latency_stats
        else:
            print("No latency measurements available")
            return {}
    
    def evaluate_scalability(self, model: Any, data_loader: Any, 
                           user_batches: List[int] = [10, 50, 100, 500]) -> Dict[str, List[float]]:
        
        
        print(f"Evaluating scalability for {type(model).__name__}")
        
        scalability_results = {
            'batch_size': [],
            'total_time': [],
            'avg_time_per_user': [],
            'throughput': []
        }
        
        for batch_size in user_batches:
            print(f"Testing batch size: {batch_size}")
            
            # Get sample users
            if hasattr(data_loader, 'reverse_user_map'):
                sample_users = list(data_loader.reverse_user_map.keys())[:batch_size]
            else:
                sample_users = list(range(min(batch_size, data_loader.num_users)))
            
            start_time = time.time()
            successful_users = 0
            
            for user_idx in sample_users:
                try:
                    if hasattr(model, 'get_user_recommendations'):
                        model.get_user_recommendations(user_idx, n_recommendations=10)
                        successful_users += 1
                except Exception as e:
                    print(f"Error in scalability test for user {user_idx}: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            # Avoid division by zero
            if successful_users > 0 and total_time > 0:
                avg_time_per_user = total_time / successful_users
                throughput = successful_users / total_time  # users per second
            else:
                avg_time_per_user = 0.0
                throughput = 0.0
            
            scalability_results['batch_size'].append(batch_size)
            scalability_results['total_time'].append(total_time)
            scalability_results['avg_time_per_user'].append(avg_time_per_user)
            scalability_results['throughput'].append(throughput)
        
        print(f"Scalability Results: {scalability_results}")
        return scalability_results
    
    def plot_comparison_results(self, comparison_df: pd.DataFrame, 
                               metrics: List[str] = None):
        
        
        if metrics is None:
            metrics = ['precision@10', 'recall@10', 'ndcg@10', 'rmse']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No metrics available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric.replace("@", " @").upper()}')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_latency_distribution(self, model: Any, data_loader: Any, 
                                num_users: int = 100):
        
        
        # Get latency measurements
        latency_stats = self.evaluate_latency(model, data_loader, num_users)
        
        if not latency_stats:
            return
        
        # Get individual latencies for plotting
        if hasattr(data_loader, 'reverse_user_map'):
            sample_users = list(data_loader.reverse_user_map.keys())[:num_users]
        else:
            sample_users = list(range(min(num_users, data_loader.num_users)))
        
        latencies = []
        for user_idx in sample_users:
            start_time = time.time()
            try:
                if hasattr(model, 'get_user_recommendations'):
                    model.get_user_recommendations(user_idx, n_recommendations=10)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            except:
                continue
        
        if latencies:
            plt.figure(figsize=(12, 5))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(latencies, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Frequency')
            plt.title('Latency Distribution')
            plt.axvline(np.mean(latencies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(latencies):.2f}ms')
            plt.legend()
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(latencies)
            plt.ylabel('Latency (ms)')
            plt.title('Latency Box Plot')
            
            plt.tight_layout()
            plt.show()
    
    def plot_scalability_results(self, scalability_results: Dict[str, List[float]]):
        
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total time vs batch size
        axes[0, 0].plot(scalability_results['batch_size'], scalability_results['total_time'], 'o-')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Total Time (s)')
        axes[0, 0].set_title('Total Time vs Batch Size')
        axes[0, 0].grid(True)
        
        # Average time per user vs batch size
        axes[0, 1].plot(scalability_results['batch_size'], scalability_results['avg_time_per_user'], 'o-')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Average Time per User (s)')
        axes[0, 1].set_title('Average Time per User vs Batch Size')
        axes[0, 1].grid(True)
        
        # Throughput vs batch size
        axes[1, 0].plot(scalability_results['batch_size'], scalability_results['throughput'], 'o-')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Throughput (users/s)')
        axes[1, 0].set_title('Throughput vs Batch Size')
        axes[1, 0].grid(True)
        
        # Log scale for throughput
        axes[1, 1].semilogy(scalability_results['batch_size'], scalability_results['throughput'], 'o-')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Throughput (users/s) - Log Scale')
        axes[1, 1].set_title('Throughput vs Batch Size (Log Scale)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, model: Any, test_data: pd.DataFrame, 
                                 data_loader: Any) -> Dict[str, Any]:
                    
        
        print("Generating comprehensive evaluation report...")
        
        # Evaluate model performance
        performance_results = self.evaluate_model(model, test_data, data_loader)
        
        # Evaluate latency
        latency_results = self.evaluate_latency(model, data_loader)
        
        # Evaluate scalability
        scalability_results = self.evaluate_scalability(model, data_loader)
        
        # Compile report
        report = {
            'model_name': type(model).__name__,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': performance_results,
            'latency_metrics': latency_results,
            'scalability_metrics': scalability_results,
            'summary': {
                'precision_at_10': performance_results.get('precision@10', 0),
                'mean_latency_ms': latency_results.get('mean_latency_ms', 0),
                'max_throughput': max(scalability_results.get('throughput', [0]))
            }
        }
        
        # Add recommendations
        if performance_results.get('precision@10', 0) >= 0.85:
            report['recommendations'] = ['Model meets precision@10 target of 85%']
        else:
            report['recommendations'] = ['Consider hyperparameter tuning to improve precision@10']
        
        if latency_results.get('mean_latency_ms', 0) <= 200:
            report['recommendations'].append('Model meets latency target of <200ms')
        else:
            report['recommendations'].append('Consider optimization to reduce latency')
        
        print("Evaluation report generated successfully")
        return report 