import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional, Callable
import time
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class ABTestingFramework:
    
    
    def __init__(self, experiment_name: str = "recommendation_ab_test"):
        self.experiment_name = experiment_name
        self.experiments = {}
        self.results = {}
        self.user_assignments = {}
        self.metrics_history = []
        
    def create_experiment(self, experiment_id: str, variants: Dict[str, Callable], 
                         traffic_split: Dict[str, float] = None) -> str:
        
        if traffic_split is None:
            # Default to equal split
            num_variants = len(variants)
            traffic_split = {variant: 1.0 / num_variants for variant in variants.keys()}
        
        # Validate traffic split
        total_traffic = sum(traffic_split.values())
        if abs(total_traffic - 1.0) > 1e-6:
            raise ValueError("Traffic split must sum to 1.0")
        
        experiment = {
            'id': experiment_id,
            'variants': variants,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'status': 'active',
            'metrics': {},
            'user_assignments': {}
        }
        
        self.experiments[experiment_id] = experiment
        print(f"Created experiment {experiment_id} with variants: {list(variants.keys())}")
        
        return experiment_id
    
    def assign_user_to_variant(self, user_id: str, experiment_id: str) -> str:
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Check if user is already assigned
        if user_id in experiment['user_assignments']:
            return experiment['user_assignments'][user_id]
        
        # Assign user based on traffic split
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for variant, traffic_share in experiment['traffic_split'].items():
            cumulative_prob += traffic_share
            if rand_val <= cumulative_prob:
                experiment['user_assignments'][user_id] = variant
                return variant
        
        # Fallback to first variant
        first_variant = list(experiment['traffic_split'].keys())[0]
        experiment['user_assignments'][user_id] = first_variant
        return first_variant
    
    def get_recommendations(self, user_id: str, experiment_id: str, 
                          user_context: Dict = None) -> List[int]:

        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Assign user to variant if not already assigned
        variant = self.assign_user_to_variant(user_id, experiment_id)
        
        # Get recommendation function for the variant
        recommendation_func = experiment['variants'][variant]
        
        # Generate recommendations
        recommendations = recommendation_func(user_id, user_context)
        
        return recommendations
    
    def record_interaction(self, user_id: str, experiment_id: str, 
                          item_id: int, interaction_type: str = 'click',
                          rating: float = None, timestamp: datetime = None):
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get user's variant
        variant = self.experiments[experiment_id]['user_assignments'].get(user_id)
        if variant is None:
            return  # User not in experiment
        
        # Record interaction
        interaction = {
            'user_id': user_id,
            'experiment_id': experiment_id,
            'variant': variant,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': timestamp
        }
        
        # Store in experiment results
        if experiment_id not in self.results:
            self.results[experiment_id] = []
        
        self.results[experiment_id].append(interaction)
    
    def calculate_experiment_metrics(self, experiment_id: str, 
                                   time_window: timedelta = None) -> Dict[str, Dict[str, float]]:
        
        if experiment_id not in self.experiments or experiment_id not in self.results:
            return {}
        
        experiment = self.experiments[experiment_id]
        interactions = self.results[experiment_id]
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            interactions = [i for i in interactions if i['timestamp'] >= cutoff_time]
        
        # Group interactions by variant
        variant_interactions = {}
        for interaction in interactions:
            variant = interaction['variant']
            if variant not in variant_interactions:
                variant_interactions[variant] = []
            variant_interactions[variant].append(interaction)
        
        # Calculate metrics for each variant
        metrics = {}
        for variant, variant_data in variant_interactions.items():
            metrics[variant] = self._calculate_variant_metrics(variant_data)
        
        # Store metrics in experiment
        experiment['metrics'] = metrics
        
        return metrics
    
    def _calculate_variant_metrics(self, interactions: List[Dict]) -> Dict[str, float]:
        
        if not interactions:
            return {}
        
        # Basic metrics
        total_interactions = len(interactions)
        unique_users = len(set(i['user_id'] for i in interactions))
        
        # Interaction types
        clicks = len([i for i in interactions if i['interaction_type'] == 'click'])
        ratings = [i['rating'] for i in interactions if i['rating'] is not None]
        
        # Engagement metrics
        avg_rating = np.mean(ratings) if ratings else 0.0
        click_through_rate = clicks / total_interactions if total_interactions > 0 else 0.0
        
        # User engagement (interactions per user)
        interactions_per_user = total_interactions / unique_users if unique_users > 0 else 0.0
        
        return {
            'total_interactions': total_interactions,
            'unique_users': unique_users,
            'clicks': clicks,
            'avg_rating': avg_rating,
            'click_through_rate': click_through_rate,
            'interactions_per_user': interactions_per_user
        }
    
    def run_statistical_test(self, experiment_id: str, metric: str = 'click_through_rate') -> Dict[str, float]:
        
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        metrics = experiment.get('metrics', {})
        
        if len(metrics) < 2:
            return {}
        
        # Get metric values for each variant
        variant_values = {}
        for variant, variant_metrics in metrics.items():
            if metric in variant_metrics:
                variant_values[variant] = variant_metrics[metric]
        
        if len(variant_values) < 2:
            return {}
        
        # Perform t-test between variants
        variants = list(variant_values.keys())
        results = {}
        
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                variant_a = variants[i]
                variant_b = variants[j]
                
                # Get interaction data for statistical test
                interactions_a = [i for i in self.results[experiment_id] 
                                if i['variant'] == variant_a]
                interactions_b = [i for i in self.results[experiment_id] 
                                if i['variant'] == variant_b]
                
                # Calculate metric values for each user
                user_metrics_a = self._calculate_user_metrics(interactions_a, metric)
                user_metrics_b = self._calculate_user_metrics(interactions_b, metric)
                
                if user_metrics_a and user_metrics_b:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(user_metrics_a, user_metrics_b)
                    
                    results[f'{variant_a}_vs_{variant_b}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return results
    
    def _calculate_user_metrics(self, interactions: List[Dict], metric: str) -> List[float]:

        user_metrics = {}
        
        for interaction in interactions:
            user_id = interaction['user_id']
            
            if user_id not in user_metrics:
                user_metrics[user_id] = {
                    'interactions': 0,
                    'clicks': 0,
                    'ratings': []
                }
            
            user_metrics[user_id]['interactions'] += 1
            
            if interaction['interaction_type'] == 'click':
                user_metrics[user_id]['clicks'] += 1
            
            if interaction['rating'] is not None:
                user_metrics[user_id]['ratings'].append(interaction['rating'])
        
        # Calculate metric for each user
        metric_values = []
        for user_data in user_metrics.values():
            if metric == 'click_through_rate':
                ctr = user_data['clicks'] / user_data['interactions']
                metric_values.append(ctr)
            elif metric == 'avg_rating':
                if user_data['ratings']:
                    avg_rating = np.mean(user_data['ratings'])
                    metric_values.append(avg_rating)
            elif metric == 'interactions_per_user':
                metric_values.append(user_data['interactions'])
        
        return metric_values
    
    def generate_report(self, experiment_id: str) -> Dict:
        
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        metrics = experiment.get('metrics', {})
        
        # Calculate statistical significance
        statistical_tests = self.run_statistical_test(experiment_id)
        
        # Calculate improvement metrics
        improvements = {}
        baseline_variant = list(metrics.keys())[0] if metrics else None
        
        if baseline_variant:
            baseline_metrics = metrics[baseline_variant]
            
            for variant, variant_metrics in metrics.items():
                if variant != baseline_variant:
                    variant_improvements = {}
                    for metric_name, baseline_value in baseline_metrics.items():
                        if metric_name in variant_metrics and baseline_value > 0:
                            improvement = ((variant_metrics[metric_name] - baseline_value) / baseline_value) * 100
                            variant_improvements[metric_name] = improvement
                    
                    improvements[variant] = variant_improvements
        
        report = {
            'experiment_id': experiment_id,
            'status': experiment['status'],
            'start_time': experiment['start_time'].isoformat(),
            'duration': (datetime.now() - experiment['start_time']).days,
            'traffic_split': experiment['traffic_split'],
            'metrics': metrics,
            'statistical_tests': statistical_tests,
            'improvements': improvements,
            'total_users': len(experiment['user_assignments']),
            'total_interactions': len(self.results.get(experiment_id, []))
        }
        
        return report
    
    def plot_experiment_results(self, experiment_id: str, metrics: List[str] = None):
        
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        experiment_metrics = experiment.get('metrics', {})
        
        if not experiment_metrics:
            print("No metrics available for plotting")
            return
        
        if metrics is None:
            metrics = ['click_through_rate', 'avg_rating', 'interactions_per_user']
        
        # Create subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in experiment_metrics.get(list(experiment_metrics.keys())[0], {}):
                variants = list(experiment_metrics.keys())
                values = [experiment_metrics[variant].get(metric, 0) for variant in variants]
                
                axes[i].bar(variants, values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def stop_experiment(self, experiment_id: str):
        
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = 'stopped'
            print(f"Experiment {experiment_id} stopped")
    
    def save_experiment_data(self, filepath: str):
        
        data = {
            'experiments': self.experiments,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, default=str, indent=2)
    
    def load_experiment_data(self, filepath: str):
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.experiments = data.get('experiments', {})
        self.results = data.get('results', {})


class ExperimentManager:
    
    
    def __init__(self):
        self.ab_framework = ABTestingFramework()
        self.experiment_configs = {}
    
    def create_recommendation_experiment(self, experiment_id: str, 
                                       models: Dict[str, object],
                                       traffic_split: Dict[str, float] = None) -> str:
        
        
        def create_recommendation_function(model, model_name):
            def recommend(user_id, user_context=None):
                # Get recommendations from the model
                if hasattr(model, 'get_user_recommendations'):
                    return model.get_user_recommendations(user_id, n_recommendations=10)
                else:
                    # Fallback for models without get_user_recommendations method
                    return []
            return recommend
        
        # Create recommendation functions for each model
        variants = {}
        for model_name, model in models.items():
            variants[model_name] = create_recommendation_function(model, model_name)
        
        # Create experiment
        return self.ab_framework.create_experiment(experiment_id, variants, traffic_split)
    
    def run_experiment(self, experiment_id: str, duration_days: int = 30, 
                      evaluation_interval: int = 7):
        
        print(f"Starting experiment {experiment_id} for {duration_days} days")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(days=duration_days)
        
        while datetime.now() < end_time:
            # Calculate and log metrics
            metrics = self.ab_framework.calculate_experiment_metrics(experiment_id)
            
            if metrics:
                print(f"Experiment {experiment_id} metrics at {datetime.now()}:")
                for variant, variant_metrics in metrics.items():
                    print(f"  {variant}: {variant_metrics}")
            
            # Wait for evaluation interval
            time.sleep(evaluation_interval * 24 * 3600)  # Convert days to seconds
        
        # Generate final report
        report = self.ab_framework.generate_report(experiment_id)
        print(f"Experiment {experiment_id} completed. Final report:")
        print(json.dumps(report, indent=2))
        
        return report 