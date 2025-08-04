#!/usr/bin/env python3
"""
Movie Recommendation Engine Training Script

This script trains various recommendation models including:
- Matrix Factorization
- Neural Collaborative Filtering  
- Hybrid Models
- Traditional Collaborative Filtering

Usage:
    python train.py --data_path data/movielens --model_type hybrid --epochs 100
"""

import argparse
import os
import sys
import time
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.data_loader import MovieLensDataLoader, DataProcessor
from models import (
    MatrixFactorization, NeuralCollaborativeFiltering, HybridRecommendationModel,
    UserBasedCF, ItemBasedCF, SVDMatrixFactorization, NMFMatrixFactorization
)
from evaluation.evaluator import ModelEvaluator
from evaluation.ab_testing import ABTestingFramework
from utils.trainer import ModelTrainer


def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Train Movie Recommendation Models')
    
    parser.add_argument('--data_path', type=str, default='data/movielens',
                       help='Path to MovieLens dataset')
    parser.add_argument('--dataset_size', type=str, default='1m',
                       choices=['100k', '1m', '10m', '25m'],
                       help='MovieLens dataset size')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['mf', 'neural_cf', 'hybrid', 'user_cf', 'item_cf', 'svd', 'nmf', 'all'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='Embedding dimension for neural models')
    parser.add_argument('--num_factors', type=int, default=50,
                       help='Number of factors for matrix factorization')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--ab_test', action='store_true',
                       help='Run A/B testing')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                       help='Run hyperparameter optimization')
    
    return parser.parse_args()


def load_and_prepare_data(args):
    
    print("Loading MovieLens dataset...")
    
    # Initialize data loader
    data_loader = MovieLensDataLoader(args.data_path, args.dataset_size)
    
    # Load data
    ratings_df, movies_df, users_df = data_loader.load_data()
    
    # Create mappings
    data_loader.create_mappings()
    
    # Split data
    train_df, val_df, test_df = data_loader.split_data()
    
    print(f"Dataset loaded: {data_loader.num_users} users, {data_loader.num_items} items")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return data_loader, train_df, val_df, test_df, movies_df, users_df


def create_data_loaders(train_df, val_df, test_df, data_loader, args):
    
    print("Creating data loaders...")
    
    processor = DataProcessor(batch_size=args.batch_size)
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        train_df, val_df, test_df, data_loader
    )
    
    return train_loader, val_loader, test_loader


def create_models(data_loader, args):
    
    models = {}
    
    if args.model_type in ['mf', 'all']:
        print("Creating Matrix Factorization model...")
        models['matrix_factorization'] = MatrixFactorization(
            num_users=data_loader.num_users,
            num_items=data_loader.num_items,
            num_factors=args.num_factors
        )
    
    if args.model_type in ['neural_cf', 'all']:
        print("Creating Neural Collaborative Filtering model...")
        models['neural_cf'] = NeuralCollaborativeFiltering(
            num_users=data_loader.num_users,
            num_items=data_loader.num_items,
            embedding_dim=args.embedding_dim
        )
    
    if args.model_type in ['hybrid', 'all']:
        print("Creating Hybrid Recommendation model...")
        models['hybrid'] = HybridRecommendationModel(
            num_users=data_loader.num_users,
            num_items=data_loader.num_items,
            mf_factors=args.num_factors,
            neural_layers=[100, 50, 20],
            embedding_dim=args.embedding_dim
        )
    
    if args.model_type in ['user_cf', 'all']:
        print("Creating User-based Collaborative Filtering model...")
        models['user_cf'] = UserBasedCF()
    
    if args.model_type in ['item_cf', 'all']:
        print("Creating Item-based Collaborative Filtering model...")
        models['item_cf'] = ItemBasedCF()
    
    if args.model_type in ['svd', 'all']:
        print("Creating SVD Matrix Factorization model...")
        models['svd'] = SVDMatrixFactorization(n_factors=args.num_factors)
    
    if args.model_type in ['nmf', 'all']:
        print("Creating NMF Matrix Factorization model...")
        models['nmf'] = NMFMatrixFactorization(n_factors=args.num_factors)
    
    return models


def train_models(models, train_loader, val_loader, args):
    
    trainer = ModelTrainer()
    training_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        if model_name in ['matrix_factorization', 'neural_cf', 'hybrid']:
            # PyTorch models
            if model_name == 'matrix_factorization':
                history = trainer.train_matrix_factorization(
                    model, train_loader, val_loader,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    early_stopping_patience=args.early_stopping_patience
                )
            elif model_name == 'neural_cf':
                history = trainer.train_neural_cf(
                    model, train_loader, val_loader,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    early_stopping_patience=args.early_stopping_patience
                )
            elif model_name == 'hybrid':
                history = trainer.train_hybrid_model(
                    model, train_loader, val_loader,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    early_stopping_patience=args.early_stopping_patience
                )
            
            training_results[model_name] = history
            
        elif model_name in ['user_cf', 'item_cf', 'svd', 'nmf']:
            # Traditional models - fit directly
            print(f"Fitting {model_name}...")
            
            # Get ratings matrix for traditional models
            ratings_matrix = data_loader.get_ratings_matrix(sparse=False)
            
            if model_name in ['user_cf', 'item_cf']:
                model.fit(ratings_matrix)
            elif model_name == 'svd':
                model.fit(ratings_matrix)
            elif model_name == 'nmf':
                model.fit(ratings_matrix)
            
            training_results[model_name] = {'status': 'fitted'}
        
        training_time = time.time() - start_time
        print(f"{model_name} training completed in {training_time:.2f} seconds")
        
        # Save model if requested
        if args.save_model:
            model_path = f"models/{model_name}_model.pth"
            os.makedirs("models", exist_ok=True)
            
            if hasattr(model, 'state_dict'):
                trainer.save_model(model, model_path)
            else:
                # For non-PyTorch models, save using joblib
                import joblib
                joblib.dump(model, model_path.replace('.pth', '.pkl'))
    
    return trainer, training_results


def evaluate_models(models, test_df, data_loader, args):
    
    if not args.evaluate:
        return
    
    print("\n" + "="*50)
    print("Evaluating models...")
    print("="*50)
    
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        try:
            results = evaluator.evaluate_model(model, test_df, data_loader)
            print(f"{model_name} evaluation results: {results}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Compare models if multiple were trained
    if len(models) > 1:
        print("\nComparing models...")
        comparison_df = evaluator.compare_models(models, test_df, data_loader)
        
        # Save comparison results
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print("Model comparison results saved to model_comparison_results.csv")
        
        # Plot comparison results
        evaluator.plot_comparison_results(comparison_df)


def run_ab_testing(models, data_loader, args):
    
    if not args.ab_test:
        return
    
    print("\n" + "="*50)
    print("Setting up A/B testing...")
    print("="*50)
    
    ab_framework = ABTestingFramework()
    
    # Create recommendation functions for each model
    def create_recommendation_function(model, model_name):
        def recommend(user_id, user_context=None):
            try:
                if hasattr(model, 'get_user_recommendations'):
                    return model.get_user_recommendations(user_id, n_recommendations=10)
                else:
                    # For traditional models, convert user_id to index
                    if hasattr(data_loader, 'user_id_map') and user_id in data_loader.user_id_map:
                        user_idx = data_loader.user_id_map[user_id]
                        return model.get_user_recommendations(user_idx, n_recommendations=10)
                    return []
            except Exception as e:
                print(f"Error getting recommendations for {model_name}: {e}")
                return []
        return recommend
    
    # Create experiment variants
    variants = {}
    for model_name, model in models.items():
        variants[model_name] = create_recommendation_function(model, model_name)
    
    # Create A/B test experiment
    experiment_id = ab_framework.create_experiment(
        "recommendation_comparison",
        variants,
        traffic_split={'matrix_factorization': 0.25, 'neural_cf': 0.25, 'hybrid': 0.5}
    )
    
    print(f"A/B test experiment created with ID: {experiment_id}")
    print("Experiment variants:", list(variants.keys()))
    
    # Simulate some user interactions for demonstration
    print("Simulating user interactions for A/B testing...")
    
    sample_users = list(data_loader.user_id_map.keys())[:100]
    sample_items = list(data_loader.item_id_map.keys())[:50]
    
    for user_id in sample_users:
        # Get recommendations for user
        recommendations = ab_framework.get_recommendations(user_id, experiment_id)
        
        # Simulate interactions
        for item_id in recommendations[:5]:  # Top 5 recommendations
            # Simulate click with some probability
            if np.random.random() < 0.3:  # 30% click probability
                ab_framework.record_interaction(
                    user_id, experiment_id, item_id, 
                    interaction_type='click'
                )
            
            # Simulate rating with some probability
            if np.random.random() < 0.1:  # 10% rating probability
                rating = np.random.randint(1, 6)  # Random rating 1-5
                ab_framework.record_interaction(
                    user_id, experiment_id, item_id,
                    interaction_type='rating',
                    rating=rating
                )
    
    # Calculate experiment metrics
    metrics = ab_framework.calculate_experiment_metrics(experiment_id)
    print("A/B test metrics:", metrics)
    
    # Generate report
    report = ab_framework.generate_report(experiment_id)
    print("A/B test report:", json.dumps(report, indent=2))
    
    # Save experiment data
    ab_framework.save_experiment_data('ab_test_results.json')


def run_hyperparameter_optimization(models, train_loader, val_loader, args):
    
    if not args.optimize_hyperparams:
        return
    
    print("\n" + "="*50)
    print("Running hyperparameter optimization...")
    print("="*50)
    
    trainer = ModelTrainer()
    
    # Define parameter grids for different models
    param_grids = {
        'matrix_factorization': {
            'num_factors': [20, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1],
            'reg_param': [0.01, 0.1, 1.0]
        },
        'neural_cf': {
            'embedding_dim': [32, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout': [0.1, 0.2, 0.3]
        },
        'hybrid': {
            'mf_factors': [20, 50, 100],
            'embedding_dim': [32, 50, 100],
            'learning_rate': [0.001, 0.01, 0.1]
        }
    }
    
    optimization_results = {}
    
    for model_name in ['matrix_factorization', 'neural_cf', 'hybrid']:
        if model_name in models:
            print(f"\nOptimizing {model_name}...")
            
            # Get parameter grid
            param_grid = param_grids.get(model_name, {})
            
            # Create model class
            if model_name == 'matrix_factorization':
                model_class = MatrixFactorization
            elif model_name == 'neural_cf':
                model_class = NeuralCollaborativeFiltering
            elif model_name == 'hybrid':
                model_class = HybridRecommendationModel
            
            # Run optimization
            results = trainer.hyperparameter_optimization(
                model_class, train_loader, val_loader, param_grid, num_trials=10
            )
            
            optimization_results[model_name] = results
            print(f"{model_name} optimization completed. Best params: {results['best_params']}")
    
    # Save optimization results
    with open('hyperparameter_optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=2, default=str)
    
    print("Hyperparameter optimization results saved to hyperparameter_optimization_results.json")


def main():
            
    args = parse_arguments()
    
    print("Movie Recommendation Engine Training")
    print("="*50)
    print(f"Model type: {args.model_type}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load and prepare data
        data_loader, train_df, val_df, test_df, movies_df, users_df = load_and_prepare_data(args)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, data_loader, args
        )
        
        # Create models
        models = create_models(data_loader, args)
        
        # Train models
        trainer, training_results = train_models(models, train_loader, val_loader, args)
        
        # Plot training history
        if hasattr(trainer, 'training_history') and trainer.training_history:
            trainer.plot_training_history()
        
        # Evaluate models
        evaluate_models(models, test_df, data_loader, args)
        
        # Run A/B testing
        run_ab_testing(models, data_loader, args)
        
        # Run hyperparameter optimization
        run_hyperparameter_optimization(models, train_loader, val_loader, args)
        
        # Save training history
        trainer.save_training_history('training_history.json')
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        
        # Print model summaries
        for model_name, model in models.items():
            if hasattr(trainer, 'get_model_summary'):
                summary = trainer.get_model_summary(model)
                print(f"\n{model_name} Summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 