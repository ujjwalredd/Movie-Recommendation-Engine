import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import time

from .matrix_factorization import MatrixFactorization
from .neural_cf import NeuralCollaborativeFiltering


class HybridRecommendationModel(nn.Module):
    
    def __init__(self, num_users: int, num_items: int, 
                 mf_factors: int = 50, neural_layers: List[int] = [100, 50, 20],
                 embedding_dim: int = 50, dropout: float = 0.1,
                 alpha: float = 0.5, learning_rate: float = 0.001):
        super(HybridRecommendationModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.mf_factors = mf_factors
        self.neural_layers = neural_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.alpha = alpha  # Weight for combining MF and Neural predictions
        self.learning_rate = learning_rate
        
        # Matrix Factorization component
        self.mf_model = MatrixFactorization(num_users, num_items, mf_factors)
        
        # Neural Collaborative Filtering component
        self.neural_model = NeuralCollaborativeFiltering(
            num_users, num_items, embedding_dim, neural_layers, dropout
        )
        
        # Fusion layer to combine predictions
        self.fusion_layer = nn.Linear(2, 1)  # Combines MF and Neural predictions
        
        # Initialize fusion layer
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
        
    def forward(self, user_indices, item_indices):
        """Forward pass combining MF and Neural predictions."""
        # Get predictions from both models
        mf_prediction = self.mf_model(user_indices, item_indices)
        neural_prediction = self.neural_model(user_indices, item_indices)
        
        # Concatenate predictions
        combined_predictions = torch.stack([mf_prediction, neural_prediction], dim=1)
        
        # Fusion layer
        final_prediction = self.fusion_layer(combined_predictions)
        
        return final_prediction.squeeze()
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a specific user-item pair."""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.forward(user_tensor, item_tensor)
            return torch.sigmoid(prediction).item() * 5.0
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Get top N recommendations for a user with sub-200ms latency."""
        start_time = time.time()
        
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items)
            item_tensor = torch.arange(self.num_items)
            predictions = self.forward(user_tensor, item_tensor)
            
            # Apply sigmoid and scale
            predictions = torch.sigmoid(predictions) * 5.0
            
            # Get top N items
            _, indices = torch.topk(predictions, n_recommendations)
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"Recommendation latency: {latency:.2f}ms")
            
            return indices.cpu().numpy().tolist()
    
    def get_embeddings(self):
        """Get user and item embeddings from both components."""
        mf_user_emb, mf_item_emb = self.mf_model.get_embeddings()
        neural_user_emb, neural_item_emb = self.neural_model.get_embeddings()
        
        return {
            'mf': (mf_user_emb, mf_item_emb),
            'neural': (neural_user_emb, neural_item_emb)
        }


class WeightedHybridModel(nn.Module):
    
    def __init__(self, num_users: int, num_items: int, 
                 mf_factors: int = 50, neural_layers: List[int] = [100, 50, 20],
                 embedding_dim: int = 50, dropout: float = 0.1,
                 learning_rate: float = 0.001):
        super(WeightedHybridModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.mf_factors = mf_factors
        self.neural_layers = neural_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Matrix Factorization component
        self.mf_model = MatrixFactorization(num_users, num_items, mf_factors)
        
        # Neural Collaborative Filtering component
        self.neural_model = NeuralCollaborativeFiltering(
            num_users, num_items, embedding_dim, neural_layers, dropout
        )
        
        # Learnable weight for combining predictions
        self.mf_weight = nn.Parameter(torch.tensor(0.5))
        self.neural_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, user_indices, item_indices):
        mf_prediction = self.mf_model(user_indices, item_indices)
        neural_prediction = self.neural_model(user_indices, item_indices)
        
        # Apply learnable weights
        weighted_prediction = (self.mf_weight * mf_prediction + 
                             self.neural_weight * neural_prediction)
        
        return weighted_prediction
    
    def predict(self, user_id: int, item_id: int) -> float:
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.forward(user_tensor, item_tensor)
            return torch.sigmoid(prediction).item() * 5.0
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        start_time = time.time()
        
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items)
            item_tensor = torch.arange(self.num_items)
            predictions = self.forward(user_tensor, item_tensor)
            
            # Apply sigmoid and scale
            predictions = torch.sigmoid(predictions) * 5.0
            
            # Get top N items
            _, indices = torch.topk(predictions, n_recommendations)
            
            latency = (time.time() - start_time) * 1000
            print(f"Recommendation latency: {latency:.2f}ms")
            
            return indices.cpu().numpy().tolist()


class EnsembleHybridModel:
    
    def __init__(self, num_users: int, num_items: int, 
                 models: List[nn.Module] = None, weights: List[float] = None):
        self.num_users = num_users
        self.num_items = num_items
        self.models = models or []
        self.weights = weights or [1.0] * len(models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def add_model(self, model: nn.Module, weight: float = 1.0):
        self.models.append(model)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Get ensemble prediction."""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model.predict(user_id, item_id)
                predictions.append(pred)
        
        # Weighted average
        ensemble_prediction = sum(p * w for p, w in zip(predictions, self.weights))
        return ensemble_prediction
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        start_time = time.time()
        
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_id] * self.num_items)
                item_tensor = torch.arange(self.num_items)
                predictions = model.forward(user_tensor, item_tensor)
                
                if hasattr(model, 'neural_model'):
                    # For hybrid models, apply sigmoid
                    predictions = torch.sigmoid(predictions) * 5.0
                
                all_predictions.append(predictions.cpu().numpy())
        
        # Weighted ensemble
        ensemble_predictions = np.zeros(self.num_items)
        for pred, weight in zip(all_predictions, self.weights):
            ensemble_predictions += pred * weight
        
        # Get top N items
        top_indices = np.argsort(ensemble_predictions)[::-1][:n_recommendations]
        
        latency = (time.time() - start_time) * 1000
        print(f"Ensemble recommendation latency: {latency:.2f}ms")
        
        return top_indices.tolist() 