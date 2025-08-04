import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import Tuple, Optional
import time


class MatrixFactorization(nn.Module):
    
    def __init__(self, num_users: int, num_items: int, num_factors: int = 50, 
                 reg_param: float = 0.01, learning_rate: float = 0.001):
        super(MatrixFactorization, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.reg_param = reg_param
        self.learning_rate = learning_rate
        
        # User and item embeddings
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_factors.weight)
        nn.init.xavier_uniform_(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_factors(user_indices)
        item_embedding = self.item_factors(item_indices)
        
        user_bias = self.user_bias(user_indices).squeeze()
        item_bias = self.item_bias(item_indices).squeeze()
        
        # Compute prediction: global_bias + user_bias + item_bias + user_embedding * item_embedding
        prediction = (self.global_bias + user_bias + item_bias + 
                     torch.sum(user_embedding * item_embedding, dim=1))
        
        return prediction
    
    def get_embeddings(self):
        return (self.user_factors.weight.detach().cpu().numpy(),
                self.item_factors.weight.detach().cpu().numpy())
    
    def predict(self, user_id: int, item_id: int) -> float:
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.forward(user_tensor, item_tensor)
            return prediction.item()
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> list:
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items)
            item_tensor = torch.arange(self.num_items)
            predictions = self.forward(user_tensor, item_tensor)
            
            # Get top N items
            _, indices = torch.topk(predictions, n_recommendations)
            return indices.cpu().numpy().tolist()


class SVDMatrixFactorization:
    
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        
    def fit(self, ratings_matrix: np.ndarray):
        # Calculate global bias
        self.global_bias = np.mean(ratings_matrix[ratings_matrix != 0])
        
        # Center the data
        centered_matrix = ratings_matrix.copy()
        centered_matrix[ratings_matrix != 0] -= self.global_bias
        
        # Apply SVD
        U, sigma, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
        
        # Keep only top n_factors
        self.user_factors = U[:, :self.n_factors] * np.sqrt(sigma[:self.n_factors])
        self.item_factors = Vt[:self.n_factors, :].T * np.sqrt(sigma[:self.n_factors])
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        if self.user_factors is None:
            raise ValueError("Model must be fitted before making predictions")
        
        prediction = (self.global_bias + 
                     np.dot(self.user_factors[user_id], self.item_factors[item_id]))
        return max(1.0, min(5.0, prediction))  # Clip to rating range
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> list:
        """Get top N recommendations for a user."""
        if self.user_factors is None:
            raise ValueError("Model must be fitted before making predictions")
        
        user_predictions = (self.global_bias + 
                          np.dot(self.user_factors[user_id], self.item_factors.T))
        
        # Get top N items
        top_indices = np.argsort(user_predictions)[::-1][:n_recommendations]
        return top_indices.tolist()


class NMFMatrixFactorization:
    
    def __init__(self, n_factors: int = 50, max_iter: int = 200):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.model = NMF(n_components=n_factors, max_iter=max_iter, random_state=42)
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, ratings_matrix: np.ndarray):
        # Fill missing values with 0 for NMF
        filled_matrix = ratings_matrix.copy()
        filled_matrix[np.isnan(filled_matrix)] = 0
        
        # Fit NMF
        self.user_factors = self.model.fit_transform(filled_matrix)
        self.item_factors = self.model.components_.T
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        if self.user_factors is None:
            raise ValueError("Model must be fitted before making predictions")
        
        prediction = np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return max(1.0, min(5.0, prediction))  # Clip to rating range
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> list:
        if self.user_factors is None:
            raise ValueError("Model must be fitted before making predictions")
        
        user_predictions = np.dot(self.user_factors[user_id], self.item_factors.T)
        
        
        top_indices = np.argsort(user_predictions)[::-1][:n_recommendations]
        return top_indices.tolist() 