import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import time


class NeuralCollaborativeFiltering(nn.Module):  
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50,
                 layers: List[int] = [100, 50, 20], dropout: float = 0.1,
                 learning_rate: float = 0.001):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Build MLP layers
        self.mlp_layers = nn.ModuleList()
        input_dim = embedding_dim * 2  # Concatenated user and item embeddings
        
        for layer_size in layers:
            self.mlp_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, user_indices, item_indices):
        # Get embeddings
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        concat_embedding = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Pass through MLP layers
        x = concat_embedding
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = self.dropout_layer(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output.squeeze()
    
    def predict(self, user_id: int, item_id: int) -> float:
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.forward(user_tensor, item_tensor)
            return torch.sigmoid(prediction).item() * 5.0  # Scale to 0-5 range
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items)
            item_tensor = torch.arange(self.num_items)
            predictions = self.forward(user_tensor, item_tensor)
            
            # Apply sigmoid and scale
            predictions = torch.sigmoid(predictions) * 5.0
            
            # Get top N items
            _, indices = torch.topk(predictions, n_recommendations)
            return indices.cpu().numpy().tolist()
    
    def get_embeddings(self):
        return (self.user_embedding.weight.detach().cpu().numpy(),
                self.item_embedding.weight.detach().cpu().numpy())


class GMF(nn.Module):
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50):
        super(GMF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Element-wise product
        return user_embedding * item_embedding


class MLP(nn.Module):   
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50,
                 layers: List[int] = [100, 50, 20], dropout: float = 0.1):
        super(MLP, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Build MLP layers
        self.mlp_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for layer_size in layers:
            self.mlp_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        concat_embedding = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Pass through MLP layers
        x = concat_embedding
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        return x


class NCF(nn.Module):
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50,
                 mlp_layers: List[int] = [100, 50, 20], dropout: float = 0.1):
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # GMF and MLP components
        self.gmf = GMF(num_users, num_items, embedding_dim)
        self.mlp = MLP(num_users, num_items, embedding_dim, mlp_layers, dropout)
        
        # Fusion layer
        fusion_input_dim = embedding_dim + mlp_layers[-1]
        self.fusion_layer = nn.Linear(fusion_input_dim, 1)
        
        # Initialize fusion layer
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
        
    def forward(self, user_indices, item_indices):
        gmf_output = self.gmf(user_indices, item_indices)
        mlp_output = self.mlp(user_indices, item_indices)
        
        # Concatenate GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction
        prediction = self.fusion_layer(concat_output)
        
        return prediction.squeeze()
    
    def predict(self, user_id: int, item_id: int) -> float:
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.forward(user_tensor, item_tensor)
            return torch.sigmoid(prediction).item() * 5.0
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]: 
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items)
            item_tensor = torch.arange(self.num_items)
            predictions = self.forward(user_tensor, item_tensor)
            
            # Apply sigmoid and scale
            predictions = torch.sigmoid(predictions) * 5.0
            
            # Get top N items
            _, indices = torch.topk(predictions, n_recommendations)
            return indices.cpu().numpy().tolist() 