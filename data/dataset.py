import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, List


class MovieRecommendationDataset(Dataset):  
    
    def __init__(self, df: pd.DataFrame, data_loader, implicit: bool = False):
        self.df = df
        self.data_loader = data_loader
        self.implicit = implicit
        
        # Get user-item pairs
        self.user_indices, self.item_indices, self.ratings = data_loader.get_user_item_pairs(df)
        
        # Convert to tensors
        self.user_tensor = torch.LongTensor(self.user_indices)
        self.item_tensor = torch.LongTensor(self.item_indices)
        self.rating_tensor = torch.FloatTensor(self.ratings)
        
        if implicit:
            # Convert to binary for implicit feedback
            self.rating_tensor = (self.rating_tensor >= 3.5).float()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'user_idx': self.user_tensor[idx],
            'item_idx': self.item_tensor[idx],
            'rating': self.rating_tensor[idx]
        }


class MovieRecommendationDatasetWithNegatives(Dataset): 
    
    def __init__(self, positive_df: pd.DataFrame, data_loader, 
                 num_negatives: int = 1, implicit: bool = True):
        self.positive_df = positive_df
        self.data_loader = data_loader
        self.num_negatives = num_negatives
        self.implicit = implicit
        
        # Create negative samples
        self.negative_df = self._create_negative_samples()
        
        # Combine positive and negative samples
        self.combined_df = pd.concat([positive_df, self.negative_df], ignore_index=True)
        
        # Get user-item pairs
        self.user_indices, self.item_indices, self.ratings = data_loader.get_user_item_pairs(self.combined_df)
        
        # Convert to tensors
        self.user_tensor = torch.LongTensor(self.user_indices)
        self.item_tensor = torch.LongTensor(self.item_indices)
        self.rating_tensor = torch.FloatTensor(self.ratings)
        
        if implicit:
            # Convert to binary for implicit feedback
            self.rating_tensor = (self.rating_tensor >= 3.5).float()
    
    def _create_negative_samples(self) -> pd.DataFrame:
        negative_samples = []
        
        for _, row in self.positive_df.iterrows():
            user_id = row['userId']
            
            # Get items rated by this user
            user_rated_items = set(self.positive_df[self.positive_df['userId'] == user_id]['movieId'])
            
            # Sample negative items
            all_items = set(self.data_loader.item_id_map.keys())
            negative_items = list(all_items - user_rated_items)
            
            if len(negative_items) >= self.num_negatives:
                sampled_negatives = np.random.choice(negative_items, self.num_negatives, replace=False)
                
                for neg_item in sampled_negatives:
                    negative_samples.append({
                        'userId': user_id,
                        'movieId': neg_item,
                        'rating': 0.0,  # Negative sample
                        'timestamp': row['timestamp']
                    })
        
        return pd.DataFrame(negative_samples)
    
    def __len__(self):
        return len(self.combined_df)
    
    def __getitem__(self, idx):
        return {
            'user_idx': self.user_tensor[idx],
            'item_idx': self.item_tensor[idx],
            'rating': self.rating_tensor[idx]
        }


class MovieRecommendationDatasetWithFeatures(Dataset):
    
    def __init__(self, df: pd.DataFrame, data_loader, movies_df: pd.DataFrame = None, 
                 users_df: pd.DataFrame = None):
        self.df = df
        self.data_loader = data_loader
        self.movies_df = movies_df
        self.users_df = users_df
        
        # Get user-item pairs
        self.user_indices, self.item_indices, self.ratings = data_loader.get_user_item_pairs(df)
        
        # Convert to tensors
        self.user_tensor = torch.LongTensor(self.user_indices)
        self.item_tensor = torch.LongTensor(self.item_indices)
        self.rating_tensor = torch.FloatTensor(self.ratings)
        
        # Create feature tensors if available
        self.user_features = None
        self.item_features = None
        
        if users_df is not None:
            self.user_features = self._create_user_features()
        
        if movies_df is not None:
            self.item_features = self._create_item_features()
    
    def _create_user_features(self) -> torch.Tensor:
        # Simple feature encoding for demo
        # In practice, you'd use more sophisticated feature engineering
        features = []
        
        for user_id in self.df['userId'].unique():
            user_data = self.users_df[self.users_df['userId'] == user_id].iloc[0]
            
            # Encode gender (0: F, 1: M)
            gender_feature = 1.0 if user_data['gender'] == 'M' else 0.0
            
            # Normalize age (assuming age is numeric)
            age_feature = user_data['age'] / 100.0  # Normalize to [0, 1]
            
            # Encode occupation (simple one-hot for demo)
            occupation_feature = hash(user_data['occupation']) % 10 / 10.0  # Simple hash
            
            features.append([gender_feature, age_feature, occupation_feature])
        
        return torch.FloatTensor(features)
    
    def _create_item_features(self) -> torch.Tensor:    
        # Simple feature encoding for demo
        # In practice, you'd use more sophisticated feature engineering
        features = []
        
        for movie_id in self.df['movieId'].unique():
            movie_data = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            
            # Encode genres (simple hash for demo)
            genre_feature = hash(movie_data['genres']) % 20 / 20.0
            
            # Title length feature
            title_length = len(movie_data['title']) / 100.0  # Normalize
            
            features.append([genre_feature, title_length])
        
        return torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {
            'user_idx': self.user_tensor[idx],
            'item_idx': self.item_tensor[idx],
            'rating': self.rating_tensor[idx]
        }
        
        if self.user_features is not None:
            item['user_features'] = self.user_features[self.user_tensor[idx]]
        
        if self.item_features is not None:
            item['item_features'] = self.item_features[self.item_tensor[idx]]
        
        return item 