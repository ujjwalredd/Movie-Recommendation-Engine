import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from typing import Tuple, Dict, List, Optional
import requests
import zipfile
from tqdm import tqdm


class MovieLensDataLoader:
    
    
    def __init__(self, data_path: str = "data/movielens", dataset_size: str = "1m"):
        self.data_path = data_path
        self.dataset_size = dataset_size
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
        # Mappings
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        # Statistics
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0
        
    def download_dataset(self):
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
        
        # Dataset URLs
        urls = {
            "100k": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
            "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
            "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        }
        
        if self.dataset_size not in urls:
            raise ValueError(f"Dataset size {self.dataset_size} not supported")
        
        url = urls[self.dataset_size]
        zip_path = os.path.join(self.data_path, f"ml-{self.dataset_size}.zip")
        
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens {self.dataset_size} dataset...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        self.download_dataset()
        
        # Determine the correct path based on dataset size
        if self.dataset_size == "100k":
            base_path = os.path.join(self.data_path, "ml-latest-small")
        else:
            base_path = os.path.join(self.data_path, f"ml-{self.dataset_size}")
        
        # Load data files
        ratings_file = os.path.join(base_path, "ratings.csv")
        movies_file = os.path.join(base_path, "movies.csv")
        
        if self.dataset_size == "1m":
            ratings_file = os.path.join(base_path, "ratings.dat")
            movies_file = os.path.join(base_path, "movies.dat")
            users_file = os.path.join(base_path, "users.dat")
            
            # Load 1M dataset with custom separators and proper encoding
            try:
                self.ratings_df = pd.read_csv(ratings_file, sep='::', 
                                            names=['userId', 'movieId', 'rating', 'timestamp'],
                                            engine='python', encoding='latin-1')
                self.movies_df = pd.read_csv(movies_file, sep='::', 
                                           names=['movieId', 'title', 'genres'],
                                           engine='python', encoding='latin-1')
                self.users_df = pd.read_csv(users_file, sep='::',
                                          names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
                                          engine='python', encoding='latin-1')
            except UnicodeDecodeError:
                # Fallback to different encoding if latin-1 fails
                self.ratings_df = pd.read_csv(ratings_file, sep='::', 
                                            names=['userId', 'movieId', 'rating', 'timestamp'],
                                            engine='python', encoding='iso-8859-1')
                self.movies_df = pd.read_csv(movies_file, sep='::', 
                                           names=['movieId', 'title', 'genres'],
                                           engine='python', encoding='iso-8859-1')
                self.users_df = pd.read_csv(users_file, sep='::',
                                          names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
                                          engine='python', encoding='iso-8859-1')
        else:
            # Load other datasets with proper encoding
            try:
                self.ratings_df = pd.read_csv(ratings_file, encoding='utf-8')
                self.movies_df = pd.read_csv(movies_file, encoding='utf-8')
            except UnicodeDecodeError:
                self.ratings_df = pd.read_csv(ratings_file, encoding='latin-1')
                self.movies_df = pd.read_csv(movies_file, encoding='latin-1')
            self.users_df = None
        
        print(f"Loaded {len(self.ratings_df)} ratings from {self.ratings_df['userId'].nunique()} users and {self.ratings_df['movieId'].nunique()} movies")
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def create_mappings(self):
        
        if self.ratings_df is None:
            raise ValueError("Data must be loaded before creating mappings")
        
        # Create user ID mapping
        unique_users = sorted(self.ratings_df['userId'].unique())
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        
        # Create item ID mapping
        unique_items = sorted(self.ratings_df['movieId'].unique())
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        # Update statistics
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        self.num_ratings = len(self.ratings_df)
        
        print(f"Created mappings: {self.num_users} users, {self.num_items} items, {self.num_ratings} ratings")
    
    def get_ratings_matrix(self, sparse: bool = True) -> np.ndarray:
        
        if not self.user_id_map:
            self.create_mappings()
        
        # Create sparse matrix
        if sparse:
            # Use scipy sparse matrix for memory efficiency
            from scipy.sparse import csr_matrix
            
            user_indices = [self.user_id_map[uid] for uid in self.ratings_df['userId']]
            item_indices = [self.item_id_map[mid] for mid in self.ratings_df['movieId']]
            ratings = self.ratings_df['rating'].values
            
            ratings_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                      shape=(self.num_users, self.num_items))
            return ratings_matrix
        else:
            # Create dense matrix (use with caution for large datasets)
            ratings_matrix = np.full((self.num_users, self.num_items), np.nan)
            
            for _, row in self.ratings_df.iterrows():
                user_idx = self.user_id_map[row['userId']]
                item_idx = self.item_id_map[row['movieId']]
                ratings_matrix[user_idx, item_idx] = row['rating']
            
            return ratings_matrix
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        if self.ratings_df is None:
            raise ValueError("Data must be loaded before splitting")
        
        # Sort by timestamp for temporal split
        self.ratings_df = self.ratings_df.sort_values('timestamp')
        
        # Split data
        train_df, temp_df = train_test_split(self.ratings_df, test_size=test_size + val_size, 
                                           random_state=random_state, shuffle=False)
        
        val_size_adjusted = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(temp_df, test_size=1 - val_size_adjusted, 
                                         random_state=random_state, shuffle=False)
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_user_item_pairs(self, df: pd.DataFrame) -> Tuple[List[int], List[int], List[float]]:
        
        if not self.user_id_map:
            self.create_mappings()
        
        user_indices = [self.user_id_map[uid] for uid in df['userId']]
        item_indices = [self.item_id_map[mid] for mid in df['movieId']]
        ratings = df['rating'].values
        
        return user_indices, item_indices, ratings
    
    def save_mappings(self, filepath: str):
        
        mappings = {
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'num_users': self.num_users,
            'num_items': self.num_items
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(mappings, f)
    
    def load_mappings(self, filepath: str):
        
        with open(filepath, 'rb') as f:
            mappings = pickle.load(f)
        
        self.user_id_map = mappings['user_id_map']
        self.item_id_map = mappings['item_id_map']
        self.reverse_user_map = mappings['reverse_user_map']
        self.reverse_item_map = mappings['reverse_item_map']
        self.num_users = mappings['num_users']
        self.num_items = mappings['num_items']


class DataProcessor:
    
    
    def __init__(self, batch_size: int = 1024, num_workers: int = 4):
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, data_loader: MovieLensDataLoader) -> Tuple[DataLoader, DataLoader, DataLoader]:

        from .dataset import MovieRecommendationDataset
        
        # Create datasets
        train_dataset = MovieRecommendationDataset(train_df, data_loader)
        val_dataset = MovieRecommendationDataset(val_df, data_loader)
        test_dataset = MovieRecommendationDataset(test_df, data_loader)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                              shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=self.num_workers)
        
        return train_loader, val_loader, test_loader
    
    def create_negative_samples(self, positive_df: pd.DataFrame, 
                              data_loader: MovieLensDataLoader, 
                              num_negatives: int = 1) -> pd.DataFrame:
        
        if not data_loader.user_id_map:
            data_loader.create_mappings()
        
        negative_samples = []
        
        for _, row in positive_df.iterrows():
            user_id = row['userId']
            user_idx = data_loader.user_id_map[user_id]
            
            # Get items rated by this user
            user_rated_items = set(positive_df[positive_df['userId'] == user_id]['movieId'])
            
            # Sample negative items
            all_items = set(data_loader.item_id_map.keys())
            negative_items = list(all_items - user_rated_items)
            
            if len(negative_items) >= num_negatives:
                sampled_negatives = np.random.choice(negative_items, num_negatives, replace=False)
                
                for neg_item in sampled_negatives:
                    negative_samples.append({
                        'userId': user_id,
                        'movieId': neg_item,
                        'rating': 0.0,  # Negative sample
                        'timestamp': row['timestamp']
                    })
        
        negative_df = pd.DataFrame(negative_samples)
        return negative_df
    
    def preprocess_for_implicit(self, df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
        
        df_implicit = df.copy()
        df_implicit['rating'] = (df_implicit['rating'] >= threshold).astype(float)
        return df_implicit 