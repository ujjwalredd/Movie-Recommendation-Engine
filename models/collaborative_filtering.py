import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict, Optional
import time
from collections import defaultdict


class UserBasedCF:
    
    def __init__(self, k_neighbors: int = 50, min_similarity: float = 0.1):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.ratings_matrix = None
        self.user_similarity_matrix = None
        self.user_means = None
        
    def fit(self, ratings_matrix: np.ndarray):
        self.ratings_matrix = ratings_matrix.copy()
        
        # Calculate user means
        self.user_means = np.nanmean(ratings_matrix, axis=1, keepdims=True)
        
        # Center the ratings
        centered_ratings = ratings_matrix - self.user_means
        centered_ratings = np.nan_to_num(centered_ratings, 0)
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(centered_ratings)
        
        # Set diagonal to 0 (user similarity with itself)
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get user's mean rating
        user_mean = self.user_means[user_id][0]
        
        # Find similar users who rated this item
        similar_users = []
        similarities = []
        
        for other_user in range(self.ratings_matrix.shape[0]):
            if other_user != user_id and not np.isnan(self.ratings_matrix[other_user, item_id]):
                similarity = self.user_similarity_matrix[user_id, other_user]
                if similarity > self.min_similarity:
                    similar_users.append(other_user)
                    similarities.append(similarity)
        
        if not similar_users:
            return user_mean
        
        # Get top k similar users
        if len(similar_users) > self.k_neighbors:
            indices = np.argsort(similarities)[::-1][:self.k_neighbors]
            similar_users = [similar_users[i] for i in indices]
            similarities = [similarities[i] for i in indices]
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for other_user, similarity in zip(similar_users, similarities):
            rating = self.ratings_matrix[other_user, item_id]
            other_user_mean = self.user_means[other_user][0]
            
            numerator += similarity * (rating - other_user_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return user_mean
        
        prediction = user_mean + (numerator / denominator)
        return max(1.0, min(5.0, prediction))  # Clip to rating range
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        start_time = time.time()
        
        # Get items the user hasn't rated
        user_ratings = self.ratings_matrix[user_id]
        unrated_items = np.where(np.isnan(user_ratings))[0]
        
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommendations = [item_id for item_id, _ in predictions[:n_recommendations]]
        
        latency = (time.time() - start_time) * 1000
        print(f"User-based CF recommendation latency: {latency:.2f}ms")
        
        return recommendations


class ItemBasedCF:
    
    def __init__(self, k_neighbors: int = 50, min_similarity: float = 0.1):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.ratings_matrix = None
        self.item_similarity_matrix = None
        self.item_means = None
        
    def fit(self, ratings_matrix: np.ndarray):
        self.ratings_matrix = ratings_matrix.copy()
        
        # Calculate item means
        self.item_means = np.nanmean(ratings_matrix, axis=0, keepdims=True)
        
        # Center the ratings
        centered_ratings = ratings_matrix - self.item_means
        centered_ratings = np.nan_to_num(centered_ratings, 0)
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(centered_ratings.T)
        
        # Set diagonal to 0 (item similarity with itself)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get item's mean rating
        item_mean = self.item_means[0, item_id]
        
        # Find similar items that the user rated
        similar_items = []
        similarities = []
        
        for other_item in range(self.ratings_matrix.shape[1]):
            if other_item != item_id and not np.isnan(self.ratings_matrix[user_id, other_item]):
                similarity = self.item_similarity_matrix[item_id, other_item]
                if similarity > self.min_similarity:
                    similar_items.append(other_item)
                    similarities.append(similarity)
        
        if not similar_items:
            return item_mean
        
        # Get top k similar items
        if len(similar_items) > self.k_neighbors:
            indices = np.argsort(similarities)[::-1][:self.k_neighbors]
            similar_items = [similar_items[i] for i in indices]
            similarities = [similarities[i] for i in indices]
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for other_item, similarity in zip(similar_items, similarities):
            rating = self.ratings_matrix[user_id, other_item]
            other_item_mean = self.item_means[0, other_item]
            
            numerator += similarity * (rating - other_item_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return item_mean
        
        prediction = item_mean + (numerator / denominator)
        return max(1.0, min(5.0, prediction))  # Clip to rating range
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        start_time = time.time()
        
        # Get items the user hasn't rated
        user_ratings = self.ratings_matrix[user_id]
        unrated_items = np.where(np.isnan(user_ratings))[0]
        
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommendations = [item_id for item_id, _ in predictions[:n_recommendations]]
        
        latency = (time.time() - start_time) * 1000
        print(f"Item-based CF recommendation latency: {latency:.2f}ms")
        
        return recommendations


class SlopeOneCF:
    
    def __init__(self):
        self.ratings_matrix = None
        self.deviations = None
        self.frequencies = None
        
    def fit(self, ratings_matrix: np.ndarray):
        self.ratings_matrix = ratings_matrix.copy()
        num_items = ratings_matrix.shape[1]
        
        # Initialize deviation and frequency matrices
        self.deviations = np.zeros((num_items, num_items))
        self.frequencies = np.zeros((num_items, num_items))
        
        # Calculate deviations and frequencies
        for i in range(num_items):
            for j in range(num_items):
                if i != j:
                    # Find users who rated both items
                    mask = ~(np.isnan(ratings_matrix[:, i]) | np.isnan(ratings_matrix[:, j]))
                    if np.sum(mask) > 0:
                        ratings_i = ratings_matrix[mask, i]
                        ratings_j = ratings_matrix[mask, j]
                        
                        # Calculate deviation
                        deviation = np.mean(ratings_i - ratings_j)
                        self.deviations[i, j] = deviation
                        self.frequencies[i, j] = len(ratings_i)
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        user_ratings = self.ratings_matrix[user_id]
        rated_items = ~np.isnan(user_ratings)
        
        if not np.any(rated_items):
            return 3.0  # Default rating
        
        numerator = 0
        denominator = 0
        
        for other_item in range(len(user_ratings)):
            if rated_items[other_item] and other_item != item_id:
                deviation = self.deviations[item_id, other_item]
                frequency = self.frequencies[item_id, other_item]
                
                if frequency > 0:
                    predicted_rating = user_ratings[other_item] + deviation
                    numerator += frequency * predicted_rating
                    denominator += frequency
        
        if denominator == 0:
            return 3.0  # Default rating
        
        prediction = numerator / denominator
        return max(1.0, min(5.0, prediction))  # Clip to rating range
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        start_time = time.time()
        
        # Get items the user hasn't rated
        user_ratings = self.ratings_matrix[user_id]
        unrated_items = np.where(np.isnan(user_ratings))[0]
        
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommendations = [item_id for item_id, _ in predictions[:n_recommendations]]
        
        latency = (time.time() - start_time) * 1000
        print(f"Slope One CF recommendation latency: {latency:.2f}ms")
        
        return recommendations 