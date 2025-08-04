import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time


class RecommendationMetrics:
    
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_rmse(self, y_true: List[float], y_pred: List[float]) -> float:
        
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true: List[float], y_pred: List[float]) -> float:
        
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_precision_at_k(self, y_true: List[int], y_pred: List[int], k: int = 10) -> float:
        
        if len(y_pred) == 0:
            return 0.0
        
        # Get top-k predictions
        top_k_pred = y_pred[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = sum(1 for item in top_k_pred if item in y_true)
        
        return relevant_in_top_k / len(top_k_pred)
    
    def calculate_recall_at_k(self, y_true: List[int], y_pred: List[int], k: int = 10) -> float:
        
        if len(y_true) == 0:
            return 0.0
        
        # Get top-k predictions
        top_k_pred = y_pred[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = sum(1 for item in top_k_pred if item in y_true)
        
        return relevant_in_top_k / len(y_true)
    
    def calculate_ndcg_at_k(self, y_true: List[int], y_pred: List[int], k: int = 10) -> float:
        
        if len(y_pred) == 0:
            return 0.0
        
        # Get top-k predictions
        top_k_pred = y_pred[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_pred):
            if item in y_true:
                dcg += 1.0 / np.log2(i + 2)  # log2(i+2) because i starts from 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        num_relevant = min(len(y_true), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_map_at_k(self, y_true: List[int], y_pred: List[int], k: int = 10) -> float:
        
        if len(y_pred) == 0:
            return 0.0
        
        # Get top-k predictions
        top_k_pred = y_pred[:k]
        
        # Calculate average precision
        relevant_count = 0
        precision_sum = 0.0
        
        for i, item in enumerate(top_k_pred):
            if item in y_true:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(y_true) if len(y_true) > 0 else 0.0
    
    def calculate_hit_rate_at_k(self, y_true: List[int], y_pred: List[int], k: int = 10) -> float:
        
        if len(y_pred) == 0:
            return 0.0
        
        # Get top-k predictions
        top_k_pred = y_pred[:k]
        
        # Check if any relevant item is in top-k
        return 1.0 if any(item in y_true for item in top_k_pred) else 0.0
    
    def calculate_diversity(self, recommendations: List[List[int]], item_popularity: Dict[int, int] = None) -> float:
        
        if not recommendations:
            return 0.0
        
        diversity_scores = []
        
        for rec_list in recommendations:
            if len(rec_list) < 2:
                diversity_scores.append(0.0)
                continue
            
            # Calculate pairwise diversity
            diversity = 0.0
            count = 0
            
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    # Simple diversity: different items are diverse
                    # In practice, you'd use item features or categories
                    diversity += 1.0 if rec_list[i] != rec_list[j] else 0.0
                    count += 1
            
            diversity_scores.append(diversity / count if count > 0 else 0.0)
        
        return np.mean(diversity_scores)
    
    def calculate_novelty(self, recommendations: List[List[int]], item_popularity: Dict[int, int]) -> float:
        
        if not recommendations or not item_popularity:
            return 0.0
        
        novelty_scores = []
        total_interactions = sum(item_popularity.values())
        
        for rec_list in recommendations:
            if not rec_list:
                novelty_scores.append(0.0)
                continue
            
            # Calculate average novelty
            item_novelties = []
            for item in rec_list:
                if item in item_popularity:
                    # Novelty = -log2(popularity)
                    popularity = item_popularity[item] / total_interactions
                    novelty = -np.log2(popularity + 1e-10)  # Add small epsilon to avoid log(0)
                    item_novelties.append(novelty)
            
            novelty_scores.append(np.mean(item_novelties) if item_novelties else 0.0)
        
        return np.mean(novelty_scores)
    
    def calculate_coverage(self, recommendations: List[List[int]], total_items: int) -> float:

        if not recommendations or total_items == 0:
            return 0.0
        
        # Get all unique recommended items
        recommended_items = set()
        for rec_list in recommendations:
            recommended_items.update(rec_list)
        
        return len(recommended_items) / total_items
    
    def evaluate_recommendations(self, y_true: List[List[int]], y_pred: List[List[int]], 
                               k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        
        results = {}
        
        for k in k_values:
            # Calculate metrics for each k
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            map_scores = []
            hit_rate_scores = []
            
            for true_items, pred_items in zip(y_true, y_pred):
                precision_scores.append(self.calculate_precision_at_k(true_items, pred_items, k))
                recall_scores.append(self.calculate_recall_at_k(true_items, pred_items, k))
                ndcg_scores.append(self.calculate_ndcg_at_k(true_items, pred_items, k))
                map_scores.append(self.calculate_map_at_k(true_items, pred_items, k))
                hit_rate_scores.append(self.calculate_hit_rate_at_k(true_items, pred_items, k))
            
            # Store average metrics
            results[f'precision@{k}'] = np.mean(precision_scores)
            results[f'recall@{k}'] = np.mean(recall_scores)
            results[f'ndcg@{k}'] = np.mean(ndcg_scores)
            results[f'map@{k}'] = np.mean(map_scores)
            results[f'hit_rate@{k}'] = np.mean(hit_rate_scores)
        
        return results


class PrecisionAtK:
    
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def calculate(self, y_true: List[int], y_pred: List[int]) -> float:
        
        if len(y_pred) == 0:
            return 0.0
        
        top_k_pred = y_pred[:self.k]
        relevant_in_top_k = sum(1 for item in top_k_pred if item in y_true)
        
        return relevant_in_top_k / len(top_k_pred)


class RecallAtK:
    
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def calculate(self, y_true: List[int], y_pred: List[int]) -> float:
        
        if len(y_true) == 0:
            return 0.0
        
        top_k_pred = y_pred[:self.k]
        relevant_in_top_k = sum(1 for item in top_k_pred if item in y_true)
        
        return relevant_in_top_k / len(y_true)


class NDCGAtK:
    
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def calculate(self, y_true: List[int], y_pred: List[int]) -> float:
                    
        if len(y_pred) == 0:
            return 0.0
        
        top_k_pred = y_pred[:self.k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_pred):
            if item in y_true:
                dcg += 1.0 / np.log2(i + 2)
        
        # Calculate IDCG
        idcg = 0.0
        num_relevant = min(len(y_true), self.k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0 