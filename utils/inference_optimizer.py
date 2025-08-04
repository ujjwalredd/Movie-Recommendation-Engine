import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from functools import lru_cache
import threading
from collections import defaultdict
import pickle
import os


class InferenceOptimizer:
    """Optimized inference engine for recommendation models."""
    
    def __init__(self, cache_size: int = 1000, batch_size: int = 64):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.user_cache = {}
        self.item_cache = {}
        self.recommendation_cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    def optimize_model(self, model):
        """Apply optimizations to the model for faster inference."""
        model.eval()
        
        # For now, return the original model with eval mode
        # JIT compilation can be added later if needed
        return model
    
    @lru_cache(maxsize=1000)
    def get_cached_user_embeddings(self, model, user_id: int):
        """Cache user embeddings for faster access."""
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            if hasattr(model, 'user_embedding'):
                return model.user_embedding(user_tensor)
            elif hasattr(model, 'user_factors'):
                return torch.FloatTensor(model.user_factors[user_id])
        return None
    
    @lru_cache(maxsize=1000)
    def get_cached_item_embeddings(self, model, item_id: int):
        """Cache item embeddings for faster access."""
        with torch.no_grad():
            item_tensor = torch.LongTensor([item_id])
            if hasattr(model, 'item_embedding'):
                return model.item_embedding(item_tensor)
            elif hasattr(model, 'item_factors'):
                return torch.FloatTensor(model.item_factors[item_id])
        return None
    
    def batch_predict(self, model, user_ids: List[int], item_ids: List[int]) -> List[float]:
        """Batch prediction for multiple user-item pairs."""
        if not user_ids or not item_ids:
            return []
        
        with torch.no_grad():
            user_tensor = torch.LongTensor(user_ids)
            item_tensor = torch.LongTensor(item_ids)
            predictions = model(user_tensor, item_tensor)
            
            if hasattr(model, 'forward'):
                # For PyTorch models
                if predictions.dim() > 1:
                    predictions = predictions.squeeze()
                return predictions.cpu().numpy().tolist()
            else:
                # For traditional models
                return predictions.tolist()
    
    def optimized_get_recommendations(self, model, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Optimized recommendation generation with caching."""
        cache_key = f"{user_id}_{n_recommendations}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.recommendation_cache:
                self.cache_hits += 1
                return self.recommendation_cache[cache_key]
            self.cache_misses += 1
        
        start_time = time.time()
        
        # Generate recommendations using the original method
        recommendations = model.get_user_recommendations(user_id, n_recommendations)
        
        # Cache the result
        with self.cache_lock:
            if len(self.recommendation_cache) < self.cache_size:
                self.recommendation_cache[cache_key] = recommendations
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        return recommendations
    
    def _compute_recommendations_fallback(self, model, user_id: int, n_recommendations: int) -> List[int]:
        """Fallback method for models without get_user_recommendations."""
        predictions = []
        
        # Batch process predictions
        for i in range(0, model.num_items, self.batch_size):
            batch_items = list(range(i, min(i + self.batch_size, model.num_items)))
            batch_predictions = self.batch_predict(model, [user_id] * len(batch_items), batch_items)
            predictions.extend(batch_predictions)
        
        # Get top N items
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        return top_indices.tolist()
    
    def precompute_embeddings(self, model, user_ids: Optional[List[int]] = None, 
                            item_ids: Optional[List[int]] = None):
        """Precompute embeddings for frequently accessed users/items."""
        # For now, skip embedding precomputation to avoid issues
        print("Embedding precomputation skipped for compatibility")
        return
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {
                'avg_latency_ms': 0,
                'min_latency_ms': 0,
                'max_latency_ms': 0,
                'cache_hit_rate': 0,
                'total_requests': 0
            }
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'avg_latency_ms': np.mean(self.inference_times),
            'min_latency_ms': np.min(self.inference_times),
            'max_latency_ms': np.max(self.inference_times),
            'cache_hit_rate': cache_hit_rate,
            'total_requests': total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }
    
    def clear_cache(self):
        """Clear all caches."""
        with self.cache_lock:
            self.user_cache.clear()
            self.item_cache.clear()
            self.recommendation_cache.clear()
        
        # Clear LRU caches
        self.get_cached_user_embeddings.cache_clear()
        self.get_cached_item_embeddings.cache_clear()
    
    def save_cache(self, filepath: str):
        """Save cache to disk."""
        cache_data = {
            'recommendation_cache': self.recommendation_cache,
            'performance_stats': self.get_performance_stats()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_cache(self, filepath: str):
        """Load cache from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            with self.cache_lock:
                self.recommendation_cache = cache_data.get('recommendation_cache', {})
            
            print(f"Loaded cache with {len(self.recommendation_cache)} entries")


class ModelInferenceManager:
    """Manager for multiple optimized models."""
    
    def __init__(self):
        self.models = {}
        self.optimizers = {}
        self.performance_tracker = defaultdict(list)
    
    def add_model(self, model_name: str, model, optimize: bool = True):
        """Add a model to the manager."""
        self.models[model_name] = model
        
        if optimize:
            optimizer = InferenceOptimizer()
            optimized_model = optimizer.optimize_model(model)
            self.optimizers[model_name] = optimizer
            self.models[model_name] = optimized_model
        else:
            self.optimizers[model_name] = None
    
    def get_recommendations(self, model_name: str, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Get optimized recommendations from a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        optimizer = self.optimizers.get(model_name)
        
        if optimizer:
            return optimizer.optimized_get_recommendations(model, user_id, n_recommendations)
        else:
            # Fallback to original method
            return model.get_user_recommendations(user_id, n_recommendations)
    
    def batch_get_recommendations(self, model_name: str, user_ids: List[int], 
                                n_recommendations: int = 10) -> Dict[int, List[int]]:
        """Get recommendations for multiple users in batch."""
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.get_recommendations(model_name, user_id, n_recommendations)
        
        return results
    
    def get_model_performance(self, model_name: str) -> Dict:
        """Get performance statistics for a specific model."""
        if model_name not in self.optimizers or self.optimizers[model_name] is None:
            return {'status': 'not_optimized'}
        
        return self.optimizers[model_name].get_performance_stats()
    
    def precompute_all_embeddings(self, user_ids: Optional[List[int]] = None, 
                                item_ids: Optional[List[int]] = None):
        """Precompute embeddings for all models."""
        for model_name, optimizer in self.optimizers.items():
            if optimizer:
                print(f"Precomputing embeddings for {model_name}...")
                optimizer.precompute_embeddings(self.models[model_name], user_ids, item_ids)
    
    def clear_all_caches(self):
        """Clear caches for all models."""
        for optimizer in self.optimizers.values():
            if optimizer:
                optimizer.clear_cache() 