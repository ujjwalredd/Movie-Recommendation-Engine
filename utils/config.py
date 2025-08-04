"""
Configuration settings for the Movie Recommendation Engine.
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for the recommendation engine."""
    
    # Data settings
    DATA_PATH = os.getenv("DATA_PATH", "data/movielens")
    DATASET_SIZE = os.getenv("DATASET_SIZE", "1m")  # 100k, 1m, 10m, 25m
    
    # Model settings
    DEFAULT_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "50"))
    DEFAULT_NUM_FACTORS = int(os.getenv("NUM_FACTORS", "50"))
    DEFAULT_LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
    DEFAULT_EPOCHS = int(os.getenv("EPOCHS", "100"))
    
    # Training settings
    EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
    VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))
    TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.1"))
    
    # Model paths
    MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models")
    TRAINING_HISTORY_PATH = os.getenv("TRAINING_HISTORY_PATH", "training_history.json")
    
    # Evaluation settings
    EVALUATION_METRICS = ["precision@10", "recall@10", "ndcg@10", "rmse", "mae"]
    LATENCY_TARGET_MS = float(os.getenv("LATENCY_TARGET_MS", "200"))
    PRECISION_TARGET = float(os.getenv("PRECISION_TARGET", "0.85"))
    
    # A/B Testing settings
    AB_TEST_DEFAULT_DURATION = int(os.getenv("AB_TEST_DURATION", "30"))
    AB_TEST_CONFIDENCE_LEVEL = float(os.getenv("AB_TEST_CONFIDENCE", "0.95"))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))
    
    # Web interface settings
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance settings
    MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "50"))
    BATCH_PROCESSING_SIZE = int(os.getenv("BATCH_PROCESSING_SIZE", "1000"))
    
    # Hyperparameter optimization
    HYPERPARAM_OPTIMIZATION_TRIALS = int(os.getenv("HYPERPARAM_TRIALS", "20"))
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        
        configs = {
            "matrix_factorization": {
                "num_factors": cls.DEFAULT_NUM_FACTORS,
                "learning_rate": cls.DEFAULT_LEARNING_RATE,
                "reg_param": 0.01,
                "epochs": cls.DEFAULT_EPOCHS,
                "batch_size": cls.DEFAULT_BATCH_SIZE
            },
            "neural_cf": {
                "embedding_dim": cls.DEFAULT_EMBEDDING_DIM,
                "layers": [100, 50, 20],
                "dropout": 0.1,
                "learning_rate": cls.DEFAULT_LEARNING_RATE,
                "epochs": cls.DEFAULT_EPOCHS,
                "batch_size": cls.DEFAULT_BATCH_SIZE
            },
            "hybrid": {
                "mf_factors": cls.DEFAULT_NUM_FACTORS,
                "embedding_dim": cls.DEFAULT_EMBEDDING_DIM,
                "neural_layers": [100, 50, 20],
                "dropout": 0.1,
                "learning_rate": cls.DEFAULT_LEARNING_RATE,
                "epochs": cls.DEFAULT_EPOCHS,
                "batch_size": cls.DEFAULT_BATCH_SIZE
            }
        }
        
        return configs.get(model_type, {})
    
    @classmethod
    def get_hyperparameter_grid(cls, model_type: str) -> Dict[str, list]:
        """Get hyperparameter grid for optimization."""
        
        grids = {
            "matrix_factorization": {
                "num_factors": [20, 50, 100, 200],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                "reg_param": [0.001, 0.01, 0.1, 1.0]
            },
            "neural_cf": {
                "embedding_dim": [32, 50, 100, 200],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1],
                "dropout": [0.1, 0.2, 0.3, 0.5]
            },
            "hybrid": {
                "mf_factors": [20, 50, 100],
                "embedding_dim": [32, 50, 100],
                "learning_rate": [0.0001, 0.001, 0.01]
            }
        }
        
        return grids.get(model_type, {})
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        
        # Check data path
        if not os.path.exists(cls.DATA_PATH):
            print(f"Warning: Data path {cls.DATA_PATH} does not exist")
        
        # Check model save path
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        
        # Validate numeric settings
        if cls.DEFAULT_LEARNING_RATE <= 0 or cls.DEFAULT_LEARNING_RATE > 1:
            print("Error: Learning rate must be between 0 and 1")
            return False
        
        if cls.DEFAULT_BATCH_SIZE <= 0:
            print("Error: Batch size must be positive")
            return False
        
        if cls.DEFAULT_EPOCHS <= 0:
            print("Error: Number of epochs must be positive")
            return False
        
        if cls.VALIDATION_SPLIT <= 0 or cls.VALIDATION_SPLIT >= 1:
            print("Error: Validation split must be between 0 and 1")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        
        print("Movie Recommendation Engine Configuration:")
        print("=" * 50)
        print(f"Data Path: {cls.DATA_PATH}")
        print(f"Dataset Size: {cls.DATASET_SIZE}")
        print(f"Embedding Dimension: {cls.DEFAULT_EMBEDDING_DIM}")
        print(f"Number of Factors: {cls.DEFAULT_NUM_FACTORS}")
        print(f"Learning Rate: {cls.DEFAULT_LEARNING_RATE}")
        print(f"Batch Size: {cls.DEFAULT_BATCH_SIZE}")
        print(f"Epochs: {cls.DEFAULT_EPOCHS}")
        print(f"Validation Split: {cls.VALIDATION_SPLIT}")
        print(f"Test Split: {cls.TEST_SPLIT}")
        print(f"Model Save Path: {cls.MODEL_SAVE_PATH}")
        print(f"API Host: {cls.API_HOST}")
        print(f"API Port: {cls.API_PORT}")
        print(f"Latency Target: {cls.LATENCY_TARGET_MS}ms")
        print(f"Precision Target: {cls.PRECISION_TARGET}")
        print("=" * 50) 