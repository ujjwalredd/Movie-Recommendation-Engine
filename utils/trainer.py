import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {}
        self.best_models = {}
        
    def train_matrix_factorization(self, model: nn.Module, train_loader: DataLoader, 
                                 val_loader: DataLoader, num_epochs: int = 100,
                                 learning_rate: float = 0.001, weight_decay: float = 0.01,
                                 early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        
        print(f"Training Matrix Factorization model on {self.device}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                user_indices = batch['user_idx'].to(self.device)
                item_indices = batch['item_idx'].to(self.device)
                ratings = batch['rating'].float().to(self.device)
                
                # Forward pass
                predictions = model(user_indices, item_indices)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    user_indices = batch['user_idx'].to(self.device)
                    item_indices = batch['item_idx'].to(self.device)
                    ratings = batch['rating'].float().to(self.device)
                    
                    predictions = model(user_indices, item_indices)
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_mf_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load('best_mf_model.pth'))
        
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        self.training_history['matrix_factorization'] = training_history
        self.best_models['matrix_factorization'] = model
        
        return training_history
    
    def train_neural_cf(self, model: nn.Module, train_loader: DataLoader, 
                       val_loader: DataLoader, num_epochs: int = 100,
                       learning_rate: float = 0.001, weight_decay: float = 0.01,
                       early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        
        print(f"Training Neural CF model on {self.device}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()  # For binary classification (implicit feedback)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                user_indices = batch['user_idx'].to(self.device)
                item_indices = batch['item_idx'].to(self.device)
                ratings = batch['rating'].float().to(self.device)
                
                # Forward pass
                predictions = model(user_indices, item_indices)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    user_indices = batch['user_idx'].to(self.device)
                    item_indices = batch['item_idx'].to(self.device)
                    ratings = batch['rating'].float().to(self.device)
                    
                    predictions = model(user_indices, item_indices)
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_neural_cf_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load('best_neural_cf_model.pth'))
        
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        self.training_history['neural_cf'] = training_history
        self.best_models['neural_cf'] = model
        
        return training_history
    
    def train_hybrid_model(self, model: nn.Module, train_loader: DataLoader, 
                          val_loader: DataLoader, num_epochs: int = 100,
                          learning_rate: float = 0.001, weight_decay: float = 0.01,
                          early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        
        print(f"Training Hybrid model on {self.device}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                user_indices = batch['user_idx'].to(self.device)
                item_indices = batch['item_idx'].to(self.device)
                ratings = batch['rating'].float().to(self.device)
                
                # Forward pass
                predictions = model(user_indices, item_indices)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    user_indices = batch['user_idx'].to(self.device)
                    item_indices = batch['item_idx'].to(self.device)
                    ratings = batch['rating'].float().to(self.device)
                    
                    predictions = model(user_indices, item_indices)
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_hybrid_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load('best_hybrid_model.pth'))
        
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        self.training_history['hybrid'] = training_history
        self.best_models['hybrid'] = model
        
        return training_history
    
    def hyperparameter_optimization(self, model_class: type, train_loader: DataLoader, 
                                  val_loader: DataLoader, param_grid: Dict[str, List],
                                  num_trials: int = 20) -> Dict[str, Any]:
        
        print("Starting hyperparameter optimization...")
        
        best_params = None
        best_val_loss = float('inf')
        optimization_results = []
        
        for trial in range(num_trials):
            # Sample parameters from grid
            params = {}
            for param_name, param_values in param_grid.items():
                params[param_name] = np.random.choice(param_values)
            
            print(f"Trial {trial+1}/{num_trials}: Testing params {params}")
            
            # Create model with current parameters
            model = model_class(**params)
            
            # Train model
            if 'num_factors' in params or 'embedding_dim' in params:
                # Matrix factorization or neural CF
                training_history = self.train_matrix_factorization(
                    model, train_loader, val_loader, 
                    num_epochs=50,  # Shorter training for optimization
                    learning_rate=params.get('learning_rate', 0.001)
                )
            else:
                # Hybrid model
                training_history = self.train_hybrid_model(
                    model, train_loader, val_loader,
                    num_epochs=50,
                    learning_rate=params.get('learning_rate', 0.001)
                )
            
            val_loss = training_history['best_val_loss']
            
            # Store results
            result = {
                'trial': trial + 1,
                'params': params,
                'val_loss': val_loss
            }
            optimization_results.append(result)
            
            # Update best parameters
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params.copy()
                print(f"New best validation loss: {best_val_loss:.4f}")
        
        print(f"Optimization completed. Best params: {best_params}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return {
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'all_results': optimization_results
        }
    
    def plot_training_history(self, model_name: str = None):
        
        if model_name:
            if model_name not in self.training_history:
                print(f"No training history found for {model_name}")
                return
            
            history = self.training_history[model_name]
            self._plot_single_training_history(model_name, history)
        else:
            # Plot all models
            fig, axes = plt.subplots(1, len(self.training_history), figsize=(5*len(self.training_history), 5))
            if len(self.training_history) == 1:
                axes = [axes]
            
            for i, (name, history) in enumerate(self.training_history.items()):
                self._plot_single_training_history(name, history, axes[i])
            
            plt.tight_layout()
            plt.show()
    
    def _plot_single_training_history(self, model_name: str, history: Dict[str, List[float]], ax=None):
        
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        ax.set_title(f'Training History - {model_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        if ax is None:
            plt.show()
    
    def save_model(self, model: nn.Module, filepath: str):
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model: nn.Module, filepath: str):
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        model.eval()
        print(f"Model loaded from {filepath}")
        return model
    
    def save_training_history(self, filepath: str):
        # Convert numpy arrays to lists for JSON serialization
        history_for_save = {}
        for model_name, history in self.training_history.items():
            history_for_save[model_name] = {
                'train_losses': [float(x) for x in history['train_losses']],
                'val_losses': [float(x) for x in history['val_losses']],
                'best_val_loss': float(history['best_val_loss'])
            }
        
        with open(filepath, 'w') as f:
            json.dump(history_for_save, f, indent=2)
        
        print(f"Training history saved to {filepath}")
    
    def load_training_history(self, filepath: str):     
        with open(filepath, 'r') as f:
            self.training_history = json.load(f)
        
        print(f"Training history loaded from {filepath}")
    
    def get_model_summary(self, model) -> Dict[str, Any]:
        
        # Check if it's a PyTorch model
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            device = next(model.parameters()).device
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        else:
            # For traditional models (UserBasedCF, ItemBasedCF, etc.)
            total_params = 0
            trainable_params = 0
            device = 'cpu'
            model_size_mb = 0.0
        
        summary = {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': device,
            'model_size_mb': model_size_mb
        }
        
        return summary 