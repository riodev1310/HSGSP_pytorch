import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from typing import Dict, Optional, List
from config import Config

class Visualizer:
    """Visualization utilities"""
    
    def __init__(self, config: Config):
        self.config = config
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['savefig.dpi'] = 300

    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot training history with separate axes for loss and accuracy
        """
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.close()

    def plot_lr_schedule(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot learning rate schedule
        """
        epochs = range(1, len(history['lr']) + 1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['lr'], 'g-', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"LR schedule plot saved to {save_path}")
        
        plt.close()

    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
        """
        Plot comparison of different models
        """
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.sort_values('val_accuracy', ascending=False)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.bar(df.index, df['val_accuracy'], color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Inference Time (ms)', color=color)
        ax2.plot(df.index, df['inference_time'], color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.close()

    def plot_feature_maps(self, model: nn.Module, input_tensor: torch.Tensor, layer_name: str, save_path: Optional[str] = None):
        """
        Visualize feature maps from a specific layer
        """
        # Hook to get activations
        activations = {}
        def hook_fn(module, input, output):
            activations['output'] = output
        
        # Get layer
        layer = dict(model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            model(input_tensor)
        
        hook.remove()
        
        feature_maps = activations['output'][0]  # First image in batch
        
        num_maps = feature_maps.shape[0]
        grid_size = int(num_maps ** 0.5) + 1
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_maps):
            axes[i].imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
            axes[i].axis('off')
        
        for i in range(num_maps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Feature maps plot saved to {save_path}")
        
        plt.close()

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], classes: List[str], save_path: Optional[str] = None):
        """
        Plot confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()