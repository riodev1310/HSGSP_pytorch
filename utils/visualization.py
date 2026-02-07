import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns

class Visualizer:
    """Visualization utilities for HSGSP"""
    
    def __init__(self, config):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_frequency_spectrum(self,
                               dct_coefficients: np.ndarray,
                               layer_name: str,
                               save_path: Optional[str] = None):
        """Plot frequency spectrum of filters"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Average spectrum across filters
        avg_spectrum = np.mean(np.abs(dct_coefficients), axis=(2, 3))
        
        # Plot 2D spectrum
        im1 = axes[0, 0].imshow(avg_spectrum, cmap='hot', aspect='auto')
        axes[0, 0].set_title(f'Average Frequency Spectrum - {layer_name}')
        axes[0, 0].set_xlabel('Frequency X')
        axes[0, 0].set_ylabel('Frequency Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot radial frequency profile
        radial_profile = self._compute_radial_profile(avg_spectrum)
        axes[0, 1].plot(radial_profile)
        axes[0, 1].set_title('Radial Frequency Profile')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True)
        
        # Plot frequency band distribution
        bands = ['Low', 'Mid', 'High']
        band_energies = self._compute_band_energies(dct_coefficients)
        axes[1, 0].bar(bands, band_energies)
        axes[1, 0].set_title('Frequency Band Energy Distribution')
        axes[1, 0].set_ylabel('Energy')
        
        # Plot filter-wise frequency characteristics
        filter_energies = np.sum(np.abs(dct_coefficients)**2, axis=(0, 1, 2))
        axes[1, 1].hist(filter_energies, bins=30, edgecolor='black')
        axes[1, 1].set_title('Filter Energy Distribution')
        axes[1, 1].set_xlabel('Energy')
        axes[1, 1].set_ylabel('Number of Filters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pruning_analysis(self,
                            importance_scores: Dict[str, np.ndarray],
                            pruning_ratios: Dict[str, float],
                            save_path: Optional[str] = None):
        """Plot pruning analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        layer_names = list(importance_scores.keys())[:6] # Plot first 6 layers
        
        for idx, layer_name in enumerate(layer_names):
            row, col = idx // 3, idx % 3
            
            scores = importance_scores[layer_name]
            ratio = pruning_ratios.get(layer_name, 0)
            threshold = np.percentile(scores, ratio * 100)
            
            axes[row, col].hist(scores, bins=30, alpha=0.7, label='All filters')
            axes[row, col].axvline(threshold, color='r', linestyle='--',
                                  label=f'Pruning threshold ({ratio:.1%})')
            axes[row, col].set_title(f'{layer_name}')
            axes[row, col].set_xlabel('Importance Score')
            axes[row, col].set_ylabel('Number of Filters')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.suptitle('Filter Importance Distribution by Layer', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self,
                            history: Dict[str, List],
                            save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_compression_comparison(self,
                                   original_metrics: Dict,
                                   pruned_metrics: Dict,
                                   save_path: Optional[str] = None):
        """Plot model compression comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Parameters comparison
        categories = ['Total\nParams', 'Model\nSize (MB)', 'FLOPs\n(M)']
        original_values = [
            original_metrics['total_params'] / 1e6,
            original_metrics['model_size_mb'],
            original_metrics['flops'] / 1e6
        ]
        pruned_values = [
            pruned_metrics['total_params'] / 1e6,
            pruned_metrics['model_size_mb'],
            pruned_metrics['flops'] / 1e6
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, original_values, width, label='Original', color='blue')
        axes[0, 0].bar(x + width/2, pruned_values, width, label='Pruned', color='red')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Model Complexity Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(categories)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Compression ratios
        compression_data = {
            'Parameter\nReduction': 1 - pruned_metrics['total_params'] / original_metrics['total_params'],
            'Size\nReduction': 1 - pruned_metrics['model_size_mb'] / original_metrics['model_size_mb'],
            'FLOP\nReduction': 1 - pruned_metrics['flops'] / original_metrics['flops'],
            'Speedup': original_metrics['inference_time_ms'] / pruned_metrics['inference_time_ms']
        }
        
        axes[0, 1].bar(compression_data.keys(), compression_data.values(), color='green')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].set_title('Compression Metrics')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy comparison
        datasets = ['Dataset']
        original_acc = [original_metrics.get('accuracy', 0)]
        pruned_acc = [pruned_metrics.get('accuracy', 0)]
        
        x = np.arange(len(datasets))
        axes[1, 0].bar(x - width/2, original_acc, width, label='Original', color='blue')
        axes[1, 0].bar(x + width/2, pruned_acc, width, label='Pruned', color='red')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(datasets)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Inference time comparison
        inference_data = {
            'Original': original_metrics['inference_time_ms'],
            'Pruned': pruned_metrics['inference_time_ms']
        }
        
        axes[1, 1].bar(inference_data.keys(), inference_data.values(), color=['blue', 'red'])
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].set_title('Inference Time per Sample')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Model Compression Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self,
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _compute_radial_profile(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute radial frequency profile"""
        h, w = spectrum.shape
        center = (h // 2, w // 2)
        
        # Create radial bins
        max_radius = min(center)
        radial_profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)
        
        for i in range(h):
            for j in range(w):
                radius = int(np.sqrt((i - center[0])**2 + (j - center[1])**2))
                if radius < max_radius:
                    radial_profile[radius] += spectrum[i, j]
                    counts[radius] += 1
        
        # Average
        radial_profile = radial_profile / (counts + 1e-8)
        
        return radial_profile
    
    def _compute_band_energies(self, dct_coefficients: np.ndarray) -> List[float]:
        """Compute energy in frequency bands"""
        h, w = dct_coefficients.shape[:2]
        
        # Define frequency bands (simplified)
        low_mask = self._create_band_mask(h, w, 0, 0.3)
        mid_mask = self._create_band_mask(h, w, 0.3, 0.7)
        high_mask = self._create_band_mask(h, w, 0.7, 1.0)
        
        low_energy = np.sum(np.abs(dct_coefficients * low_mask[..., np.newaxis, np.newaxis])**2)
        mid_energy = np.sum(np.abs(dct_coefficients * mid_mask[..., np.newaxis, np.newaxis])**2)
        high_energy = np.sum(np.abs(dct_coefficients * high_mask[..., np.newaxis, np.newaxis])**2)
        
        total_energy = low_energy + mid_energy + high_energy + 1e-8
        
        return [low_energy/total_energy, mid_energy/total_energy, high_energy/total_energy]
    
    def _create_band_mask(self, h: int, w: int, low: float, high: float) -> np.ndarray:
        """Create frequency band mask"""
        fx = np.fft.fftfreq(h)[:, np.newaxis]
        fy = np.fft.fftfreq(w)[np.newaxis, :]
        freq_magnitude = np.sqrt(fx**2 + fy**2)
        
        mask = ((freq_magnitude >= low) & (freq_magnitude < high)).astype(np.float32)
        return mask