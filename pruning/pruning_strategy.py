import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Set, Callable, Any
from dataclasses import dataclass
import numpy as np 
import torch
import torch.nn as nn
import copy 

@dataclass
class LayerPruningConfig:
    """Configuration for pruning a specific layer"""
    layer_name: str
    original_filters: int
    filters_to_keep: int
    pruning_ratio: float
    importance_scores: np.ndarray
    mask: np.ndarray
class PruningStrategy:
    """
    Implements various pruning strategies for HSGSP
    """
   
    def __init__(self, config):
        self.config = config
        self.min_filters_per_layer = int(getattr(config, "hybrid_min_filters", 8))
        self.pruning_schedule = self._create_pruning_schedule()
   
    def _create_pruning_schedule(self) -> Dict[str, float]:
        """
        Create layer-wise pruning schedule
        Different layers may have different sensitivity to pruning
        """
        return {
            'early': 0.8, # Keep 80% in early layers (more important)
            'middle': 0.6, # Keep 60% in middle layers
            'late': 0.5, # Keep 50% in late layers (can prune more)
        }
   
    def compute_layer_importance(self,
                                layer_name: str,
                                layer_position: float) -> float:
        """
        Compute importance multiplier for a layer based on its position
       
        Args:
            layer_name: Name of the layer
            layer_position: Normalized position in network (0=first, 1=last)
           
        Returns:
            Importance multiplier (higher = more important)
        """
        if layer_position < 0.3:
            return self.pruning_schedule['early']
        elif layer_position < 0.7:
            return self.pruning_schedule['middle']
        else:
            return self.pruning_schedule['late']
   
    def select_filters_structured(self,
                                 importance_scores: Dict[str, np.ndarray],
                                 target_pruning_ratio: float,
                                 model: torch.nn.Module) -> Dict[str, LayerPruningConfig]:
        """
        Select filters to prune using structured pruning
       
        Args:
            importance_scores: Filter importance scores per layer
            target_pruning_ratio: Target pruning ratio
            model: Model to prune
           
        Returns:
            Pruning configuration for each layer
        """
        pruning_configs = {}
        total_filters = 0
        total_to_prune = 0
       
        # Get total layer count for position calculation
        conv_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        num_layers = len(conv_layers)
       
        for idx, layer_name in enumerate(conv_layers):
            if layer_name not in importance_scores:
                continue
           
            scores = importance_scores[layer_name]
            num_filters = len(scores)
            total_filters += num_filters
           
            # Compute layer-specific pruning ratio
            layer_position = idx / max(num_layers - 1, 1)
            layer_importance = self.compute_layer_importance(
                layer_name, layer_position
            )
           
            # Adjust pruning ratio based on layer importance
            adjusted_ratio = target_pruning_ratio * (2 - layer_importance)
            adjusted_ratio = np.clip(adjusted_ratio, 0, 0.9)
           
            # Calculate filters to keep
            filters_to_keep = max(
                int(num_filters * (1 - adjusted_ratio)),
                self.min_filters_per_layer
            )
            filters_to_prune = num_filters - filters_to_keep
            total_to_prune += filters_to_prune
           
            # Create pruning mask
            if filters_to_prune > 0:
                # Sort filters by importance
                sorted_indices = np.argsort(scores)
               
                # Create mask (True = keep, False = prune)
                mask = np.zeros(num_filters, dtype=bool)
                keep_indices = sorted_indices[-filters_to_keep:]
                mask[keep_indices] = True
            else:
                mask = np.ones(num_filters, dtype=bool)
           
            config = LayerPruningConfig(
                layer_name=layer_name,
                original_filters=num_filters,
                filters_to_keep=filters_to_keep,
                pruning_ratio=filters_to_prune / num_filters,
                importance_scores=scores,
                mask=mask
            )
           
            pruning_configs[layer_name] = config
       
        # Log pruning statistics
        print(f"Structured Pruning: {total_to_prune}/{total_filters} filters "
              f"({total_to_prune / max(total_filters, 1):.1%})")
       
        return pruning_configs
   
    def select_filters_unstructured(self,
                                   importance_scores: Dict[str, np.ndarray],
                                   target_sparsity: float) -> Dict[str, np.ndarray]:
        """
        Select weights to prune using unstructured pruning (weight-level)
       
        Args:
            importance_scores: Weight importance scores
            target_sparsity: Target sparsity level
           
        Returns:
            Binary masks for each layer
        """
        masks = {}
       
        # Collect all scores
        all_scores = []
        layer_info = []
       
        for layer_name, scores in importance_scores.items():
            flat_scores = scores.flatten()
            all_scores.extend(flat_scores)
            layer_info.extend([(layer_name, i) for i in range(len(flat_scores))])
       
        # Compute global threshold
        all_scores = np.array(all_scores)
        threshold = np.percentile(all_scores, target_sparsity * 100)
       
        # Create masks based on threshold
        for layer_name, scores in importance_scores.items():
            mask = scores > threshold
            masks[layer_name] = mask
       
        return masks
   
    def apply_structured_pruning(self,
                                model: torch.nn.Module,
                                pruning_configs: Dict[str, LayerPruningConfig]) -> torch.nn.Module:
        """
        Apply structured pruning in-place with channel-consistency across layers.
        """
        modules = dict(model.named_modules())
        conv_names = [name for name, m in modules.items() if isinstance(m, nn.Conv2d)]
        previous_conv_name = None
        for i, name in enumerate(conv_names):
            layer = modules[name]
            config = pruning_configs.get(name)
            if config is None:
                previous_conv_name = name
                continue
            out_mask = torch.from_numpy(config.mask).to(layer.weight.device)
            keep_out = out_mask.sum().item()
            # Slice output channels
            layer.weight = nn.Parameter(layer.weight[out_mask])
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias[out_mask])
            layer.out_channels = keep_out
            # Slice next BN if exists
            bn_name = name.replace('conv', 'bn')
            bn = modules.get(bn_name)
            if isinstance(bn, nn.BatchNorm2d):
                bn.weight = nn.Parameter(bn.weight[out_mask])
                bn.bias = nn.Parameter(bn.bias[out_mask])
                bn.running_mean = bn.running_mean[out_mask]
                bn.running_var = bn.running_var[out_mask]
                bn.num_features = keep_out
            # Slice input of previous Conv's output if exists
            if previous_conv_name is not None:
                prev_config = pruning_configs.get(previous_conv_name)
                if prev_config is not None:
                    prev_out_mask = torch.from_numpy(prev_config.mask).to(layer.weight.device)
                    layer.weight = nn.Parameter(layer.weight[:, prev_out_mask, :, :])
                    layer.in_channels = prev_out_mask.sum().item()
            previous_conv_name = name
        # Adjust classifier input based on last conv's out_channels
        if conv_names:
            last_name = conv_names[-1]
            last_layer = modules[last_name]
            last_channels = last_layer.out_channels
            # Assuming classifier.0 is the first Linear
            first_linear_name = 'classifier.0'
            first_linear = modules.get(first_linear_name)
            if isinstance(first_linear, nn.Linear):
                # Input is last_channels (after avgpool 1x1 and flatten)
                first_linear.weight = nn.Parameter(first_linear.weight[:, :last_channels])
                first_linear.in_features = last_channels
        return model
   
    def prune_model_structured(self,
                               model: torch.nn.Module,
                               layer_pruning_ratios: Dict[str, float],
                               importance_scores: Dict[str, np.ndarray]) -> Tuple[torch.nn.Module, Dict[str, LayerPruningConfig]]:
        """
        Convenience API: build per-layer configs from provided ratios and scores,
        then apply structured pruning consistently.
        Args:
            model: model to prune
            layer_pruning_ratios: dict layer_name -> ratio in [0,1]
            importance_scores: dict layer_name -> per-filter importance scores
        Returns:
            (pruned_model, pruning_configs)
        """
        pruning_configs: Dict[str, LayerPruningConfig] = {}
        conv_dict = {name: m for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}
        for name, layer in conv_dict.items():
            if name not in importance_scores:
                continue
            scores = importance_scores[name]
            num_filters = len(scores)
            ratio = float(layer_pruning_ratios.get(name, 0.0))
            ratio = float(np.clip(ratio, 0.0, 0.95))
            keep = max(1, int(round(num_filters * (1.0 - ratio))))
            # select top-k by importance
            idx_sorted = np.argsort(scores)
            keep_idx = idx_sorted[-keep:]
            mask = np.zeros(num_filters, dtype=bool)
            mask[keep_idx] = True
            pruning_configs[name] = LayerPruningConfig(
                layer_name=name,
                original_filters=num_filters,
                filters_to_keep=keep,
                pruning_ratio=(num_filters - keep) / max(num_filters, 1),
                importance_scores=scores,
                mask=mask
            )
        pruned_model = self.apply_structured_pruning(model, pruning_configs)
        return pruned_model, pruning_configs
   
    def compute_pruning_sensitivity(self,
                                   model: torch.nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   layer_names: List[str],
                                   sample_ratios: List[float] = [0.1, 0.3, 0.5, 0.7]) -> Dict:
        """
        Analyze sensitivity of each layer to pruning
       
        Args:
            model: Model to analyze
            dataloader: Validation dataloader
            layer_names: Layers to test
            sample_ratios: Pruning ratios to test
           
        Returns:
            Sensitivity analysis results
        """
        sensitivity_results = {}
       
        # Get baseline accuracy
        baseline_acc = self._evaluate_accuracy(model, dataloader)
       
        for layer_name in layer_names:
            layer_sensitivity = []
           
            for ratio in sample_ratios:
                # Create temporary pruned model
                temp_model = copy.deepcopy(model)
               
                # Prune single layer
                for name, module in temp_model.named_modules():
                    if name == layer_name and isinstance(module, nn.Conv2d):
                        # Simple magnitude-based pruning for testing
                        w = module.weight.data
                        threshold = np.percentile(np.abs(w.cpu().numpy()), ratio * 100)
                        mask = torch.abs(w) > threshold
                        module.weight.data = w * mask.float().to(w.device)
               
                # Evaluate accuracy drop
                pruned_acc = self._evaluate_accuracy(temp_model, dataloader)
                accuracy_drop = baseline_acc - pruned_acc
               
                layer_sensitivity.append({
                    'pruning_ratio': ratio,
                    'accuracy_drop': accuracy_drop
                })
           
            sensitivity_results[layer_name] = layer_sensitivity
       
        return sensitivity_results
   
    def _evaluate_accuracy(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
        """Quick accuracy evaluation"""
        correct = 0
        total = 0
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                predictions = model(x_batch)
                predicted_classes = torch.argmax(predictions, dim=1)
                true_classes = torch.argmax(y_batch, dim=1).to(device) if y_batch.dim() > 1 else y_batch.to(device)
                correct += (predicted_classes == true_classes).sum().item()
                total += x_batch.size(0)
       
        return correct / max(total, 1)
   
    def iterative_pruning(self,
                         model: torch.nn.Module,
                         importance_scores: Dict[str, np.ndarray],
                         target_ratio: float,
                         num_iterations: int = 5) -> torch.nn.Module:
        """
        Apply pruning iteratively (gradual pruning)
       
        Args:
            model: Model to prune
            importance_scores: Filter importance scores
            target_ratio: Final target pruning ratio
            num_iterations: Number of pruning iterations
           
        Returns:
            Iteratively pruned model
        """
        current_model = model
        ratio_per_iteration = target_ratio / num_iterations
       
        for iteration in range(num_iterations):
            print(f"Pruning iteration {iteration + 1}/{num_iterations}")
           
            # Compute current pruning ratio
            current_ratio = ratio_per_iteration * (iteration + 1)
           
            # Apply pruning
            pruning_configs = self.select_filters_structured(
                importance_scores,
                ratio_per_iteration,
                current_model
            )
           
            current_model = self.apply_structured_pruning(
                current_model,
                pruning_configs
            )
           
            # Optional: Fine-tune between iterations
            # This would require access to training data
       
        return current_model
    def iterative_prune_and_regrow(self,
                                   model: torch.nn.Module,
                                   base_layer_pruning_ratios: Dict[str, float],
                                   compute_importance_scores: Callable[[torch.nn.Module], Dict[str, np.ndarray]],
                                   iterations: int = 3,
                                   regrowth_fn: Optional[Callable[[torch.nn.Module,
                                                                   int,
                                                                   Dict[str, LayerPruningConfig]], Any]] = None,
                                   regrowth_kwargs: Optional[Dict[str, Any]] = None,
                                   initial_importance_scores: Optional[Dict[str, np.ndarray]] = None,
                                   recompute_scores: bool = True,
                                   logger: Optional[Callable[[str], None]] = None
                                   ) -> Tuple[torch.nn.Module, List[Dict[str, Any]]]:
        """Iteratively prune the model and trigger regrowth between stages.
        Args:
            model: Model to prune.
            base_layer_pruning_ratios: Final per-layer pruning ratios (0-1).
            compute_importance_scores: Callable that returns importance scores for
                the current model (used before each pruning stage).
            iterations: Number of prune/regrowth stages.
            regrowth_fn: Optional callable applied after each pruning stage. It
                should accept `(model, iteration, pruning_configs, **kwargs)` and
                return either the updated model or `(model, metrics)`.
            regrowth_kwargs: Extra keyword arguments forwarded to `regrowth_fn`.
            initial_importance_scores: Pre-computed importance scores for the
                first iteration (saves one recomputation).
            recompute_scores: Whether to re-run `compute_importance_scores` before
                every pruning step. Set `False` to reuse the previous scores when
                the architecture does not change.
            logger: Optional logging function (defaults to `print`).
        Returns:
            Tuple containing the final pruned (and regrown) model and a list of
            per-iteration records (pruning summary, optional regrowth metrics,
            etc.).
        """
        if iterations <= 0:
            raise ValueError("iterations must be >= 1")
        log = logger or (lambda msg: print(msg))
        regrowth_kwargs = regrowth_kwargs or {}
        # Precompute per-layer stage keep factors so the product reaches the final
        # target ratio after all iterations.
        per_layer_stage_keep = {}
        for layer_name, final_ratio in base_layer_pruning_ratios.items():
            final_ratio = float(np.clip(final_ratio, 0.0, 0.95))
            final_keep = max(1e-6, 1.0 - final_ratio)
            stage_keep = final_keep ** (1.0 / iterations)
            per_layer_stage_keep[layer_name] = stage_keep
        cumulative_keep = {layer: 1.0 for layer in base_layer_pruning_ratios}
        iteration_records: List[Dict[str, Any]] = []
        current_model = model
        importance_scores = initial_importance_scores
        for iteration in range(iterations):
            if importance_scores is None or recompute_scores or iteration == 0:
                importance_scores = compute_importance_scores(current_model)
            stage_pruning_ratios: Dict[str, float] = {}
            for layer_name, stage_keep in per_layer_stage_keep.items():
                if iteration == iterations - 1:
                    # Adjust final step to hit the requested final ratio.
                    target_final_keep = max(1e-6, 1.0 - base_layer_pruning_ratios[layer_name])
                    current_keep = max(1e-6, cumulative_keep.get(layer_name, 1.0))
                    # Need to prune so that (current_keep * keep_next) == target_final_keep
                    keep_needed = min(current_keep, target_final_keep) / current_keep
                    stage_ratio = 1.0 - keep_needed
                else:
                    stage_ratio = 1.0 - stage_keep
                stage_pruning_ratios[layer_name] = float(np.clip(stage_ratio, 0.0, 0.95))
            log(f"[Iterative Prune] Stage {iteration + 1}/{iterations} -> applying per-layer ratios")
            current_model, pruning_configs = self.prune_model_structured(
                current_model,
                layer_pruning_ratios=stage_pruning_ratios,
                importance_scores=importance_scores
            )
            # Assuming a summary method or something similar
            summary = self.get_pruning_summary(pruning_configs)
            log(f"[Iterative Prune] Stage {iteration + 1} summary: "
                f"overall ratio={summary['overall_pruning_ratio']:.3f}")
            # Update cumulative keep ratios for next iteration.
            for layer_name, config in pruning_configs.items():
                keep_fraction = config.filters_to_keep / max(config.original_filters, 1)
                cumulative_keep[layer_name] = cumulative_keep.get(layer_name, 1.0) * keep_fraction
            stage_record: Dict[str, Any] = {
                'iteration': iteration,
                'pruning_configs': pruning_configs,
                'summary': summary
            }
            if regrowth_fn is not None:
                log(f"[Iterative Prune] Starting regrowth for stage {iteration + 1}")
                regrowth_result = regrowth_fn(
                    current_model,
                    iteration,
                    pruning_configs,
                    **regrowth_kwargs
                )
                if isinstance(regrowth_result, tuple):
                    current_model = regrowth_result[0]
                    stage_record['regrowth_metrics'] = regrowth_result[1]
                else:
                    current_model = regrowth_result
            iteration_records.append(stage_record)
            # Force recomputation on the next iteration if requested.
            if recompute_scores:
                importance_scores = None
        return current_model, iteration_records
   
    def get_pruning_summary(self,
                          pruning_configs: Dict[str, LayerPruningConfig]) -> Dict:
        """
        Generate summary statistics for pruning
       
        Args:
            pruning_configs: Pruning configurations
           
        Returns:
            Summary statistics
        """
        total_original = 0
        total_kept = 0
        layer_stats = []
       
        for layer_name, config in pruning_configs.items():
            total_original += config.original_filters
            total_kept += config.filters_to_keep
           
            layer_stats.append({
                'layer': layer_name,
                'original': config.original_filters,
                'kept': config.filters_to_keep,
                'pruned': config.original_filters - config.filters_to_keep,
                'ratio': config.pruning_ratio
            })
       
        return {
            'total_filters_original': total_original,
            'total_filters_kept': total_kept,
            'total_filters_pruned': total_original - total_kept,
            'overall_pruning_ratio': (total_original - total_kept) / max(total_original, 1),
            'layer_statistics': layer_stats
        }